import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


class TECRNet(nn.Module):
    def __init__(self, n_dynamic, n_static, cardinalities, hidden=80, dropout=.269,
                 gru_layers=1, attention_heads=8):
        super().__init__()
        self.feature_logits = nn.Parameter(torch.zeros(n_dynamic))
        self.temporal = nn.GRU(n_dynamic, hidden, gru_layers, batch_first=True,
                               bidirectional=True,
                               dropout=dropout if gru_layers > 1 else 0.0)
        self.nonrecurrent_temporal = nn.Linear(n_dynamic, hidden * 2)
        self.temporal_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.temporal_score = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.static_encoder = nn.Sequential(
            nn.Linear(n_static, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        dims = [min(16, max(3, int(round(math.sqrt(c))))) for c in cardinalities]
        self.embeddings = nn.ModuleList([
            nn.Embedding(c, d, padding_idx=0) for c, d in zip(cardinalities, dims)])
        self.category_encoder = nn.Sequential(
            nn.Linear(sum(dims), hidden), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(hidden))
        self.modality_attention = nn.MultiheadAttention(
            hidden, attention_heads, dropout=dropout, batch_first=True)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden))
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, dynamic, static, categorical):
        gated = dynamic * torch.sigmoid(self.feature_logits)
        seq, _ = self.temporal(gated)
        time_weights = torch.softmax(self.temporal_score(seq).squeeze(-1), dim=1)
        temporal = torch.sum(seq * time_weights.unsqueeze(-1), dim=1)
        temporal = self.temporal_projection(temporal)
        static_token = self.static_encoder(static)
        embedded = torch.cat([
            emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        category_token = self.category_encoder(embedded)
        modalities = torch.stack([temporal, static_token, category_token], dim=1)
        query = self.query.expand(dynamic.size(0), -1, -1)
        attended, modal_weights = self.modality_attention(
            query, modalities, modalities, need_weights=True, average_attn_weights=True)
        fused = torch.cat([attended.squeeze(1), modalities.mean(dim=1)], dim=1)
        return self.classifier(fused).squeeze(1), time_weights, modal_weights.squeeze(1)


class Predictor:
    def __init__(self, artifact_dir):
        self.root = Path(artifact_dir)
        self.preprocessing = json.loads((self.root / "preprocessing.json").read_text())
        self.params = json.loads((self.root / "best_params.json").read_text())
        self.dynamic = self.preprocessing["dynamic_columns"]
        self.static = self.preprocessing["static_columns"]
        self.categorical = self.preprocessing["categorical_columns"]
        self.models = []
        self.thresholds = []
        self.seeds = []
        for path in sorted(self.root.glob("tecrnet_seed_*.pt")):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = TECRNet(len(self.dynamic), len(self.static),
                            self.preprocessing["cardinalities"], **self._network_params())
            # The archived full-model checkpoints omit the ablation-only
            # nonrecurrent projection; it is never used during full inference.
            model.load_state_dict(checkpoint["model_state"], strict=False)
            model.eval()
            coef = float(np.asarray(checkpoint["calibrator_coef"]).reshape(-1)[0])
            intercept = float(np.asarray(checkpoint["calibrator_intercept"]).reshape(-1)[0])
            self.models.append((model, coef, intercept))
            self.thresholds.append(float(checkpoint["threshold"]))
            self.seeds.append(int(checkpoint["seed"]))
        if not self.models:
            raise RuntimeError("No TECR-Net checkpoints found")
        self.manifest_id = "eeec8023"
        self.archived_ensemble_threshold = 0.04908876769816646
        self.legacy_seed_mean_threshold = float(np.mean(self.thresholds))
        self.symmetric_threshold = 0.07110931251823678
        self.threshold = self.archived_ensemble_threshold

    def _network_params(self):
        return {k: self.params[k] for k in
                ["hidden", "dropout", "gru_layers", "attention_heads"]}

    @staticmethod
    def _transform(values, spec):
        x = np.asarray(values, dtype=np.float32)
        median = np.asarray(spec["median"], dtype=np.float32)
        x = np.where(np.isfinite(x), x, median)
        x = np.clip(x, np.asarray(spec["lower"], dtype=np.float32),
                    np.asarray(spec["upper"], dtype=np.float32))
        return (x - np.asarray(spec["center"], dtype=np.float32)) / np.asarray(spec["scale"], dtype=np.float32)

    def defaults(self):
        dyn = dict(zip(self.dynamic, self.preprocessing["dynamic_processor"]["median"]))
        sta = dict(zip(self.static, self.preprocessing["static_processor"]["median"]))
        cat = {name: next(iter(vocab)) for name, vocab in self.preprocessing["category_vocabularies"].items()}
        return {"years": [dyn.copy(), dyn.copy(), dyn.copy()], "static": sta, "categorical": cat}

    def schema(self):
        return {
            "dynamic": self.dynamic,
            "static": self.static,
            "categories": {k: list(v.keys()) for k, v in self.preprocessing["category_vocabularies"].items()},
            "defaults": self.defaults(),
            "threshold": self.threshold,
            "archived_ensemble_threshold": self.archived_ensemble_threshold,
            "symmetric_threshold": self.symmetric_threshold,
            "legacy_seed_mean_threshold": self.legacy_seed_mean_threshold,
            "seeds": self.seeds,
        }

    def predict(self, payload):
        years = payload.get("years", [])
        if len(years) != 3:
            raise ValueError("Exactly three consecutive feature years are required")
        d_raw = [[float(year.get(c, np.nan)) for c in self.dynamic] for year in years]
        d = self._transform(d_raw, self.preprocessing["dynamic_processor"])
        s_raw = [float(payload.get("static", {}).get(c, np.nan)) for c in self.static]
        s = self._transform(s_raw, self.preprocessing["static_processor"])
        cat_payload = payload.get("categorical", {})
        cats = []
        unknown = []
        for c in self.categorical:
            value = str(cat_payload.get(c, "Unknown"))
            vocab = self.preprocessing["category_vocabularies"][c]
            code = int(vocab.get(value, 0))
            if code == 0:
                unknown.append(c)
            cats.append(code)
        dt = torch.tensor(d[None, :, :], dtype=torch.float32)
        st = torch.tensor(s[None, :], dtype=torch.float32)
        ct = torch.tensor([cats], dtype=torch.long)
        calibrated, neural, temporal, modal = [], [], [], []
        with torch.no_grad():
            for model, coef, intercept in self.models:
                logits, tw, mw = model(dt, st, ct)
                p = torch.sigmoid(logits).item()
                cp = 1.0 / (1.0 + math.exp(-(coef * p + intercept)))
                neural.append(p); calibrated.append(cp)
                temporal.append(tw.numpy()[0]); modal.append(mw.numpy()[0])
        probability = float(np.mean(calibrated))
        gates = np.mean([torch.sigmoid(m.feature_logits).detach().numpy() for m,_,_ in self.models], axis=0)
        top = sorted(zip(self.dynamic, gates.tolist()), key=lambda x: x[1], reverse=True)[:8]
        return {
            "probability": probability,
            "percentage": probability * 100,
            "threshold": self.threshold,
            "archived_ensemble_threshold": self.archived_ensemble_threshold,
            "symmetric_threshold": self.symmetric_threshold,
            "legacy_seed_mean_threshold": self.legacy_seed_mean_threshold,
            "classification": "Elevated priority" if probability >= self.threshold else "Routine priority",
            "risk_ratio_to_threshold": probability / self.threshold if self.threshold else None,
            "seed_probabilities": [{"seed": s, "probability": p} for s,p in zip(self.seeds, calibrated)],
            "seed_mean": probability,
            "seed_sd": float(np.std(calibrated, ddof=1)),
            "temporal_attention": dict(zip(["t-2", "t-1", "t"], np.mean(temporal, axis=0).tolist())),
            "modality_attention": dict(zip(["Temporal", "Static", "Categorical"], np.mean(modal, axis=0).tolist())),
            "top_global_feature_gates": [{"feature": n, "weight": w} for n,w in top],
            "unknown_categories": unknown,
            "notice": "Research prioritization score for recorded INC risk; not an enforcement decision or probability validated prospectively."
        }
