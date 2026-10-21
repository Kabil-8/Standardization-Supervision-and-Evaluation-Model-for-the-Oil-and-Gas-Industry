"""Standalone TECR-Net inference (no training, no Optuna, no Kaggle paths).

Reproduces the archived seven-seed ensemble from:
  * the platform-year panel CSV (bsee_platform_year_model.csv[.gz]);
  * sequence_index.csv   (row indices of the t-2, t-1, t windows);
  * preprocessing.json   (fitting-period clip/median/scale + category vocabularies);
  * tecrnet_seed_{seed}.pt checkpoints (weights, per-seed Platt coefficients).

Score definition (identical to the archived run):
  neural_p  = sigmoid(clip(logit, -30, 30))
  seed_p    = sigmoid(coef * neural_p + intercept)      # Platt-like, probability input
  ensemble  = mean over seeds of seed_p
Ensemble threshold is NOT stored in the checkpoints; it comes from frozen_config.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

SEEDS = [11, 22, 33, 44, 55, 66, 77]
TARGET = "next_year_any_inc"
SEQ_COLS = ["row_t_minus_2", "row_t_minus_1", "row_t"]


# ----------------------------------------------------------------- data
def load_panel(path: str | Path) -> pd.DataFrame:
    """Read the panel and apply the same ordering used when sequence_index.csv was built."""
    frame = pd.read_csv(path, low_memory=False)
    if frame.duplicated(["platform_id", "year"]).any():
        raise ValueError("Duplicate platform-year rows in panel.")
    frame = frame.sort_values(["platform_id", "year"]).reset_index(drop=True)
    return frame


class FrozenNumeric:
    """Applies stored fitting-period clipping, median imputation and robust scaling."""

    def __init__(self, state: dict):
        for k in ("lower", "upper", "median", "center", "scale"):
            setattr(self, k, np.asarray(state[k], dtype=np.float64))
        self.all_missing = np.asarray(state["all_missing"], dtype=bool)

    def transform(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64).copy()
        v[~np.isfinite(v)] = np.nan
        v = np.clip(v, self.lower, self.upper)
        v = np.where(np.isnan(v), self.median, v)
        out = (v - self.center) / self.scale
        out[:, self.all_missing] = 0.0
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_arrays(frame: pd.DataFrame, index: pd.DataFrame, prep: dict) -> dict:
    dyn_cols, st_cols, cat_cols = prep["dynamic_columns"], prep["static_columns"], prep["categorical_columns"]
    dyn, sta = FrozenNumeric(prep["dynamic_processor"]), FrozenNumeric(prep["static_processor"])
    rows = index[SEQ_COLS].to_numpy(dtype=np.int64)
    num = lambda r, c: frame.loc[r, c].apply(pd.to_numeric, errors="coerce").to_numpy()
    dynamic = np.stack([dyn.transform(num(r, dyn_cols)) for r in rows])
    cur = index["row_t"].to_numpy(dtype=np.int64)
    static = sta.transform(num(cur, st_cols))
    cat = np.zeros((len(cur), len(cat_cols)), dtype=np.int64)
    for j, c in enumerate(cat_cols):
        vals = frame.loc[cur, c].fillna("Unknown").astype(str)
        cat[:, j] = vals.map(prep["category_vocabularies"][c]).fillna(0).astype(np.int64)
    return {"dynamic": dynamic, "static": static, "categorical": cat}


# ---------------------------------------------------------------- model
class TECRNet(nn.Module):
    def __init__(self, n_dynamic, n_static, cardinalities, hidden=80, dropout=0.269,
                 gru_layers=1, attention_heads=8):
        super().__init__()
        self.feature_logits = nn.Parameter(torch.zeros(n_dynamic))
        self.temporal = nn.GRU(n_dynamic, hidden, gru_layers, batch_first=True, bidirectional=True,
                               dropout=dropout if gru_layers > 1 else 0.0)
        self.temporal_projection = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.temporal_score = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.static_encoder = nn.Sequential(nn.Linear(n_static, hidden), nn.GELU(), nn.Dropout(dropout),
                                            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        dims = [min(16, max(3, int(round(math.sqrt(c))))) for c in cardinalities]
        self.embeddings = nn.ModuleList([nn.Embedding(c, d, padding_idx=0) for c, d in zip(cardinalities, dims)])
        self.category_encoder = nn.Sequential(nn.Linear(sum(dims), hidden), nn.GELU(), nn.Dropout(dropout),
                                              nn.LayerNorm(hidden))
        self.modality_attention = nn.MultiheadAttention(hidden, attention_heads, dropout=dropout, batch_first=True)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden))
        self.classifier = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(hidden, 1))

    def forward(self, dynamic, static, categorical):
        seq, _ = self.temporal(dynamic * torch.sigmoid(self.feature_logits))
        w = torch.softmax(self.temporal_score(seq).squeeze(-1), dim=1)
        temporal = self.temporal_projection((seq * w.unsqueeze(-1)).sum(dim=1))
        static_tok = self.static_encoder(static)
        emb = torch.cat([e(categorical[:, i]) for i, e in enumerate(self.embeddings)], dim=1)
        tokens = torch.stack([temporal, static_tok, self.category_encoder(emb)], dim=1)
        attended, attn = self.modality_attention(self.query.expand(dynamic.size(0), -1, -1), tokens, tokens,
                                                 need_weights=True, average_attn_weights=True)
        fused = torch.cat([attended.squeeze(1), tokens.mean(dim=1)], dim=1)
        return self.classifier(fused).squeeze(1), attn.squeeze(1)


def load_checkpoint(path: Path, prep: dict, device="cpu") -> tuple[TECRNet, dict]:
    ck = torch.load(path, map_location=device, weights_only=True)
    p = ck["neural_params"]
    model = TECRNet(len(prep["dynamic_columns"]), len(prep["static_columns"]), prep["cardinalities"],
                    hidden=int(p["hidden"]), dropout=float(p["dropout"]),
                    gru_layers=int(p["gru_layers"]), attention_heads=int(p["attention_heads"]))
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval(), ck


@torch.no_grad()
def predict_seed(model: TECRNet, arrays: dict, device="cpu", batch=512):
    logits, attn = [], []
    n = len(arrays["dynamic"])
    for i in range(0, n, batch):
        s = slice(i, i + batch)
        out, a = model(torch.from_numpy(arrays["dynamic"][s]).to(device),
                       torch.from_numpy(arrays["static"][s]).to(device),
                       torch.from_numpy(arrays["categorical"][s]).to(device))
        logits.append(out.cpu().numpy()); attn.append(a.cpu().numpy())
    return np.concatenate(logits), np.concatenate(attn)


def score(arrays: dict, prep: dict, checkpoint_dir: str | Path, device="cpu") -> pd.DataFrame:
    """Return per-seed neural/calibrated probabilities and the ensemble probability."""
    out = {}
    cal = []
    for seed in SEEDS:
        model, ck = load_checkpoint(Path(checkpoint_dir) / f"tecrnet_seed_{seed}.pt", prep, device)
        logit, _ = predict_seed(model, arrays, device)
        neural = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
        coef, icpt = float(np.ravel(ck["calibrator_coef"])[0]), float(np.ravel(ck["calibrator_intercept"])[0])
        p = 1 / (1 + np.exp(-(coef * neural + icpt)))
        out[f"neural_probability_seed_{seed}"] = neural
        out[f"probability_seed_{seed}"] = p
        cal.append(p)
    out["ensemble_probability"] = np.mean(np.column_stack(cal), axis=1)
    return pd.DataFrame(out)


def f1_max_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Same rule as the archived select_threshold (exact grid; <=2000 unique scores)."""
    from sklearn.metrics import f1_score
    cand = np.unique(p)
    if len(cand) > 2000:
        cand = np.quantile(cand, np.linspace(0, 1, 2000))
    return float(cand[int(np.argmax([f1_score(y, p >= t, zero_division=0) for t in cand]))])
