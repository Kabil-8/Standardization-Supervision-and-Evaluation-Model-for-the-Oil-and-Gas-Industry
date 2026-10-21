"""
TECR-Net: Simplified Empirically Motivated Architecture & Strict Benchmark
Protocol:
- Fitting Split: 2014-2017 (9,449 sequences)
- Architecture Selection: 2018 (2,009 sequences) using 5 seeds (11, 22, 33, 44, 55)
- Platt Calibration: 2019 (1,875 sequences) using averaged raw logits
- Threshold Selection: 2020 (1,790 sequences) F1-maximizing
- Development/Exploratory Evaluation: 2021-2024 (6,492 sequences)
Deliverables:
- Saved to reproducible_experiment_package/
"""

import sys, os, time, math, json, logging, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, fbeta_score,
    precision_score, recall_score, balanced_accuracy_score, matthews_corrcoef,
    brier_score_loss, log_loss, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Setup Logging
LOG_FILE = Path("scratch/simplified_tecrnet.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)]
)
LOG = logging.getLogger("SimplifiedTECRNet")

OUT_DIR = Path("reproducible_experiment_package")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = OUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 22, 33, 44, 55]

# -----------------------------------------------------------------------------
# ARCHITECTURES & LOSS FUNCTIONS
# -----------------------------------------------------------------------------
class AlphaBalancedFocalLoss(nn.Module):
    """Binary alpha-balanced focal loss operating directly on logits."""
    def __init__(self, alpha: float = 0.75, gamma: float = 1.5, reduction: str = "mean"):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if gamma < 0.0:
            raise ValueError("gamma must be non-negative")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        probabilities = torch.sigmoid(logits)
        positive_probability = torch.where(targets.eq(1), probabilities, 1.0 - probabilities)
        alpha_weight = torch.where(targets.eq(1), torch.full_like(targets, self.alpha), torch.full_like(targets, 1.0 - self.alpha))
        binary_cross_entropy = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = alpha_weight * (1.0 - positive_probability).pow(self.gamma) * binary_cross_entropy
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

class SimplifiedTECRNet(nn.Module):
    """Simplified TECR-Net for three-year platform histories."""
    def __init__(
        self,
        n_dynamic: int,
        n_static: int,
        categorical_cardinalities,
        hidden_size: int = 64,
        token_size: int = 64,
        dropout: float = 0.25,
        bidirectional: bool = True,
        use_temp_att: bool = True,
        use_residual: bool = True,
        disable_static: bool = False,
        disable_categorical: bool = False
    ):
        super().__init__()
        self.n_dynamic = int(n_dynamic)
        self.n_static = int(n_static)
        self.hidden_size = int(hidden_size)
        self.token_size = int(token_size)
        self.bidirectional = bool(bidirectional)
        self.use_temp_att = bool(use_temp_att)
        self.use_residual = bool(use_residual)
        self.disable_static = bool(disable_static)
        self.disable_categorical = bool(disable_categorical)

        recurrent_size = hidden_size * (2 if bidirectional else 1)

        # Dynamic branch
        self.dynamic_normalization = nn.LayerNorm(n_dynamic)
        self.gru = nn.GRU(
            input_size=n_dynamic,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        if self.use_temp_att:
            self.temporal_attention = nn.Sequential(
                nn.Linear(recurrent_size, 32),
                nn.Tanh(),
                nn.Linear(32, 1, bias=False),
            )
        self.temporal_projection = nn.Sequential(
            nn.Linear(recurrent_size, token_size),
            nn.GELU(),
            nn.LayerNorm(token_size),
            nn.Dropout(dropout),
        )

        # Static branch
        if n_static > 0 and not self.disable_static:
            self.static_encoder = nn.Sequential(
                nn.Linear(n_static, 32),
                nn.GELU(),
                nn.LayerNorm(32),
                nn.Dropout(dropout),
                nn.Linear(32, token_size),
                nn.GELU(),
                nn.LayerNorm(token_size),
            )
        else:
            self.static_encoder = None
            self.static_fallback = nn.Parameter(torch.zeros(token_size))

        # Categorical branch
        self.embeddings = nn.ModuleList()
        embedding_dimensions = []
        for cardinality in categorical_cardinalities:
            cardinality = int(cardinality)
            embedding_dimension = min(16, max(3, int(round(1.6 * cardinality ** 0.56))))
            self.embeddings.append(
                nn.Embedding(num_embeddings=cardinality, embedding_dim=embedding_dimension, padding_idx=0)
            )
            embedding_dimensions.append(embedding_dimension)

        total_embedding_size = sum(embedding_dimensions)
        if total_embedding_size > 0 and not self.disable_categorical:
            self.categorical_encoder = nn.Sequential(
                nn.Linear(total_embedding_size, token_size),
                nn.GELU(),
                nn.LayerNorm(token_size),
                nn.Dropout(dropout),
            )
        else:
            self.categorical_encoder = None
            self.categorical_fallback = nn.Parameter(torch.zeros(token_size))

        # Fusion
        fusion_input_size = token_size * 3
        self.fusion_main = nn.Sequential(
            nn.Linear(fusion_input_size, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, token_size),
        )
        if self.use_residual:
            self.fusion_residual = nn.Linear(fusion_input_size, token_size, bias=False)
        self.fusion_normalization = nn.LayerNorm(token_size)

        # Classifier
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_size, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if parameter.ndim >= 2 and "embedding" not in name:
                nn.init.xavier_uniform_(parameter)
            elif parameter.ndim == 1 and "weight" not in name:
                nn.init.zeros_(parameter)
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            if embedding.padding_idx is not None:
                with torch.no_grad():
                    embedding.weight[embedding.padding_idx].zero_()

    def forward(self, x_dynamic, x_static, x_categorical, return_attention=False):
        dynamic_input = self.dynamic_normalization(x_dynamic)
        recurrent_output, _ = self.gru(dynamic_input)

        if self.use_temp_att:
            attention_logits = self.temporal_attention(recurrent_output).squeeze(-1)
            temporal_weights = torch.softmax(attention_logits, dim=1)
            temporal_summary = torch.sum(recurrent_output * temporal_weights.unsqueeze(-1), dim=1)
        else:
            temporal_weights = torch.ones(x_dynamic.size(0), 3, device=x_dynamic.device) / 3.0
            temporal_summary = recurrent_output.mean(dim=1)

        temporal_token = self.temporal_projection(temporal_summary)

        if self.static_encoder is not None:
            static_token = self.static_encoder(x_static)
        else:
            static_token = self.static_fallback.unsqueeze(0).expand(x_dynamic.size(0), -1)

        if self.categorical_encoder is not None:
            embedded = [embedding(x_categorical[:, index]) for index, embedding in enumerate(self.embeddings)]
            categorical_input = torch.cat(embedded, dim=1)
            categorical_token = self.categorical_encoder(categorical_input)
        else:
            categorical_token = self.categorical_fallback.unsqueeze(0).expand(x_dynamic.size(0), -1)

        fused_input = torch.cat([temporal_token, static_token, categorical_token], dim=1)
        fused_main = self.fusion_main(fused_input)
        if self.use_residual:
            fused_residual = self.fusion_residual(fused_input)
            fused = self.fusion_normalization(fused_main + fused_residual)
        else:
            fused = self.fusion_normalization(fused_main)

        logits = self.classifier(fused).squeeze(-1)
        if return_attention:
            return {
                "logits": logits,
                "temporal_attention": temporal_weights,
                "temporal_token": temporal_token,
                "static_token": static_token,
                "categorical_token": categorical_token,
            }
        return logits

class PlainBiGRU(nn.Module):
    def __init__(self, n_dynamic: int, hidden_size: int = 64, dropout: float = 0.25):
        super().__init__()
        self.gru = nn.GRU(n_dynamic, hidden_size, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x_d, x_s, x_c):
        h, _ = self.gru(x_d)
        c = h.mean(dim=1)
        return self.head(c).squeeze(-1)

class SimpleConcatMLP(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x_d, x_s, x_c):
        flat = torch.cat([x_d.flatten(1), x_s, x_c.float()], dim=1)
        return self.net(flat).squeeze(-1)

# Training loop
def train_model(model, X_d_tr, X_s_tr, X_c_tr, y_tr, X_d_val, X_s_val, X_c_val, y_val,
                loss_config, epochs=25, seed=11, lr=1e-3, batch_size=256):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    pos = max(1, int(y_tr.sum()))
    pos_weight = torch.tensor((len(y_tr) - pos) / pos, dtype=torch.float32)

    l_type = loss_config.get("loss", "weighted_bce")
    if l_type == "focal":
        criterion = AlphaBalancedFocalLoss(
            alpha=loss_config.get("alpha", 0.75),
            gamma=loss_config.get("gamma", 1.5)
        )
    elif l_type == "weighted_bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    ds = TensorDataset(torch.from_numpy(X_d_tr), torch.from_numpy(X_s_tr),
                       torch.from_numpy(X_c_tr), torch.from_numpy(y_tr.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_ap = -1.0
    best_state = None

    for ep in range(epochs):
        model.train()
        for xd_b, xs_b, xc_b, y_b in loader:
            opt.zero_grad()
            out = model(xd_b, xs_b, xc_b)
            loss = criterion(out, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_l = model(torch.from_numpy(X_d_val), torch.from_numpy(X_s_val), torch.from_numpy(X_c_val)).numpy()
            val_p = 1 / (1 + np.exp(-val_l))
            ap = average_precision_score(y_val, val_p)
            if ap > best_ap:
                best_ap = ap
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_ap

def calculate_all_metrics(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    prev = float(y_true.mean())
    ap = float(average_precision_score(y_true, prob))
    roc = float(roc_auc_score(y_true, prob))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    f2 = float(fbeta_score(y_true, pred, beta=2, zero_division=0))
    prec = float(precision_score(y_true, pred, zero_division=0))
    rec = float(recall_score(y_true, pred, zero_division=0))
    bacc = float(balanced_accuracy_score(y_true, pred))
    spec = float(tn / max(tn + fp, 1))
    mcc = float(matthews_corrcoef(y_true, pred))
    brier = float(brier_score_loss(y_true, prob))
    brier_skill = 1.0 - (brier / (prev * (1 - prev))) if prev * (1 - prev) > 0 else 0.0
    logl = float(log_loss(y_true, prob, labels=[0, 1]))

    eps = 1e-6
    clipped_p = np.clip(prob, eps, 1 - eps)
    logit_p = np.log(clipped_p / (1 - clipped_p))
    try:
        lr_cal = LogisticRegression(C=1e5, solver='lbfgs').fit(logit_p.reshape(-1, 1), y_true)
        cal_slope = float(lr_cal.coef_[0][0])
        cal_intercept = float(lr_cal.intercept_[0])
    except Exception:
        cal_slope, cal_intercept = 1.0, 0.0

    order = np.argsort(-prob)
    p50 = float(y_true[order[:min(50, len(y_true))]].mean())
    p100 = float(y_true[order[:min(100, len(y_true))]].mean())
    p200 = float(y_true[order[:min(200, len(y_true))]].mean())

    return {
        "n": len(y_true), "positives": int(y_true.sum()), "prevalence": prev,
        "average_precision": ap, "roc_auc": roc, "f1": f1, "f2": f2,
        "precision": prec, "recall": rec, "balanced_accuracy": bacc,
        "specificity": spec, "mcc": mcc, "brier": brier, "brier_skill": brier_skill,
        "log_loss": logl, "calibration_slope": cal_slope, "calibration_intercept": cal_intercept,
        "precision_at_50": p50, "precision_at_100": p100, "precision_at_200": p200,
        "threshold": threshold, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }

# -----------------------------------------------------------------------------
# MASTER EXECUTION FUNCTION
# -----------------------------------------------------------------------------
def run_simplified_pipeline():
    start_time = time.time()
    LOG.info("=== STARTING SIMPLIFIED TECR-NET SCIENTIFIC RE-RUN ===")

    # Load cached tensors & splits
    cache_file = Path("scratch/cache_tensors.joblib")
    if not cache_file.exists():
        raise FileNotFoundError("Cache tensors not found! Run execute_final_pipeline.py first.")
    cached_data = joblib.load(cache_file)
    X_d = cached_data["X_d"]
    X_s = cached_data["X_s"]
    X_c = cached_data["X_c"]
    y = cached_data["y"]
    roles = cached_data["roles"]
    vocab_sizes = cached_data["vocab_sizes"]

    # Maximum index + 1 for each categorical feature
    cat_cards = [int(X_c[:, j].max() + 1) for j in range(X_c.shape[1])]
    LOG.info("Loaded tensors: X_d=%s, X_s=%s, X_c=%s, y=%s, cat_cards=%s", X_d.shape, X_s.shape, X_c.shape, y.shape, cat_cards)

    idx_fit = np.flatnonzero(roles == "fitting")
    idx_arch = np.flatnonzero(roles == "architecture_selection")
    idx_cal = np.flatnonzero(roles == "calibration")
    idx_thresh = np.flatnonzero(roles == "threshold_selection")
    idx_eval = np.flatnonzero(roles == "evaluation")

    seq_df = pd.read_csv(OUT_DIR / "sequence_index.csv")
    eval_df = seq_df[seq_df["split_role"] == "evaluation"].reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 1. ARCHITECTURE SELECTION OVER 2018 SPLIT (5 SEEDS)
    # -------------------------------------------------------------------------
    LOG.info("=== PHASE 3: 2018 ARCHITECTURE SELECTION ACROSS 5 SEEDS ===")
    MODEL_SEARCH_SPACE = [
        {
            "name": "tecr_48_bce",
            "hidden_size": 48,
            "dropout": 0.20,
            "loss": "weighted_bce",
        },
        {
            "name": "tecr_64_bce",
            "hidden_size": 64,
            "dropout": 0.20,
            "loss": "weighted_bce",
        },
        {
            "name": "tecr_64_focal",
            "hidden_size": 64,
            "dropout": 0.20,
            "loss": "focal",
            "alpha": 0.70,
            "gamma": 1.0,
        },
        {
            "name": "tecr_64_focal_regularized",
            "hidden_size": 64,
            "dropout": 0.30,
            "loss": "focal",
            "alpha": 0.75,
            "gamma": 1.5,
        },
        {
            "name": "tecr_80_bce",
            "hidden_size": 80,
            "dropout": 0.25,
            "loss": "weighted_bce",
        },
    ]

    arch_search_records = []
    config_seed_scores = {}

    for cfg in MODEL_SEARCH_SPACE:
        name = cfg["name"]
        seed_aps = []
        param_count = None
        for s in SEEDS:
            m = SimplifiedTECRNet(
                n_dynamic=X_d.shape[2],
                n_static=X_s.shape[1],
                categorical_cardinalities=cat_cards,
                hidden_size=cfg["hidden_size"],
                token_size=cfg["hidden_size"],
                dropout=cfg["dropout"],
                bidirectional=True
            )
            if param_count is None:
                param_count = sum(p.numel() for p in m.parameters() if p.requires_grad)

            m, ap_2018 = train_model(
                m, X_d[idx_fit], X_s[idx_fit], X_c[idx_fit], y[idx_fit],
                X_d[idx_arch], X_s[idx_arch], X_c[idx_arch], y[idx_arch],
                loss_config=cfg, epochs=20, seed=s
            )
            seed_aps.append(ap_2018)
            LOG.info("Config %s | Seed %d -> 2018 AP: %.5f", name, s, ap_2018)

        mean_ap = float(np.mean(seed_aps))
        std_ap = float(np.std(seed_aps))
        config_seed_scores[name] = seed_aps
        arch_search_records.append({
            "name": name,
            "hidden_size": cfg["hidden_size"],
            "dropout": cfg["dropout"],
            "loss": cfg["loss"],
            "parameters": param_count,
            "mean_2018_ap": mean_ap,
            "std_2018_ap": std_ap,
            "seed_aps": str(seed_aps)
        })
        LOG.info("Summary %s -> Mean 2018 AP: %.5f ± %.5f (Params: %d)", name, mean_ap, std_ap, param_count)

    df_arch_search = pd.DataFrame(arch_search_records).sort_values("mean_2018_ap", ascending=False).reset_index(drop=True)
    df_arch_search.to_csv(OUT_DIR / "architecture_selection_trials.csv", index=False)
    LOG.info("Architecture Search Results:\n%s", df_arch_search[["name", "parameters", "mean_2018_ap", "std_2018_ap"]].to_string())

    # Selection rule:
    # Select by mean 2018 AP. If difference < 0.005, prefer smaller params, lower std, simpler model.
    top_config_row = df_arch_search.iloc[0]
    best_mean_ap = top_config_row["mean_2018_ap"]

    candidate_configs = df_arch_search[df_arch_search["mean_2018_ap"] >= best_mean_ap - 0.005]
    # Sort candidate configs by parameters ascending, then std ascending
    selected_row = candidate_configs.sort_values(by=["parameters", "std_2018_ap"], ascending=[True, True]).iloc[0]
    selected_name = selected_row["name"]
    selected_cfg = next(c for c in MODEL_SEARCH_SPACE if c["name"] == selected_name)
    LOG.info(">>> SELECTED CONFIGURATION BY 2018 RULE: %s (Mean AP: %.5f, Params: %d) <<<", selected_name, selected_row["mean_2018_ap"], selected_row["parameters"])

    # -------------------------------------------------------------------------
    # 2. FINAL 5-SEED TRAINING & LOGIT AVERAGING FOR ENSEMBLE
    # -------------------------------------------------------------------------
    LOG.info("=== PHASE 4: FINAL TRAINING OF SELECTED %s ACROSS 5 SEEDS ===", selected_name)
    tecr_models = []
    seed_eval_metrics = []
    eval_raw_logits_by_seed = []
    cal_raw_logits_by_seed = []
    thresh_raw_logits_by_seed = []

    for s in SEEDS:
        m = SimplifiedTECRNet(
            n_dynamic=X_d.shape[2],
            n_static=X_s.shape[1],
            categorical_cardinalities=cat_cards,
            hidden_size=selected_cfg["hidden_size"],
            token_size=selected_cfg["hidden_size"],
            dropout=selected_cfg["dropout"],
            bidirectional=True
        )
        m, _ = train_model(
            m, X_d[idx_fit], X_s[idx_fit], X_c[idx_fit], y[idx_fit],
            X_d[idx_arch], X_s[idx_arch], X_c[idx_arch], y[idx_arch],
            loss_config=selected_cfg, epochs=25, seed=s
        )
        torch.save(m.state_dict(), MODELS_DIR / f"simplified_tecrnet_seed_{s}.pt")
        tecr_models.append(m)

        with torch.no_grad():
            c_l = m(torch.from_numpy(X_d[idx_cal]), torch.from_numpy(X_s[idx_cal]), torch.from_numpy(X_c[idx_cal])).numpy()
            t_l = m(torch.from_numpy(X_d[idx_thresh]), torch.from_numpy(X_s[idx_thresh]), torch.from_numpy(X_c[idx_thresh])).numpy()
            e_l = m(torch.from_numpy(X_d[idx_eval]), torch.from_numpy(X_s[idx_eval]), torch.from_numpy(X_c[idx_eval])).numpy()

        cal_raw_logits_by_seed.append(c_l)
        thresh_raw_logits_by_seed.append(t_l)
        eval_raw_logits_by_seed.append(e_l)

        # Individual seed Platt calibration on 2019
        calibrator_s = LogisticRegression(C=1.0, solver="lbfgs").fit(c_l.reshape(-1, 1), y[idx_cal])
        tp_s = calibrator_s.predict_proba(t_l.reshape(-1, 1))[:, 1]
        ep_s = calibrator_s.predict_proba(e_l.reshape(-1, 1))[:, 1]

        # Individual threshold on 2020
        ts_cand = np.unique(np.quantile(tp_s, np.linspace(0.01, 0.99, 100)))
        f1s = [f1_score(y[idx_thresh], tp_s >= t, zero_division=0) for t in ts_cand]
        s_thresh = float(ts_cand[np.argmax(f1s)])

        m_res = calculate_all_metrics(y[idx_eval], ep_s, s_thresh)
        m_res["seed"] = s
        m_res["model"] = f"Simplified TECR-Net ({selected_name})"
        seed_eval_metrics.append(m_res)

    df_seed_metrics = pd.DataFrame(seed_eval_metrics)
    df_seed_metrics.to_csv(OUT_DIR / "per_seed_metrics.csv", index=False)
    LOG.info("5-Seed Individual AP (Mean ± SD): %.5f ± %.5f", df_seed_metrics["average_precision"].mean(), df_seed_metrics["average_precision"].std())

    # -------------------------------------------------------------------------
    # 3. ENSEMBLE: AVERAGE RAW LOGITS BEFORE PLATT CALIBRATION
    # -------------------------------------------------------------------------
    LOG.info("=== ENSEMBLE LEVEL CALIBRATION (AVERAGING RAW LOGITS) ===")
    mean_cal_logits = np.mean(cal_raw_logits_by_seed, axis=0)
    mean_thresh_logits = np.mean(thresh_raw_logits_by_seed, axis=0)
    mean_eval_logits = np.mean(eval_raw_logits_by_seed, axis=0)

    ens_calibrator = LogisticRegression(C=1.0, solver="lbfgs").fit(mean_cal_logits.reshape(-1, 1), y[idx_cal])
    joblib.dump(ens_calibrator, MODELS_DIR / "simplified_tecrnet_platt_calibrator.joblib")

    ens_thresh_probs = ens_calibrator.predict_proba(mean_thresh_logits.reshape(-1, 1))[:, 1]
    ens_eval_probs = ens_calibrator.predict_proba(mean_eval_logits.reshape(-1, 1))[:, 1]

    # Threshold selection on 2020
    ts_cand = np.unique(np.quantile(ens_thresh_probs, np.linspace(0.01, 0.99, 100)))
    f1s = [f1_score(y[idx_thresh], ens_thresh_probs >= t, zero_division=0) for t in ts_cand]
    ens_selected_thresh = float(ts_cand[np.argmax(f1s)])
    LOG.info("Ensemble Selected Operating Threshold (from 2020 F1): %.6f", ens_selected_thresh)

    tecr_ens_metrics = calculate_all_metrics(y[idx_eval], ens_eval_probs, ens_selected_thresh)
    tecr_ens_metrics["model"] = f"Simplified TECR-Net ({selected_name} Ensemble)"
    df_ensemble = pd.DataFrame([tecr_ens_metrics])
    df_ensemble.to_csv(OUT_DIR / "ensemble_metrics.csv", index=False)
    LOG.info("Ensemble Eval AP: %.5f, ROC: %.5f, F1: %.5f, P@50: %.2f, P@100: %.2f",
             tecr_ens_metrics["average_precision"], tecr_ens_metrics["roc_auc"],
             tecr_ens_metrics["f1"], tecr_ens_metrics["precision_at_50"], tecr_ens_metrics["precision_at_100"])

    # -------------------------------------------------------------------------
    # 4. SYMMETRIC BASELINE BENCHMARK ON IDENTICAL EVALUATION ROWS
    # -------------------------------------------------------------------------
    LOG.info("=== PHASE 4: SYMMETRIC BASELINE BENCHMARK ON IDENTICAL EVALUATION ROWS ===")
    X_flat = np.concatenate([X_d.reshape(len(X_d), -1), X_s, X_c.astype(np.float32)], axis=1)

    all_eval_probs = {tecr_ens_metrics["model"]: ens_eval_probs}
    baseline_records = [tecr_ens_metrics]

    # Persistence baseline
    inc_idx = 11 # 'inc_count' is index 11
    persist_pred = (X_d[idx_eval, -1, inc_idx] > 0).astype(float)
    persist_prob = persist_pred * 0.7 + 0.1
    all_eval_probs["Persistence Rule"] = persist_prob
    p_m = calculate_all_metrics(y[idx_eval], persist_prob, 0.5)
    p_m["model"] = "Persistence Rule"
    baseline_records.append(p_m)

    # Plain BiGRU (5 seeds)
    LOG.info("Training Plain BiGRU across 5 seeds...")
    gru_eval_logits = []
    gru_cal_logits = []
    gru_thresh_logits = []
    for s in SEEDS:
        m_gru = PlainBiGRU(n_dynamic=X_d.shape[2], hidden_size=selected_cfg["hidden_size"])
        m_gru, _ = train_model(
            m_gru, X_d[idx_fit], X_s[idx_fit], X_c[idx_fit], y[idx_fit],
            X_d[idx_arch], X_s[idx_arch], X_c[idx_arch], y[idx_arch],
            loss_config=selected_cfg, epochs=20, seed=s
        )
        with torch.no_grad():
            gru_cal_logits.append(m_gru(torch.from_numpy(X_d[idx_cal]), torch.from_numpy(X_s[idx_cal]), torch.from_numpy(X_c[idx_cal])).numpy())
            gru_thresh_logits.append(m_gru(torch.from_numpy(X_d[idx_thresh]), torch.from_numpy(X_s[idx_thresh]), torch.from_numpy(X_c[idx_thresh])).numpy())
            gru_eval_logits.append(m_gru(torch.from_numpy(X_d[idx_eval]), torch.from_numpy(X_s[idx_eval]), torch.from_numpy(X_c[idx_eval])).numpy())

    mean_gru_cal = np.mean(gru_cal_logits, axis=0)
    mean_gru_thresh = np.mean(gru_thresh_logits, axis=0)
    mean_gru_eval = np.mean(gru_eval_logits, axis=0)

    cal_gru = LogisticRegression(C=1.0, solver="lbfgs").fit(mean_gru_cal.reshape(-1, 1), y[idx_cal])
    tp_gru = cal_gru.predict_proba(mean_gru_thresh.reshape(-1, 1))[:, 1]
    ep_gru = cal_gru.predict_proba(mean_gru_eval.reshape(-1, 1))[:, 1]
    ts_cand = np.unique(np.quantile(tp_gru, np.linspace(0.01, 0.99, 100)))
    f1s = [f1_score(y[idx_thresh], tp_gru >= t, zero_division=0) for t in ts_cand]
    gru_thresh = float(ts_cand[np.argmax(f1s)])
    gru_m = calculate_all_metrics(y[idx_eval], ep_gru, gru_thresh)
    gru_m["model"] = "Plain BiGRU"
    baseline_records.append(gru_m)
    all_eval_probs["Plain BiGRU"] = ep_gru

    # Simple Concat MLP (5 seeds)
    LOG.info("Training Simple Concatenation MLP across 5 seeds...")
    mlp_eval_logits = []
    mlp_cal_logits = []
    mlp_thresh_logits = []
    for s in SEEDS:
        m_mlp = SimpleConcatMLP(in_dim=X_flat.shape[1])
        m_mlp, _ = train_model(
            m_mlp, X_d[idx_fit], X_s[idx_fit], X_c[idx_fit], y[idx_fit],
            X_d[idx_arch], X_s[idx_arch], X_c[idx_arch], y[idx_arch],
            loss_config=selected_cfg, epochs=20, seed=s
        )
        with torch.no_grad():
            mlp_cal_logits.append(m_mlp(torch.from_numpy(X_d[idx_cal]), torch.from_numpy(X_s[idx_cal]), torch.from_numpy(X_c[idx_cal])).numpy())
            mlp_thresh_logits.append(m_mlp(torch.from_numpy(X_d[idx_thresh]), torch.from_numpy(X_s[idx_thresh]), torch.from_numpy(X_c[idx_thresh])).numpy())
            mlp_eval_logits.append(m_mlp(torch.from_numpy(X_d[idx_eval]), torch.from_numpy(X_s[idx_eval]), torch.from_numpy(X_c[idx_eval])).numpy())

    mean_mlp_cal = np.mean(mlp_cal_logits, axis=0)
    mean_mlp_thresh = np.mean(mlp_thresh_logits, axis=0)
    mean_mlp_eval = np.mean(mlp_eval_logits, axis=0)

    cal_mlp = LogisticRegression(C=1.0, solver="lbfgs").fit(mean_mlp_cal.reshape(-1, 1), y[idx_cal])
    tp_mlp = cal_mlp.predict_proba(mean_mlp_thresh.reshape(-1, 1))[:, 1]
    ep_mlp = cal_mlp.predict_proba(mean_mlp_eval.reshape(-1, 1))[:, 1]
    ts_cand = np.unique(np.quantile(tp_mlp, np.linspace(0.01, 0.99, 100)))
    f1s = [f1_score(y[idx_thresh], tp_mlp >= t, zero_division=0) for t in ts_cand]
    mlp_thresh = float(ts_cand[np.argmax(f1s)])
    mlp_m = calculate_all_metrics(y[idx_eval], ep_mlp, mlp_thresh)
    mlp_m["model"] = "Simple Concatenation MLP"
    baseline_records.append(mlp_m)
    all_eval_probs["Simple Concatenation MLP"] = ep_mlp

    # Tabular Baselines
    tabular_clfs = {
        "Logistic Regression": LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=11),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, class_weight="balanced", random_state=11),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, scale_pos_weight=2.5, random_state=11, eval_metric="logloss"),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, scale_pos_weight=2.5, random_state=11, verbosity=-1),
        "CatBoost": cb.CatBoostClassifier(iterations=350, depth=6, learning_rate=0.05, auto_class_weights="Balanced", verbose=False, random_seed=11)
    }

    for b_name, clf in tabular_clfs.items():
        LOG.info("Fitting Tabular Baseline: %s...", b_name)
        clf.fit(X_flat[idx_fit], y[idx_fit])
        joblib.dump(clf, MODELS_DIR / f"{b_name.lower().replace(' ', '_')}.joblib")

        cal_p = clf.predict_proba(X_flat[idx_cal])[:, 1]
        thresh_p = clf.predict_proba(X_flat[idx_thresh])[:, 1]
        eval_p = clf.predict_proba(X_flat[idx_eval])[:, 1]

        eps = 1e-6
        cal_l = np.log(np.clip(cal_p, eps, 1 - eps) / (1 - np.clip(cal_p, eps, 1 - eps)))
        thresh_l = np.log(np.clip(thresh_p, eps, 1 - eps) / (1 - np.clip(thresh_p, eps, 1 - eps)))
        eval_l = np.log(np.clip(eval_p, eps, 1 - eps) / (1 - np.clip(eval_p, eps, 1 - eps)))

        calibrator_b = LogisticRegression(C=1.0, solver="lbfgs").fit(cal_l.reshape(-1, 1), y[idx_cal])
        cal_thresh_p = calibrator_b.predict_proba(thresh_l.reshape(-1, 1))[:, 1]
        cal_eval_p = calibrator_b.predict_proba(eval_l.reshape(-1, 1))[:, 1]

        ts_cand = np.unique(np.quantile(cal_thresh_p, np.linspace(0.01, 0.99, 100)))
        f1s = [f1_score(y[idx_thresh], cal_thresh_p >= t, zero_division=0) for t in ts_cand]
        b_thresh = float(ts_cand[np.argmax(f1s)])

        b_metrics = calculate_all_metrics(y[idx_eval], cal_eval_p, b_thresh)
        b_metrics["model"] = b_name
        baseline_records.append(b_metrics)
        all_eval_probs[b_name] = cal_eval_p

    df_baselines = pd.DataFrame(baseline_records).sort_values("average_precision", ascending=False).reset_index(drop=True)
    df_baselines.to_csv(OUT_DIR / "baseline_metrics.csv", index=False)
    LOG.info("Baseline Benchmark Summary:\n%s", df_baselines[["model", "average_precision", "roc_auc", "f1", "precision_at_50", "precision_at_100"]].to_string())

    # -------------------------------------------------------------------------
    # 5. ARCHITECTURAL ABLATIONS ON SIMPLIFIED MODEL
    # -------------------------------------------------------------------------
    LOG.info("=== PHASE 5: ARCHITECTURAL ABLATION MATRIX (SIMPLIFIED MODEL) ===")
    ablations = [
        {"variant": "Full Simplified TECR-Net", "temp_att": True, "residual": True, "static": True, "cat": True},
        {"variant": "Without Temporal Attention", "temp_att": False, "residual": True, "static": True, "cat": True},
        {"variant": "Without Residual Fusion", "temp_att": True, "residual": False, "static": True, "cat": True},
        {"variant": "Without Static Branch", "temp_att": True, "residual": True, "static": False, "cat": True},
        {"variant": "Without Categorical Branch", "temp_att": True, "residual": True, "static": True, "cat": False},
    ]

    ablation_records = []
    ablation_eval_probs = {}

    for abl in ablations:
        v_name = abl["variant"]
        LOG.info("Training Ablation: %s...", v_name)
        m_abl = SimplifiedTECRNet(
            n_dynamic=X_d.shape[2],
            n_static=X_s.shape[1],
            categorical_cardinalities=cat_cards,
            hidden_size=selected_cfg["hidden_size"],
            token_size=selected_cfg["hidden_size"],
            dropout=selected_cfg["dropout"],
            bidirectional=True,
            use_temp_att=abl["temp_att"],
            use_residual=abl["residual"],
            disable_static=not abl["static"],
            disable_categorical=not abl["cat"]
        )
        m_abl, _ = train_model(
            m_abl, X_d[idx_fit], X_s[idx_fit], X_c[idx_fit], y[idx_fit],
            X_d[idx_arch], X_s[idx_arch], X_c[idx_arch], y[idx_arch],
            loss_config=selected_cfg, epochs=20, seed=11
        )
        with torch.no_grad():
            c_l = m_abl(torch.from_numpy(X_d[idx_cal]), torch.from_numpy(X_s[idx_cal]), torch.from_numpy(X_c[idx_cal])).numpy()
            t_l = m_abl(torch.from_numpy(X_d[idx_thresh]), torch.from_numpy(X_s[idx_thresh]), torch.from_numpy(X_c[idx_thresh])).numpy()
            e_l = m_abl(torch.from_numpy(X_d[idx_eval]), torch.from_numpy(X_s[idx_eval]), torch.from_numpy(X_c[idx_eval])).numpy()

        cal_abl = LogisticRegression(C=1.0, solver="lbfgs").fit(c_l.reshape(-1, 1), y[idx_cal])
        tp_abl = cal_abl.predict_proba(t_l.reshape(-1, 1))[:, 1]
        ep_abl = cal_abl.predict_proba(e_l.reshape(-1, 1))[:, 1]

        ts_cand = np.unique(np.quantile(tp_abl, np.linspace(0.01, 0.99, 100)))
        f1s = [f1_score(y[idx_thresh], tp_abl >= t, zero_division=0) for t in ts_cand]
        abl_thresh = float(ts_cand[np.argmax(f1s)])

        abl_m = calculate_all_metrics(y[idx_eval], ep_abl, abl_thresh)
        abl_m["variant"] = v_name
        ablation_records.append(abl_m)
        ablation_eval_probs[v_name] = ep_abl

    mlp_abl_m = dict(mlp_m)
    mlp_abl_m["variant"] = "Simple Concatenation MLP"
    ablation_records.append(mlp_abl_m)
    ablation_eval_probs["Simple Concatenation MLP"] = ep_mlp

    df_ablation = pd.DataFrame(ablation_records).sort_values("average_precision", ascending=False).reset_index(drop=True)
    df_ablation.to_csv(OUT_DIR / "ablation_metrics.csv", index=False)
    LOG.info("Ablation Results Summary:\n%s", df_ablation[["variant", "average_precision", "roc_auc", "f1", "precision_at_100"]].to_string())

    # -------------------------------------------------------------------------
    # 6. RESTRICTED TIME-VALID PROFILE SENSITIVITY
    # -------------------------------------------------------------------------
    LOG.info("=== RESTRICTED TIME-VALID PREDICTOR SENSITIVITY ===")
    prov_df = pd.read_csv(OUT_DIR / "feature_provenance.csv")
    restr_dyn_names = [
        'active_months', 'has_production_record', 'producing_wells_mean',
        'producing_wells_max', 'bopd_mean', 'bopd_max', 'mcfd_mean', 'mcfd_max',
        'boepd_mean', 'bwpd_mean', 'platform_age', 'inc_count', 'enforcement_code_count'
    ]
    # DYNAMIC_FEATURES list indices
    dynamic_features_list = [
        'active_months', 'has_production_record', 'producing_wells_mean',
        'producing_wells_max', 'bopd_mean', 'bopd_max', 'mcfd_mean', 'mcfd_max',
        'boepd_mean', 'bwpd_mean', 'platform_age', 'inc_count',
        'enforcement_code_count', 'operator_facilities_inspected',
        'operator_inspection_records', 'operator_inspection_types',
        'operator_warning_incs', 'operator_component_shutin_incs',
        'operator_facility_shutin_incs', 'previous_year_inc_count',
        'inc_count_previous_3_calendar_years', 'bopd_change', 'mcfd_change'
    ]
    restr_indices = [dynamic_features_list.index(c) for c in restr_dyn_names]
    X_d_restr = X_d[:, :, restr_indices]

    m_restr = SimplifiedTECRNet(
        n_dynamic=X_d_restr.shape[2],
        n_static=X_s.shape[1],
        categorical_cardinalities=cat_cards,
        hidden_size=selected_cfg["hidden_size"],
        token_size=selected_cfg["hidden_size"],
        dropout=selected_cfg["dropout"],
        bidirectional=True
    )
    m_restr, _ = train_model(
        m_restr, X_d_restr[idx_fit], X_s[idx_fit], X_c[idx_fit], y[idx_fit],
        X_d_restr[idx_arch], X_s[idx_arch], X_c[idx_arch], y[idx_arch],
        loss_config=selected_cfg, epochs=20, seed=11
    )
    with torch.no_grad():
        c_l = m_restr(torch.from_numpy(X_d_restr[idx_cal]), torch.from_numpy(X_s[idx_cal]), torch.from_numpy(X_c[idx_cal])).numpy()
        t_l = m_restr(torch.from_numpy(X_d_restr[idx_thresh]), torch.from_numpy(X_s[idx_thresh]), torch.from_numpy(X_c[idx_thresh])).numpy()
        e_l = m_restr(torch.from_numpy(X_d_restr[idx_eval]), torch.from_numpy(X_s[idx_eval]), torch.from_numpy(X_c[idx_eval])).numpy()

    cal_restr = LogisticRegression(C=1.0, solver="lbfgs").fit(c_l.reshape(-1, 1), y[idx_cal])
    tp_restr = cal_restr.predict_proba(t_l.reshape(-1, 1))[:, 1]
    ep_restr = cal_restr.predict_proba(e_l.reshape(-1, 1))[:, 1]

    ts_cand = np.unique(np.quantile(tp_restr, np.linspace(0.01, 0.99, 100)))
    f1s = [f1_score(y[idx_thresh], tp_restr >= t, zero_division=0) for t in ts_cand]
    restr_thresh = float(ts_cand[np.argmax(f1s)])
    restr_m = calculate_all_metrics(y[idx_eval], ep_restr, restr_thresh)
    restr_m["model"] = "Restricted Time-Valid Predictors"
    all_eval_probs["Restricted Time-Valid"] = ep_restr

    # -------------------------------------------------------------------------
    # 7. CALIBRATION & RELIABILITY DIAGRAM DATA
    # -------------------------------------------------------------------------
    LOG.info("=== PHASE 6: CALIBRATION & RELIABILITY BINS ===")
    prob_true, prob_pred = calibration_curve(y[idx_eval], ens_eval_probs, n_bins=10, strategy="quantile")
    df_reliability = pd.DataFrame({
        "bin": range(1, len(prob_true) + 1),
        "mean_predicted_probability": prob_pred,
        "observed_event_rate": prob_true
    })
    df_reliability.to_csv(OUT_DIR / "reliability_bins.csv", index=False)

    cal_summary = pd.DataFrame([
        {
            "cohort": "Calibration Split (2019)",
            "brier_score": float(brier_score_loss(y[idx_cal], ens_calibrator.predict_proba(mean_cal_logits.reshape(-1, 1))[:, 1])),
            "brier_skill": float(1.0 - (brier_score_loss(y[idx_cal], ens_calibrator.predict_proba(mean_cal_logits.reshape(-1, 1))[:, 1]) / (y[idx_cal].mean() * (1 - y[idx_cal].mean())))),
            "observed_prevalence": float(y[idx_cal].mean()),
            "mean_calibrated_prob": float(ens_calibrator.predict_proba(mean_cal_logits.reshape(-1, 1))[:, 1].mean())
        },
        {
            "cohort": "Evaluation Split (2021-2024)",
            "brier_score": float(tecr_ens_metrics["brier"]),
            "brier_skill": float(tecr_ens_metrics["brier_skill"]),
            "observed_prevalence": float(y[idx_eval].mean()),
            "mean_calibrated_prob": float(ens_eval_probs.mean())
        }
    ])
    cal_summary.to_csv(OUT_DIR / "calibration_metrics.csv", index=False)

    # -------------------------------------------------------------------------
    # 8. PAIRED PLATFORM-CLUSTERED BOOTSTRAP (1,000 REPLICATES)
    # -------------------------------------------------------------------------
    LOG.info("=== PHASE 6: 1,000 PLATFORM-CLUSTERED BOOTSTRAP REPLICATES ===")
    platforms_eval = eval_df["platform_id"].unique()
    y_eval = y[idx_eval]
    tecr_p = ens_eval_probs

    grouped_indices = {pid: eval_df[eval_df["platform_id"] == pid].index.to_numpy() for pid in platforms_eval}

    rng = np.random.default_rng(2026)
    n_reps = 1000

    comparators_to_test = [
        ("CatBoost", all_eval_probs["CatBoost"]),
        ("LightGBM", all_eval_probs["LightGBM"]),
        ("XGBoost", all_eval_probs["XGBoost"]),
        ("HistGradientBoosting", all_eval_probs["HistGradientBoosting"]),
        ("Logistic Regression", all_eval_probs["Logistic Regression"]),
        ("Plain BiGRU", all_eval_probs["Plain BiGRU"]),
        ("Simple Concatenation MLP", all_eval_probs["Simple Concatenation MLP"]),
        ("Persistence Rule", all_eval_probs["Persistence Rule"]),
        ("Ablation: Without Temporal Attention", ablation_eval_probs["Without Temporal Attention"]),
        ("Ablation: Without Residual Fusion", ablation_eval_probs["Without Residual Fusion"]),
        ("Ablation: Without Static Branch", ablation_eval_probs["Without Static Branch"]),
        ("Ablation: Without Categorical Branch", ablation_eval_probs["Without Categorical Branch"]),
        ("Restricted Time-Valid Profile", all_eval_probs["Restricted Time-Valid"])
    ]

    delta_aps = {comp_name: [] for comp_name, _ in comparators_to_test}
    for _ in range(n_reps):
        sampled_pids = rng.choice(platforms_eval, size=len(platforms_eval), replace=True)
        sampled_rows = np.concatenate([grouped_indices[pid] for pid in sampled_pids])

        y_samp = y_eval[sampled_rows]
        if len(np.unique(y_samp)) < 2:
            continue

        ap_tecr = average_precision_score(y_samp, tecr_p[sampled_rows])
        for comp_name, comp_p in comparators_to_test:
            ap_comp = average_precision_score(y_samp, comp_p[sampled_rows])
            delta_aps[comp_name].append(ap_tecr - ap_comp)

    boot_rows = []
    for comp_name, comp_p in comparators_to_test:
        arr = np.array(delta_aps[comp_name])
        obs_delta = tecr_ens_metrics["average_precision"] - average_precision_score(y_eval, comp_p)
        boot_rows.append({
            "comparison": f"Simplified TECR-Net minus {comp_name}",
            "observed_delta_ap": obs_delta,
            "bootstrap_mean_delta_ap": float(np.mean(arr)),
            "ci_95_lower": float(np.quantile(arr, 0.025)),
            "ci_95_upper": float(np.quantile(arr, 0.975)),
            "p_delta_greater_zero": float(np.mean(arr > 0))
        })

    df_paired_boot = pd.DataFrame(boot_rows)
    df_paired_boot.to_csv(OUT_DIR / "paired_platform_bootstrap.csv", index=False)
    LOG.info("Bootstrap Paired Intervals:\n%s", df_paired_boot[["comparison", "observed_delta_ap", "ci_95_lower", "ci_95_upper", "p_delta_greater_zero"]].to_string())

    # -------------------------------------------------------------------------
    # 9. EXPORT PREDICTIONS & MANIFEST
    # -------------------------------------------------------------------------
    pred_export = eval_df[["platform_id", "feature_year", "target_year", "target"]].copy()
    pred_export["simplified_tecrnet_prob"] = ens_eval_probs
    pred_export["logistic_regression_prob"] = all_eval_probs["Logistic Regression"]
    pred_export["simple_concat_mlp_prob"] = all_eval_probs["Simple Concatenation MLP"]
    pred_export["plain_bigru_prob"] = all_eval_probs["Plain BiGRU"]
    pred_export["catboost_prob"] = all_eval_probs["CatBoost"]
    pred_export["lightgbm_prob"] = all_eval_probs["LightGBM"]
    pred_export["xgboost_prob"] = all_eval_probs["XGBoost"]
    pred_export["hist_gradient_boosting_prob"] = all_eval_probs["HistGradientBoosting"]
    pred_export["selected_threshold"] = ens_selected_thresh
    pred_export["tecrnet_flag"] = (ens_eval_probs >= ens_selected_thresh).astype(int)
    pred_export.to_csv(OUT_DIR / "predictions.csv", index=False)

    def sha256_file(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1024*1024), b""):
                h.update(b)
        return h.hexdigest()

    manifest_files = [
        "cohort_flow.csv",
        "sequence_index.csv",
        "split_summary.csv",
        "feature_provenance.csv",
        "protocol_comparison.csv",
        "architecture_selection_trials.csv",
        "per_seed_metrics.csv",
        "ensemble_metrics.csv",
        "baseline_metrics.csv",
        "ablation_metrics.csv",
        "paired_platform_bootstrap.csv",
        "calibration_metrics.csv",
        "reliability_bins.csv",
        "predictions.csv"
    ]
    file_hashes = {f: sha256_file(OUT_DIR / f) for f in manifest_files}

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_runtime_seconds": round(time.time() - start_time, 2),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "sklearn": "1.8.0",
            "catboost": cb.__version__,
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__
        },
        "selected_architecture": {
            "name": selected_name,
            "configuration": selected_cfg,
            "trainable_parameters": int(selected_row["parameters"]),
            "selection_criterion": "2018 AP with parsimony tie-breaker (< 0.005 AP delta)"
        },
        "seeds": SEEDS,
        "protocol": {
            "fitting_years": "2014-2017",
            "architecture_selection_year": "2018",
            "calibration_year": "2019",
            "threshold_selection_year": "2020",
            "evaluation_years": "2021-2024",
            "evaluation_status": "development_exploratory_previously_inspected",
            "confirmatory_untouched_trial": "UNAVAILABLE"
        },
        "file_hashes": file_hashes
    }
    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    LOG.info("=== COMPLETED: All deliverables and manifest saved under %s ===", OUT_DIR)

if __name__ == "__main__":
    run_simplified_pipeline()
