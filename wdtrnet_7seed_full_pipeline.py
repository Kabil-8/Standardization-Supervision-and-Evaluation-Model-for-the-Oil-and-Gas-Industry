"""Seven-seed WDTR-Net development pipeline for Antigravity/Kaggle.

All learned models receive the same platform histories, split years, tuning
trial count, seven final seeds, ensemble calibration, threshold rule and
evaluation rows. The script deliberately separates calibration (2019),
threshold selection (2020), and developmental evaluation (2021).

This code cannot turn a previously inspected cohort into independent evidence.
The 2022-2023 output is therefore marked exploratory, and feature year 2024 is
excluded unless explicitly enabled after verifying 2025 outcome completeness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, fbeta_score, log_loss, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import optuna
except ImportError as exc:
    raise RuntimeError("Install optuna before running this script") from exc

LOG = logging.getLogger("bsee-rerun")
PACKAGE_VERSION = "wdtr-1.0.0"
SEEDS = (11, 22, 33, 44, 55, 66, 77)
TUNING_SEEDS = (11,)

DYNAMIC = [
    "active_months", "producing_wells_mean", "producing_wells_max",
    "bopd_mean", "bopd_max", "mcfd_mean", "mcfd_max", "boepd_mean",
    "bwpd_mean", "platform_age", "operator_facilities_inspected",
    "operator_inspection_records", "operator_inspection_types",
    "investigated_accident_count", "accident_type_count",
    "previous_year_inc_count", "inc_count_previous_3_observations",
    "previous_year_accident_count", "production_change_bopd",
    "production_change_mcfd", "inc_count", "enforcement_code_count",
    "any_inc", "operator_warning_incs", "operator_component_shutin_incs",
    "operator_facility_shutin_incs",
]
STATIC = ["water_depth", "latitude", "longitude"]
CATEGORICAL = [
    "area_code", "region_code", "structure_type", "major_structure_flag",
    "district_code", "heliport_flag", "attended_8hr_flag", "manned_24hr_flag",
]
CONSERVATIVE_DYNAMIC = [
    "active_months", "producing_wells_mean", "producing_wells_max",
    "bopd_mean", "bopd_max", "mcfd_mean", "mcfd_max", "boepd_mean",
    "bwpd_mean", "production_change_bopd", "production_change_mcfd",
    "inc_count", "enforcement_code_count", "any_inc",
    "previous_year_inc_count", "inc_count_previous_3_observations",
]
TARGET_CANDIDATES = ("next_year_any_inc", "target", "label", "y")
PLATFORM_CANDIDATES = ("platform_id", "platform_key", "complex_id", "platform")


@dataclass
class Config:
    panel: str | None = None
    output_dir: str = "/kaggle/working/bsee_corrected_symmetric_rerun"
    trials: int = 25
    bootstrap_replicates: int = 2000
    epochs: int = 60
    patience: int = 8
    batch_size: int = 256
    device: str = "cuda"
    allow_2024_feature_year: bool = False
    smoke: bool = False
    profiles: str = "full_37,conservative_event_year"
    expected_sequences: int = 21615
    allow_cohort_mismatch: bool = False
    raw_platform_inc: str | None = None
    audit_only: bool = False


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--panel")
    p.add_argument("--output-dir", default=Config.output_dir)
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--bootstrap-replicates", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-2024-feature-year", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--profiles", default="full_37,conservative_event_year")
    p.add_argument("--expected-sequences", type=int, default=Config.expected_sequences)
    p.add_argument("--allow-cohort-mismatch", action="store_true")
    p.add_argument("--raw-platform-inc")
    p.add_argument("--audit-only", action="store_true")
    a, unknown = p.parse_known_args()
    if unknown:
        LOG.info("Ignoring notebook arguments: %s", unknown)
    return Config(**vars(a))


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def find_panel(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file(): raise FileNotFoundError(path)
        return path
    roots = [Path("/kaggle/input"), Path.cwd()]
    names = [
        "risk_set_all_platform_years.csv.gz", "risk_set_all_platform_years.csv",
        "bsee_platform_year_corrected.csv.gz", "bsee_platform_year_corrected.csv",
        "corrected_platform_year_panel.csv.gz", "corrected_platform_year_panel.csv",
    ]
    found = []
    for root in roots:
        if root.exists():
            for name in names: found.extend(root.rglob(name))
    if not found:
        raise FileNotFoundError(
            "No corrected platform-year panel found. The archived bsee_platform_year_full "
            "panel is intentionally rejected. Build or pass "
            "risk_set_all_platform_years.csv.gz explicitly with --panel."
        )
    found = sorted(set(found), key=lambda x: ("corrected" not in x.name.lower(), len(str(x))))
    LOG.info("Auto-selected panel: %s", found[0])
    return found[0]


def choose_column(columns, candidates, label):
    lut = {str(c).lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lut: return lut[c.lower()]
    raise KeyError(f"Cannot identify {label}; candidates={candidates}")


def normalize_panel(path: Path, allow_2024: bool) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path, low_memory=False)
    platform = choose_column(df.columns, PLATFORM_CANDIDATES, "platform column")
    target = choose_column(df.columns, TARGET_CANDIDATES, "target column")
    year = choose_column(df.columns, ("year", "feature_year"), "year column")
    df = df.rename(columns={platform: "platform_id", year: "year", target: "target"})
    required = set(DYNAMIC + STATIC + CATEGORICAL + ["platform_id", "year", "target"])
    missing = sorted(required - set(df.columns))
    if missing: raise KeyError(f"Panel is missing required columns: {missing}")
    df["platform_id"] = df["platform_id"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    for col in DYNAMIC + STATIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[DYNAMIC + STATIC] = df[DYNAMIC + STATIC].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["platform_id", "year", "target"]).copy()
    df["year"] = df["year"].astype(int); df["target"] = df["target"].astype(int)
    if not set(df.target.unique()).issubset({0, 1}): raise ValueError("Target must be binary")
    # Retain feature year 2024 for cohort reconciliation. It is excluded from
    # modelling masks unless the user separately verifies 2025 completeness.
    dup = df.duplicated(["platform_id", "year"], keep=False)
    if dup.any():
        sample = df.loc[dup, ["platform_id", "year"]].head(20).to_dict("records")
        raise ValueError(f"Duplicate platform-year keys detected: {sample}")
    return df.sort_values(["platform_id", "year"]), "platform_id", "target"


def exact_sequences(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rows = {(r.platform_id, int(r.year)): r for r in df.itertuples(index=False)}
    dyn, stat, cat, meta = [], [], [], []
    for r in df.itertuples(index=False):
        t = int(r.year); pid = r.platform_id
        if t < 2014 or t > 2024: continue
        keys = [(pid, t - 2), (pid, t - 1), (pid, t)]
        if not all(k in rows for k in keys): continue
        hist = [rows[k] for k in keys]
        dyn.append([[getattr(q, f) for f in DYNAMIC] for q in hist])
        stat.append([getattr(r, f) for f in STATIC])
        cat.append([getattr(r, f) for f in CATEGORICAL])
        meta.append({"platform_id": pid, "feature_year": t, "target_year": t + 1,
                     "target": int(r.target), "sample_id": f"{pid}|{t}"})
    if not meta: raise ValueError("No exact three-year histories were constructed")
    return np.asarray(dyn, dtype=object), np.asarray(stat, dtype=object), np.asarray(cat, dtype=object), pd.DataFrame(meta)


def audit_corrected_cohort(df, meta, out, expected, raw_inc=None):
    """Write audit evidence before allowing any expensive model fitting."""
    audit = {
        "platform_year_rows": int(len(df)), "platforms": int(df.platform_id.nunique()),
        "duplicate_platform_years": int(df.duplicated(["platform_id","year"]).sum()),
        "calendar_exact_three_year_histories": int(len(meta)),
        "expected_histories": int(expected), "history_count_matches": bool(len(meta)==expected),
        "positive_histories": int(meta.target.sum()), "history_prevalence": float(meta.target.mean()),
        "eligibility_uses_t_plus_1_production": False,
        "negative_outcome_completeness_verified": False,
        "historical_publication_timing_verified": False,
    }
    pd.DataFrame([audit]).to_csv(out/"00_corrected_cohort_audit.csv",index=False)
    meta.groupby("feature_year").agg(histories=("sample_id","size"),platforms=("platform_id","nunique"),
        positives=("target","sum"),prevalence=("target","mean")).reset_index().to_csv(out/"01_cohort_by_feature_year.csv",index=False)
    if raw_inc and Path(raw_inc).is_file():
        path=Path(raw_inc)
        inc=pd.read_csv(path,low_memory=False)
        date_col=choose_column(inc.columns,("INC_ISSUE_DATE","inc_issue_date","INC_DATE","inc_date"),"INC date")
        pid_col=choose_column(inc.columns,("ID_NUMBER","platform_id","platform_key"),"INC platform key")
        parsed=pd.to_datetime(inc[date_col],errors="coerce",format="mixed")
        stable=[c for c in (pid_col,"INC_NUMBER","INC_ENFRC_CODE",date_col) if c in inc.columns]
        raw_audit={"path":str(path),"sha256":sha256(path),"rows":len(inc),"parsed_dates":int(parsed.notna().sum()),
            "unparsed_dates":int(parsed.isna().sum()),"duplicate_rows_on_available_stable_fields":int(inc.duplicated(stable).sum()),
            "platforms":int(inc[pid_col].astype(str).nunique()),"min_date":str(parsed.min()),"max_date":str(parsed.max())}
        pd.DataFrame([raw_audit]).to_csv(out/"02_raw_platform_inc_date_audit.csv",index=False)
    else:
        pd.DataFrame([{"status":"NOT_RUN","reason":"Pass --raw-platform-inc to audit source INC dates and duplicates"}]).to_csv(out/"02_raw_platform_inc_date_audit.csv",index=False)
    return audit


class Preprocessor:
    def __init__(self, dynamic_names, static_names, categorical_names):
        self.dynamic_names = list(dynamic_names); self.static_names = list(static_names)
        self.categorical_names = list(categorical_names)
        self.dynamic_imp = SimpleImputer(strategy="median", keep_empty_features=True)
        self.dynamic_scale = RobustScaler(quantile_range=(25, 75))
        self.static_imp = SimpleImputer(strategy="median", keep_empty_features=True) if self.static_names else None
        self.static_scale = RobustScaler(quantile_range=(25, 75)) if self.static_names else None
        self.vocabs: list[dict[str, int]] = []

    def fit(self, xd, xs, xc):
        nd = np.asarray(xd, float).reshape(-1, len(self.dynamic_names))
        self.dynamic_imp.fit(nd); self.dynamic_scale.fit(self.dynamic_imp.transform(nd))
        if self.static_names:
            ns = np.asarray(xs, float)
            self.static_imp.fit(ns); self.static_scale.fit(self.static_imp.transform(ns))
        self.vocabs = []
        for j in range(xc.shape[1]):
            vals = sorted({str(v) for v in xc[:, j] if pd.notna(v)})
            self.vocabs.append({v: i + 1 for i, v in enumerate(vals)})
        return self

    def transform(self, xd, xs, xc):
        nd = np.asarray(xd, float); ns = np.asarray(xs, float)
        zd = self.dynamic_scale.transform(self.dynamic_imp.transform(nd.reshape(-1, nd.shape[-1]))).astype("float32").reshape(nd.shape)
        if self.static_names:
            zs = self.static_scale.transform(self.static_imp.transform(ns)).astype("float32")
        else:
            zs = np.zeros((len(nd), 0), dtype="float32")
        zc = np.zeros(xc.shape, dtype="int64")
        for j, vocab in enumerate(self.vocabs):
            zc[:, j] = [vocab.get(str(v), 0) if pd.notna(v) else 0 for v in xc[:, j]]
        return zd, zs, zc

    @property
    def cardinalities(self): return [len(v) + 1 for v in self.vocabs]


def subset_profile(xd, xs, xc, profile):
    if profile == "full_37": return xd, xs, xc, DYNAMIC, STATIC, CATEGORICAL
    if profile == "conservative_event_year":
        idx = [DYNAMIC.index(x) for x in CONSERVATIVE_DYNAMIC]
        return xd[:, :, idx], np.zeros((len(xd), 0)), np.zeros((len(xd), 0), dtype=object), CONSERVATIVE_DYNAMIC, [], []
    raise ValueError(profile)


def split_masks(meta, allow_2024=False):
    y = meta.feature_year
    return {
        "fit": y.between(2014, 2017).to_numpy(),
        "tune": (y == 2018).to_numpy(),
        "calibration": (y == 2019).to_numpy(),
        "threshold": (y == 2020).to_numpy(),
        "evaluation": (y == 2021).to_numpy(),
        "exploratory_later": y.between(2022, 2024 if allow_2024 else 2023).to_numpy(),
    }


def validate_splits(meta, masks):
    out = []
    for role, m in masks.items():
        yy = meta.loc[m, "target"].to_numpy()
        if len(yy) == 0 or len(np.unique(yy)) < 2: raise ValueError(f"Split {role} lacks rows or both classes")
        out.append({"role": role, "n": len(yy), "platforms": meta.loc[m, "platform_id"].nunique(),
                    "positives": int(yy.sum()), "prevalence": float(yy.mean()),
                    "feature_year_min": int(meta.loc[m, "feature_year"].min()),
                    "feature_year_max": int(meta.loc[m, "feature_year"].max())})
    return pd.DataFrame(out)


class Platt:
    def __init__(self): self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    def fit(self, p, y): self.model.fit(logit(np.clip(p, 1e-6, 1-1e-6)).reshape(-1,1), y); return self
    def predict(self, p): return self.model.predict_proba(logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1))[:,1]


def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    if not len(th): return 0.5
    f = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    return float(th[int(np.nanargmax(f))])


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    base = float(np.mean(y) * (1 - np.mean(y)))
    brier = brier_score_loss(y, p)
    result = {
        "n": len(y), "positives": int(np.sum(y)), "prevalence": float(np.mean(y)),
        "AP": average_precision_score(y,p), "ROC_AUC": roc_auc_score(y,p),
        "F1": f1_score(y,pred,zero_division=0), "F2": fbeta_score(y,pred,beta=2,zero_division=0),
        "precision": precision_score(y,pred,zero_division=0), "recall": recall_score(y,pred,zero_division=0),
        "specificity": tn/max(tn+fp,1), "balanced_accuracy": balanced_accuracy_score(y,pred),
        "MCC": matthews_corrcoef(y,pred), "Brier": brier,
        "Brier_skill": 1-brier/base if base>0 else np.nan, "log_loss": log_loss(y,p,labels=[0,1]),
        "threshold": threshold, "TN":int(tn), "FP":int(fp), "FN":int(fn), "TP":int(tp),
    }
    order=np.argsort(-p)
    for k in (50,100,200):
        kk=min(k,len(y)); pk=float(np.mean(np.asarray(y)[order[:kk]])) if kk else np.nan
        result[f"Precision_at_{k}"]=pk; result[f"Lift_at_{k}"]=pk/np.mean(y) if np.mean(y)>0 else np.nan
    return result


class FocalLoss(nn.Module):
    def __init__(self, alpha=.75, gamma=2.0): super().__init__(); self.alpha=alpha; self.gamma=gamma
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits); pt = targets*p + (1-targets)*(1-p)
        at = targets*self.alpha + (1-targets)*(1-self.alpha)
        return (at*(1-pt).pow(self.gamma)*bce).mean()


class TemporalAttention(nn.Module):
    def __init__(self, d): super().__init__(); self.net=nn.Sequential(nn.Linear(d,d),nn.Tanh(),nn.Linear(d,1,bias=False))
    def forward(self,x):
        w=torch.softmax(self.net(x).squeeze(-1),dim=1); return torch.sum(x*w.unsqueeze(-1),dim=1),w


class WDTRNet(nn.Module):
    """Wide-and-Deep Temporal Residual Network returning raw logits."""
    def __init__(self, nd, ns, cards, hidden=48, dropout=.20, heads=8,
                 plain_gru=False, variant="full"):
        super().__init__()
        self.plain_gru=plain_gru; self.variant=variant; self.nd=nd; self.ns=ns; self.cards=cards
        token=hidden
        self.dynamic_norm=nn.LayerNorm(nd)
        self.gru=nn.GRU(nd,hidden,batch_first=True,bidirectional=True)
        self.temporal_attention=TemporalAttention(2*hidden)
        if plain_gru:
            self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(2*hidden,1)); return
        self.temporal_proj=nn.Sequential(nn.Linear(2*hidden,token),nn.GELU(),nn.LayerNorm(token),nn.Dropout(dropout))
        self.static=nn.Sequential(nn.Linear(max(ns,1),32),nn.GELU(),nn.LayerNorm(32),nn.Dropout(dropout),
                                  nn.Linear(32,token),nn.GELU(),nn.LayerNorm(token))
        dims=[max(3,min(12,round(math.sqrt(c)))) for c in cards]
        self.emb=nn.ModuleList([nn.Embedding(c,d,padding_idx=0) for c,d in zip(cards,dims)])
        self.cat=nn.Sequential(nn.Linear(max(sum(dims),1),token),nn.GELU(),nn.LayerNorm(token),nn.Dropout(dropout))
        wide_dim=3*nd+ns
        self.wide_norm=nn.LayerNorm(wide_dim)
        self.wide_logit=nn.Linear(wide_dim,1)
        self.residual_net=nn.Sequential(nn.Linear(3*token,96),nn.GELU(),nn.LayerNorm(96),nn.Dropout(dropout),
                                        nn.Linear(96,32),nn.GELU(),nn.Dropout(dropout),nn.Linear(32,1))
        self.residual_scale_logit=nn.Parameter(torch.tensor(-2.20))

    def forward(self,xd,xs,xc,return_components=False):
        h,_=self.gru(self.dynamic_norm(xd))
        if self.variant=="no_temporal_attention":
            pooled=h.mean(1); temporal_weights=torch.full((len(xd),xd.shape[1]),1/xd.shape[1],device=xd.device)
        else:
            pooled,temporal_weights=self.temporal_attention(h)
        if self.plain_gru: return self.head(pooled).squeeze(-1)
        zt=self.temporal_proj(pooled)
        zs=self.static(xs if xs.shape[1] else torch.zeros((len(xs),1),device=xs.device))
        if len(self.cards) and self.variant!="no_categorical":
            zc=self.cat(torch.cat([e(xc[:,j]) for j,e in enumerate(self.emb)],dim=1))
        else:
            zc=torch.zeros_like(zt)
        if self.variant=="no_static": zs=torch.zeros_like(zt)
        fused=torch.cat([zt,zs,zc],dim=1)
        residual=self.residual_net(fused).squeeze(-1)
        wide=self.wide_logit(self.wide_norm(torch.cat([xd.reshape(len(xd),-1),xs],dim=1))).squeeze(-1)
        scale=torch.sigmoid(self.residual_scale_logit)
        if self.variant=="no_wide": logits=scale*residual
        elif self.variant=="no_neural_residual": logits=wide
        else: logits=wide+scale*residual
        if return_components:
            return {"logits":logits,"linear_logit":wide,"residual_logit":residual,
                    "residual_scale":scale,"temporal_attention":temporal_weights}
        return logits


TECRNet = WDTRNet


def neural_predict(model, arrays, device, batch=1024):
    model.eval(); out=[]
    ds=TensorDataset(*(torch.as_tensor(a) for a in arrays))
    with torch.no_grad():
        for xd,xs,xc in DataLoader(ds,batch_size=batch,shuffle=False):
            out.append(torch.sigmoid(model(xd.float().to(device),xs.float().to(device),xc.long().to(device))).cpu().numpy())
    return np.concatenate(out)


def fit_neural(params, train, valid, ytr, yva, seed, cfg, plain=False, fixed_epochs=None, variant="full"):
    seed_everything(seed); dev=torch.device(cfg.device if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    hidden=int(params.get("hidden",48)); heads=int(params.get("heads",8)); heads=heads if hidden%heads==0 else 4
    model=TECRNet(train[0].shape[-1],train[1].shape[-1],params["cards"],hidden,float(params.get("dropout",.25)),heads,plain,variant).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=float(params.get("lr",8e-4)),weight_decay=float(params.get("weight_decay",1e-4)))
    loss_name=params.get("loss","weighted_bce")
    if loss_name=="focal":
        lossfn=FocalLoss(float(params.get("alpha",.75)),float(params.get("gamma",1.5)))
    else:
        positives=max(float(np.sum(ytr)),1.0); negatives=max(float(len(ytr)-np.sum(ytr)),1.0)
        lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives/positives,device=dev))
    ds=TensorDataset(*(torch.as_tensor(a) for a in train),torch.as_tensor(ytr,dtype=torch.float32))
    loader=DataLoader(ds,batch_size=cfg.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=-1.; state=None; bad=0; best_epoch=1; epochs=fixed_epochs or cfg.epochs
    for epoch in range(1,epochs+1):
        model.train()
        for xd,xs,xc,y in loader:
            opt.zero_grad(set_to_none=True)
            if not plain and variant=="full":
                parts=model(xd.float().to(dev),xs.float().to(dev),xc.long().to(dev),return_components=True)
                z=parts["logits"]; loss=lossfn(z,y.to(dev))+float(params.get("residual_penalty",1e-4))*parts["residual_logit"].pow(2).mean()
            else:
                z=model(xd.float().to(dev),xs.float().to(dev),xc.long().to(dev)); loss=lossfn(z,y.to(dev))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if fixed_epochs is None:
            pv=neural_predict(model,valid,dev); score=average_precision_score(yva,pv)
            if score>best+1e-5: best=score; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_epoch=epoch; bad=0
            else: bad+=1
            if bad>=cfg.patience: break
    if state is not None: model.load_state_dict(state)
    return model, dev, best_epoch


def tabular_matrix(xd,xs,xc): return np.concatenate([xd.reshape(len(xd),-1),xs,xc.astype(float)],axis=1)


def make_tabular(name, params, seed, categorical_count):
    if name=="logistic":
        if categorical_count:
            pre = ColumnTransformer([
                ("numeric", RobustScaler(), slice(0, -categorical_count)),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), slice(-categorical_count, None)),
            ])
        else:
            pre = RobustScaler()
        return Pipeline([("preprocess",pre),
                         ("model",LogisticRegression(C=params.get("C",1.),class_weight="balanced",max_iter=3000,random_state=seed))])
    if name=="histgradientboosting":
        return HistGradientBoostingClassifier(learning_rate=params.get("learning_rate",.05),max_iter=params.get("max_iter",300),
            max_leaf_nodes=params.get("max_leaf_nodes",31),l2_regularization=params.get("l2",1.),class_weight="balanced",random_state=seed)
    if name=="catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=params.get("iterations",500),depth=params.get("depth",6),learning_rate=params.get("learning_rate",.05),
            l2_leaf_reg=params.get("l2",3.),loss_function="Logloss",eval_metric="PRAUC",auto_class_weights="Balanced",verbose=False,random_seed=seed,thread_count=-1)
    if name=="lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=params.get("n_estimators",500),num_leaves=params.get("num_leaves",31),learning_rate=params.get("learning_rate",.05),
            min_child_samples=params.get("min_child_samples",20),subsample=params.get("subsample",.9),colsample_bytree=params.get("colsample",.9),
            reg_lambda=params.get("reg_lambda",1.),class_weight="balanced",random_state=seed,n_jobs=-1,verbosity=-1)
    if name=="xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=params.get("n_estimators",500),max_depth=params.get("max_depth",5),learning_rate=params.get("learning_rate",.05),
            min_child_weight=params.get("min_child_weight",2.),subsample=params.get("subsample",.9),colsample_bytree=params.get("colsample",.9),
            reg_lambda=params.get("reg_lambda",1.),eval_metric="aucpr",tree_method="hist",random_state=seed,n_jobs=-1)
    raise ValueError(name)


def suggest(trial, name, cards):
    if name in ("tecr-net","plain-gru"):
        loss=trial.suggest_categorical("loss",["weighted_bce","focal"])
        result={"hidden":trial.suggest_categorical("hidden",[32,48,64,80]),"heads":8,
            "dropout":trial.suggest_float("dropout",.1,.4),"lr":trial.suggest_float("lr",1e-4,3e-3,log=True),
            "weight_decay":trial.suggest_float("weight_decay",1e-6,1e-2,log=True),
            "residual_penalty":trial.suggest_float("residual_penalty",1e-6,1e-2,log=True),"loss":loss,"cards":cards}
        if loss=="focal":
            result["alpha"]=trial.suggest_float("alpha",.55,.9); result["gamma"]=trial.suggest_float("gamma",.5,2.5)
        return result
    if name=="logistic": return {"C":trial.suggest_float("C",1e-3,100,log=True)}
    if name=="histgradientboosting": return {"learning_rate":trial.suggest_float("learning_rate",.01,.2,log=True),"max_iter":trial.suggest_int("max_iter",150,700),"max_leaf_nodes":trial.suggest_int("max_leaf_nodes",15,63),"l2":trial.suggest_float("l2",1e-4,10,log=True)}
    if name=="catboost": return {"iterations":trial.suggest_int("iterations",250,900),"depth":trial.suggest_int("depth",4,9),"learning_rate":trial.suggest_float("learning_rate",.015,.15,log=True),"l2":trial.suggest_float("l2",1.,12.,log=True)}
    if name=="lightgbm": return {"n_estimators":trial.suggest_int("n_estimators",250,900),"num_leaves":trial.suggest_int("num_leaves",15,63),"learning_rate":trial.suggest_float("learning_rate",.015,.15,log=True),"min_child_samples":trial.suggest_int("min_child_samples",10,80),"subsample":trial.suggest_float("subsample",.65,1.),"colsample":trial.suggest_float("colsample",.65,1.),"reg_lambda":trial.suggest_float("reg_lambda",1e-3,10,log=True)}
    if name=="xgboost": return {"n_estimators":trial.suggest_int("n_estimators",250,900),"max_depth":trial.suggest_int("max_depth",3,8),"learning_rate":trial.suggest_float("learning_rate",.015,.15,log=True),"min_child_weight":trial.suggest_float("min_child_weight",1.,10.),"subsample":trial.suggest_float("subsample",.65,1.),"colsample":trial.suggest_float("colsample",.65,1.),"reg_lambda":trial.suggest_float("reg_lambda",1e-3,10,log=True)}
    raise ValueError(name)


def paired_cluster_bootstrap(pred, meta, reps, seed=20261020):
    models=sorted(pred.model.unique()); ref="tecr-net"; rows=[]; rng=np.random.default_rng(seed)
    wide=pred.pivot(index="sample_id",columns="model",values="probability").join(meta.set_index("sample_id")[["platform_id","target"]]).dropna()
    groups={k:g.index.to_numpy() for k,g in wide.groupby("platform_id")}; ids=np.array(list(groups),dtype=object)
    obs={m:average_precision_score(wide.target,wide[m]) for m in models}
    vals={(m,metric):[] for m in models if m!=ref for metric in ("AP","ROC_AUC","Precision_at_50","Precision_at_100","Precision_at_200")}
    def score(frame, model, metric):
        yy=frame.target.to_numpy(); pp=frame[model].to_numpy()
        if metric=="AP": return average_precision_score(yy,pp)
        if metric=="ROC_AUC": return roc_auc_score(yy,pp)
        k=int(metric.rsplit("_",1)[1]); idx=np.argsort(-pp)[:min(k,len(pp))]; return float(np.mean(yy[idx]))
    for _ in range(reps):
        chosen=rng.choice(ids,len(ids),replace=True); idx=np.concatenate([groups[x] for x in chosen]); b=wide.loc[idx]
        if b.target.nunique()<2: continue
        for m in models:
            if m==ref: continue
            for metric in ("AP","ROC_AUC","Precision_at_50","Precision_at_100","Precision_at_200"):
                vals[(m,metric)].append(score(b,ref,metric)-score(b,m,metric))
    for (m,metric),v in vals.items():
        observed=score(wide,ref,metric)-score(wide,m,metric)
        rows.append({"reference":ref,"comparator":m,"metric":metric,"observed_difference":observed,
                     "bootstrap_mean_difference":float(np.mean(v)),"lower_95":float(np.quantile(v,.025)),
                     "upper_95":float(np.quantile(v,.975)),"replicates":len(v)})
    return pd.DataFrame(rows)


def plot_reliability(y,p,path):
    frac,mean=calibration_curve(y,p,n_bins=10,strategy="quantile")
    plt.figure(figsize=(5,5)); plt.plot([0,1],[0,1],"--",color="gray"); plt.plot(mean,frac,"o-");
    plt.xlabel("Mean predicted score"); plt.ylabel("Observed fraction"); plt.tight_layout(); plt.savefig(path,dpi=220); plt.close()
    return pd.DataFrame({"mean_predicted":mean,"observed_fraction":frac})


def run_profile(profile, xd0,xs0,xc0,meta,masks,cfg,out):
    pdir=out/profile; pdir.mkdir(parents=True,exist_ok=True)
    xd0,xs0,xc0,dnames,snames,cnames=subset_profile(xd0,xs0,xc0,profile)
    prep=Preprocessor(dnames,snames,cnames).fit(xd0[masks["fit"]],xs0[masks["fit"]],xc0[masks["fit"]])
    xd,xs,xc=prep.transform(xd0,xs0,xc0); joblib.dump(prep,pdir/"preprocessor.joblib")
    y=meta.target.to_numpy(); arrays=lambda m:(xd[m],xs[m],xc[m])
    models=["tecr-net","plain-gru","catboost","lightgbm","xgboost","histgradientboosting","logistic"]
    if cfg.smoke: models=["tecr-net","logistic"]
    best={}; best_epochs={}
    for name in models:
        hpfile=pdir/f"best_params_{name}.json"
        if hpfile.exists(): best[name]=json.loads(hpfile.read_text()); continue
        LOG.info("%s/%s tuning with %d trials",profile,name,cfg.trials)
        def objective(trial):
            params=suggest(trial,name,prep.cardinalities); scores=[]; epochs=[]
            for sd in (TUNING_SEEDS[:1] if cfg.smoke else TUNING_SEEDS):
                if name in ("tecr-net","plain-gru"):
                    mdl,dev,ep=fit_neural(params,arrays(masks["fit"]),arrays(masks["tune"]),y[masks["fit"]],y[masks["tune"]],sd,cfg,name=="plain-gru")
                    pv=neural_predict(mdl,arrays(masks["tune"]),dev); epochs.append(ep)
                else:
                    mdl=make_tabular(name,params,sd,len(cnames)); mdl.fit(tabular_matrix(*arrays(masks["fit"])),y[masks["fit"]]); pv=mdl.predict_proba(tabular_matrix(*arrays(masks["tune"])))[:,1]
                scores.append(average_precision_score(y[masks["tune"]],pv))
            if epochs: trial.set_user_attr("median_best_epoch",int(np.median(epochs)))
            return float(np.mean(scores))
        study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=11))
        study.optimize(objective,n_trials=(2 if cfg.smoke else cfg.trials),show_progress_bar=True)
        best[name]=study.best_params; best[name]["cards"]=prep.cardinalities
        if "median_best_epoch" in study.best_trial.user_attrs: best[name]["fixed_epochs"]=study.best_trial.user_attrs["median_best_epoch"]
        hpfile.write_text(json.dumps(best[name],indent=2)); study.trials_dataframe().to_csv(pdir/f"trials_{name}.csv",index=False)
    refit=masks["fit"]|masks["tune"]
    raw={role:{} for role in ("calibration","threshold","evaluation","exploratory_later")}
    seed_prediction_rows=[]; seed_metric_rows=[]
    for name in models:
        for si,sd in enumerate(SEEDS if not cfg.smoke else SEEDS[:1]):
            model_file=pdir/f"model_{name}_seed_{sd}.joblib"; pt_file=pdir/f"model_{name}_seed_{sd}.pt"
            LOG.info("%s/%s seed %d",profile,name,sd)
            if name in ("tecr-net","plain-gru"):
                mdl,dev,_=fit_neural(best[name],arrays(refit),arrays(masks["calibration"]),y[refit],y[masks["calibration"]],sd,cfg,name=="plain-gru",int(best[name].get("fixed_epochs",20)))
                torch.save({"state_dict":mdl.state_dict(),"params":best[name],"dynamic":dnames,"static":snames,"categorical":cnames},pt_file)
                pred=lambda role:neural_predict(mdl,arrays(masks[role]),dev)
            else:
                mdl=make_tabular(name,best[name],sd,len(cnames)); mdl.fit(tabular_matrix(*arrays(refit)),y[refit]); joblib.dump(mdl,model_file)
                pred=lambda role:mdl.predict_proba(tabular_matrix(*arrays(masks[role])))[:,1]
            for role in raw:
                values=pred(role); raw[role].setdefault(name,[]).append(values)
                block=meta.loc[masks[role],["sample_id","platform_id","feature_year","target_year","target"]].copy()
                block["profile"]=profile; block["model"]=name; block["seed"]=sd; block["role"]=role; block["raw_probability"]=values
                seed_prediction_rows.append(block)
                if role in ("evaluation","exploratory_later"):
                    seed_metric_rows.append({"profile":profile,"model":name,"seed":sd,"role":role,
                        "AP":average_precision_score(y[masks[role]],values),"ROC_AUC":roc_auc_score(y[masks[role]],values),
                        "n":int(masks[role].sum()),"positives":int(y[masks[role]].sum())})
    # Persistence is a deterministic administrative-history baseline.
    persistence_idx=dnames.index("any_inc") if "any_inc" in dnames else dnames.index("inc_count")
    for role in raw:
        v=(xd0[masks[role],-1,persistence_idx].astype(float)>0).astype(float)
        raw[role]["persistence"]=[v]
    all_models=models+["persistence"]; metric_rows=[]; pred_rows=[]
    for name in all_models:
        ens={role:np.mean(raw[role][name],axis=0) for role in raw}
        calibrator=Platt().fit(ens["calibration"],y[masks["calibration"]])
        joblib.dump(calibrator,pdir/f"calibrator_{name}.joblib")
        cal={role:calibrator.predict(ens[role]) for role in ens}
        threshold=best_f1_threshold(y[masks["threshold"]],cal["threshold"])
        for role in ("evaluation","exploratory_later"):
            mm=metrics(y[masks[role]],cal[role],threshold); mm.update({"profile":profile,"model":name,"role":role}); metric_rows.append(mm)
            block=meta.loc[masks[role],["sample_id","platform_id","feature_year","target_year","target"]].copy()
            block["profile"]=profile; block["model"]=name; block["role"]=role; block["probability"]=cal[role]; block["threshold"]=threshold
            pred_rows.append(block)
    metrics_df=pd.DataFrame(metric_rows); predictions=pd.concat(pred_rows,ignore_index=True)
    metrics_df.to_csv(pdir/"metrics.csv",index=False); predictions.to_csv(pdir/"predictions.csv",index=False)
    pd.concat(seed_prediction_rows,ignore_index=True).to_csv(pdir/"seed_predictions_raw.csv",index=False)
    pd.DataFrame(seed_metric_rows).to_csv(pdir/"seed_metrics_raw.csv",index=False)
    eval_pred=predictions[predictions.role.eq("evaluation")]
    paired_cluster_bootstrap(eval_pred,meta,cfg.bootstrap_replicates if not cfg.smoke else 20).to_csv(pdir/"paired_platform_cluster_bootstrap.csv",index=False)
    te=eval_pred[eval_pred.model.eq("tecr-net")]; rel=plot_reliability(te.target.to_numpy(),te.probability.to_numpy(),pdir/"reliability_tecr_net.png"); rel.to_csv(pdir/"reliability_tecr_net.csv",index=False)
    if profile=="full_37":
        variants=["no_wide","no_neural_residual","no_temporal_attention","no_static","no_categorical"]
        if cfg.smoke: variants=["no_neural_residual"]
        ablation_metrics=[]; ablation_predictions=[]
        full_eval=metrics_df[(metrics_df.model=="tecr-net")&(metrics_df.role=="evaluation")].copy(); full_eval["variant"]="full"
        ablation_metrics.append(full_eval)
        for variant in variants:
            LOG.info("Corrected-cohort ablation: %s",variant)
            vr={role:[] for role in raw}
            for sd in (SEEDS if not cfg.smoke else SEEDS[:1]):
                mdl,dev,_=fit_neural(best["tecr-net"],arrays(refit),arrays(masks["calibration"]),y[refit],y[masks["calibration"]],sd,cfg,
                    False,int(best["tecr-net"].get("fixed_epochs",20)),variant)
                torch.save({"state_dict":mdl.state_dict(),"params":best["tecr-net"],"variant":variant,
                    "dynamic":dnames,"static":snames,"categorical":cnames},pdir/f"ablation_{variant}_seed_{sd}.pt")
                for role in vr: vr[role].append(neural_predict(mdl,arrays(masks[role]),dev))
            ens={role:np.mean(vr[role],axis=0) for role in vr}; cali=Platt().fit(ens["calibration"],y[masks["calibration"]])
            calibrated={role:cali.predict(ens[role]) for role in ens}; thr=best_f1_threshold(y[masks["threshold"]],calibrated["threshold"])
            mm=metrics(y[masks["evaluation"]],calibrated["evaluation"],thr); mm.update({"profile":profile,"variant":variant,"role":"evaluation"})
            ablation_metrics.append(pd.DataFrame([mm]))
            b=meta.loc[masks["evaluation"],["sample_id","platform_id","feature_year","target_year","target"]].copy()
            b["variant"]=variant; b["probability"]=calibrated["evaluation"]; b["threshold"]=thr; ablation_predictions.append(b)
        pd.concat(ablation_metrics,ignore_index=True).to_csv(pdir/"corrected_cohort_ablation_metrics.csv",index=False)
        if ablation_predictions: pd.concat(ablation_predictions,ignore_index=True).to_csv(pdir/"corrected_cohort_ablation_predictions.csv",index=False)
    return metrics_df


def main(cfg: Config):
    LOG.info("Corrected symmetric rerun package version %s", PACKAGE_VERSION)
    out=Path(cfg.output_dir); out.mkdir(parents=True,exist_ok=True)
    panel=find_panel(cfg.panel); LOG.info("Reading corrected panel: %s",panel)
    df,_,_=normalize_panel(panel,cfg.allow_2024_feature_year)
    xd,xs,xc,meta=exact_sequences(df); meta.to_csv(out/"sequence_index.csv",index=False)
    audit=audit_corrected_cohort(df,meta,out,cfg.expected_sequences,cfg.raw_platform_inc)
    if not audit["history_count_matches"] and not cfg.allow_cohort_mismatch:
        raise RuntimeError(
            f"Corrected cohort has {len(meta):,} histories, expected {cfg.expected_sequences:,}. "
            "Audit files were written. Reconcile eligibility and label construction before training; "
            "use --allow-cohort-mismatch only for a clearly labelled diagnostic run."
        )
    if cfg.audit_only:
        print(json.dumps(audit,indent=2)); print("AUDIT ONLY: no models were fitted"); return
    masks=split_masks(meta,cfg.allow_2024_feature_year); summary=validate_splits(meta,masks); summary.to_csv(out/"split_summary.csv",index=False); print(summary.to_string(index=False))
    protocol={"fit":"2014-2017","tune":"2018","calibration":"2019","threshold":"2020","evaluation":"2021",
              "exploratory_later":"2022-2023","seeds":SEEDS,"tuning_seeds":TUNING_SEEDS,"trials":cfg.trials,
              "target":"recorded INC in t+1","independent_validation":False,"panel":str(panel),"panel_sha256":sha256(panel)}
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2))
    frames=[]
    for profile in [x.strip() for x in cfg.profiles.split(",") if x.strip()]: frames.append(run_profile(profile,xd,xs,xc,meta,masks,cfg,out))
    pd.concat(frames,ignore_index=True).to_csv(out/"all_metrics.csv",index=False)
    manifest=[]
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name!="manifest_sha256.csv": manifest.append({"path":str(p.relative_to(out)),"bytes":p.stat().st_size,"sha256":sha256(p)})
    pd.DataFrame(manifest).to_csv(out/"manifest_sha256.csv",index=False)
    readme=("COMPLETED corrected-cohort symmetric developmental rerun.\n"
            "The 2021 evaluation is held out from tuning, calibration and threshold selection, but is not claimed as an untouched prospective cohort.\n"
            "The 2022-2023 cohort is exploratory because it was previously inspected.\n")
    (out/"RESULT_INTERPRETATION.txt").write_text(readme)
    import shutil
    archive=shutil.make_archive(str(out),"zip",root_dir=out)
    print("Outputs:",out); print("Downloadable ZIP:",archive)


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    cfg=parse_args()
    if cfg.smoke: cfg.trials=2; cfg.epochs=3; cfg.patience=2; cfg.bootstrap_replicates=20
    if cfg.device.startswith("cuda") and not torch.cuda.is_available(): LOG.warning("CUDA unavailable; using CPU"); cfg.device="cpu"
    main(cfg)
