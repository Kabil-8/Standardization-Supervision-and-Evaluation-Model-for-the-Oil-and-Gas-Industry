"""Leakage-conscious, single-file BSEE modelling pipeline for Kaggle.

Kaggle usage (Internet OFF is fine):
    1. Attach the BSEE dataset containing the six files in REQUIRED_FILES.
    2. Upload this script to the notebook.
    3. Run:  !python /kaggle/input/<script-dataset>/TECR_Net_Kaggle_End_to_End.py

Outputs are written to /kaggle/working/tecr_net_corrected.  Years 2022--2024
are deliberately labelled EXPLORATORY because they were previously inspected.
They must not be described as an independent test.  A future, genuinely unseen
cohort must be evaluated once, without changing this pipeline.
"""
from __future__ import annotations

import argparse, hashlib, json, logging, math, os, random, sys, time, warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LOG = logging.getLogger("tecr-net")
REQUIRED_FILES = {
    "production": "mv_prod_by_platform_all.txt",
    "structures": "mv_platstruc_structures.txt",
    "platform_incs": "mv_platstruc_platformincs.txt",
    "operator_inspections": "mv_insp_by_oper.txt",
    "operator_incs": "mv_incs_by_oper.txt",
    "accidents": "mv_acc_investigations.txt",
}
NA = ["", " ", "NA", "N/A", "NULL", "None", "UNKNOWN"]
NUMERIC = ["active_months", "has_production_record", "producing_wells_mean",
    "producing_wells_max", "bopd_mean", "bopd_max", "mcfd_mean", "mcfd_max",
    "boepd_mean", "bwpd_mean", "platform_age", "inc_count",
    "enforcement_code_count", "investigated_accident_count", "accident_type_count",
    "operator_facilities_inspected", "operator_inspection_records",
    "operator_inspection_types", "operator_warning_incs",
    "operator_component_shutin_incs", "operator_facility_shutin_incs",
    "previous_year_inc_count", "inc_count_previous_3_calendar_years",
    "previous_year_accident_count", "bopd_change", "mcfd_change"]
CATEGORICAL = ["area_code", "region_code"]
SEEDS = [11, 22, 33, 44, 55, 66, 77]


@dataclass
class Config:
    raw_dir: str = "/kaggle/input"
    output_dir: str = "/kaggle/working/tecr_net_corrected"
    chunksize: int = 250_000
    bootstrap_replicates: int = 1000
    epochs: int = 80
    patience: int = 12
    batch_size: int = 256
    # Kaggle GPU is used automatically when enabled in Notebook Settings.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def clean(s): return s.astype("string").str.strip().replace("", pd.NA)
def ident(s): return clean(s).str.replace(r"\.0$", "", regex=True)
def pkey(a, b):
    a, b = ident(a), ident(b)
    return (a + "::" + b).where(a.notna() & b.notna())
def nums(df, cols):
    for c in cols: df[c] = pd.to_numeric(df[c], errors="coerce")
def dates(s):
    """Parse heterogeneous BSEE date strings without pandas format warnings."""
    try:
        return pd.to_datetime(s, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        # Compatibility fallback for older pandas releases without format=mixed.
        return pd.to_datetime(s, errors="coerce")
def read(path, usecols=None):
    return pd.read_csv(path, usecols=usecols, dtype="string", encoding="cp1252",
                       na_values=NA, keep_default_na=True, low_memory=False)
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def locate(root):
    found = {}
    for key, name in REQUIRED_FILES.items():
        matches = list(Path(root).rglob(name))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one {name} below {root}; found {matches}")
        found[key] = matches[0]
    return found


def production(path, chunksize):
    cols = ["COMPLEX_ID_NUM","STRUCTURE_NUMBER","AREA_CODE","PF_OPERATOR_NUM",
            "PRODUCTION_DATE","PRODUCING_WELLS","BOPD","MCFPD","BOEPD","BWPD","REGION_CODE"]
    parts = []
    for x in pd.read_csv(path, usecols=cols, dtype="string", encoding="cp1252",
                         na_values=NA, chunksize=chunksize, low_memory=False):
        x["platform_id"] = pkey(x.COMPLEX_ID_NUM, x.STRUCTURE_NUMBER)
        x["date"] = dates(x.PRODUCTION_DATE)
        x["year"] = x.date.dt.year
        nums(x, ["PRODUCING_WELLS","BOPD","MCFPD","BOEPD","BWPD"])
        x = x.dropna(subset=["platform_id","year"]).sort_values("date")
        g = x.groupby(["platform_id","year"], observed=True)
        a = g.agg(active_months=("date","nunique"), n=("date","size"),
            producing_wells_sum=("PRODUCING_WELLS","sum"), producing_wells_n=("PRODUCING_WELLS","count"),
            producing_wells_max=("PRODUCING_WELLS","max"), bopd_sum=("BOPD","sum"), bopd_n=("BOPD","count"),
            bopd_max=("BOPD","max"), mcfd_sum=("MCFPD","sum"), mcfd_n=("MCFPD","count"),
            mcfd_max=("MCFPD","max"), boepd_sum=("BOEPD","sum"), boepd_n=("BOEPD","count"),
            bwpd_sum=("BWPD","sum"), bwpd_n=("BWPD","count"),
            operator_id=("PF_OPERATOR_NUM","last"), area_code=("AREA_CODE","last"),
            region_code=("REGION_CODE","last"), last_date=("date","max")).reset_index()
        parts.append(a)
    x = pd.concat(parts, ignore_index=True).sort_values("last_date")
    sumcols = [c for c in x if c.endswith("_sum") or c.endswith("_n")]
    agg = {c:"sum" for c in ["active_months",*sumcols]}
    agg.update({c:"max" for c in ["producing_wells_max","bopd_max","mcfd_max"]})
    agg.update({c:"last" for c in ["operator_id","area_code","region_code","last_date"]})
    x = x.groupby(["platform_id","year"], observed=True).agg(agg).reset_index()
    for stem in ["producing_wells","bopd","mcfd","boepd","bwpd"]:
        x[f"{stem}_mean"] = x[f"{stem}_sum"] / x[f"{stem}_n"].replace(0,np.nan)
    return x.drop(columns=sumcols+["last_date"]).assign(year=lambda z:z.year.astype(int))


def structures(path):
    x = read(path, ["COMPLEX_ID_NUM","STRUCTURE_NUMBER","INSTALL_DATE","REMOVAL_DATE"])
    x["platform_id"] = pkey(x.COMPLEX_ID_NUM, x.STRUCTURE_NUMBER)
    x["install_date"] = dates(x.INSTALL_DATE)
    x["removal_date"] = dates(x.REMOVAL_DATE)
    x = x.dropna(subset=["platform_id"]).sort_values(["platform_id","install_date"])
    # Dates define eligibility only; contemporaneously unverified structure fields are excluded.
    return x.groupby("platform_id", as_index=False).agg(install_date=("install_date","min"), removal_date=("removal_date","max"))


def platform_incs(path):
    x = read(path, ["ID_NUMBER","STRUCTURE_NUMBER","INC_NUMBER","INC_ENFRC_CODE","MMS_COMPANY_NUM","INC_ISSUE_DATE"])
    x["platform_id"] = pkey(x.ID_NUMBER, x.STRUCTURE_NUMBER)
    x["date"] = dates(x.INC_ISSUE_DATE); x["year"] = x.date.dt.year
    x["inc_id"] = ident(x.INC_NUMBER); x["enforcement"] = clean(x.INC_ENFRC_CODE)
    x["operator_id_inc"] = ident(x.MMS_COMPANY_NUM)
    x = x.dropna(subset=["platform_id","year"])
    # Same INC may appear more than once; count a stable INC identifier once.
    # pandas 3.x rejects an Index as the fill value.  Build an aligned Series
    # so records without an INC number still receive a stable row-level key.
    fallback_key = pd.Series(
        x.index.astype(str),
        index=x.index,
        dtype="string",
    )
    x["inc_key"] = x["inc_id"].fillna(fallback_key)
    x = x.drop_duplicates(["platform_id","year","inc_key"])
    return x.groupby(["platform_id","year"], observed=True).agg(
        inc_count=("inc_key","nunique"), enforcement_code_count=("enforcement","nunique"),
        operator_id_inc=("operator_id_inc","last")).reset_index().assign(year=lambda z:z.year.astype(int))


def operator_activity(insp_path, inc_path):
    a = read(insp_path); a["year"] = dates(a["INS_DATE"]).dt.year
    a["operator_id"] = ident(a["BUS_ASC_CODE"])
    ac = ["FACILITIES_INSPECTED","INSPECTION_RECORDS","INSPECTION_TYPES"]; nums(a,ac)
    a = a.dropna(subset=["operator_id","year"]).groupby(["operator_id","year"],observed=True)[ac].sum(min_count=1).reset_index()
    a = a.rename(columns=dict(zip(ac,["operator_facilities_inspected","operator_inspection_records","operator_inspection_types"])))
    b = read(inc_path); b["year"] = dates(b["INC_DATE"]).dt.year
    b["operator_id"] = ident(b["BUS_ASC_CODE"])
    bc=["WARNING_INCS","COMP_SHUTIN_INCS","FAC_SHUTIN_INCS"]; nums(b,bc)
    b=b.dropna(subset=["operator_id","year"]).groupby(["operator_id","year"],observed=True)[bc].sum(min_count=1).reset_index()
    b=b.rename(columns=dict(zip(bc,["operator_warning_incs","operator_component_shutin_incs","operator_facility_shutin_incs"])))
    z=a.merge(b,on=["operator_id","year"],how="outer"); z.year=z.year.astype(int); return z


def accidents(path):
    x=read(path); x["year"]=dates(x["DATE_OCCURRED"]).dt.year
    x["lease_number"]=ident(x["LEASE_NUMBER"])
    return x.dropna(subset=["lease_number","year"]).groupby(["lease_number","year"],observed=True).agg(
        investigated_accident_count=("ACCIDENT_TYPE","size"), accident_type_count=("ACCIDENT_TYPE","nunique")).reset_index()


def build_panel(files, cfg, out):
    prod=production(files["production"],cfg.chunksize); st=structures(files["structures"]); inc=platform_incs(files["platform_incs"])
    op=operator_activity(files["operator_inspections"],files["operator_incs"])
    # Accident records lack a guaranteed platform key in this release. They are therefore not
    # assigned through a mutable/current lease snapshot; keep the fields missing, not falsely zero.
    _ = accidents(files["accidents"])
    flow=[]; flow.append({"stage":"structure rows","n":len(st),"platforms":st.platform_id.nunique()})
    eligible=st.dropna(subset=["install_date"]).copy(); flow.append({"stage":"known installation date","n":len(eligible),"platforms":eligible.platform_id.nunique()})
    rows=[]
    for r in eligible.itertuples():
        lo=max(2011,int(r.install_date.year)); hi=min(2024,int(r.removal_date.year) if pd.notna(r.removal_date) else 2024)
        if lo<=hi: rows.extend((r.platform_id,y,r.install_date,r.removal_date) for y in range(lo,hi+1))
    p=pd.DataFrame(rows,columns=["platform_id","year","install_date","removal_date"])
    p=p.merge(prod,on=["platform_id","year"],how="left",validate="one_to_one")
    p["has_production_record"]=p.active_months.notna().astype(float)
    p=p.merge(inc,on=["platform_id","year"],how="left",validate="one_to_one")
    p["operator_id"]=ident(p.operator_id).fillna(ident(p.operator_id_inc))
    for c in ["inc_count","enforcement_code_count"]: p[c]=p[c].fillna(0.0)
    p=p.merge(op,on=["operator_id","year"],how="left",validate="many_to_one")
    p["investigated_accident_count"]=np.nan; p["accident_type_count"]=np.nan
    p["platform_age"]=p.year-p.install_date.dt.year
    p=p.sort_values(["platform_id","year"]); g=p.groupby("platform_id",sort=False)
    p["previous_year_inc_count"]=g.inc_count.shift(1)
    p["inc_count_previous_3_calendar_years"]=sum(g.inc_count.shift(k) for k in (1,2,3))
    p["previous_year_accident_count"]=g.investigated_accident_count.shift(1)
    p["bopd_change"]=p.bopd_mean/g.bopd_mean.shift(1)-1; p["mcfd_change"]=p.mcfd_mean/g.mcfd_mean.shift(1)-1
    nxt=inc[["platform_id","year","inc_count"]].copy(); nxt.year-=1; nxt=nxt.rename(columns={"inc_count":"next_year_inc_count"})
    p=p.merge(nxt,on=["platform_id","year"],how="left"); p["next_year_inc_count"]=p.next_year_inc_count.fillna(0)
    p["target"]=(p.next_year_inc_count>0).astype(int)
    p["role"]=np.select([p.year.between(2014,2017),p.year.between(2018,2019),p.year.between(2020,2021),p.year.between(2022,2024)],
        ["inner_fit","inner_tuning","calibration","exploratory_previously_inspected"],default="history_only")
    flow.append({"stage":"complete annual risk set 2011-2024","n":len(p),"platforms":p.platform_id.nunique()})
    pd.DataFrame(flow).to_csv(out/"cohort_flow.csv",index=False)
    return p


class Prep:
    @staticmethod
    def _float_frame(p):
        """Return ordinary float64 columns, avoiding nullable-Int clip errors."""
        numeric = p.reindex(columns=NUMERIC).apply(
            pd.to_numeric,
            errors="coerce",
        ).astype("float64")
        # Production-change ratios can be infinite when the preceding year's
        # production is zero. Treat those ratios as missing before quantiles.
        return numeric.replace([np.inf, -np.inf], np.nan)

    def fit(self,p):
        x=p[p.year.between(2011,2019)]
        numeric = self._float_frame(x)
        self.med=numeric.median(); self.q1=numeric.quantile(.01); self.q99=numeric.quantile(.99)
        clipped = numeric.clip(lower=self.q1, upper=self.q99, axis="columns")
        self.center=clipped.fillna(self.med).median()
        self.scale=(numeric.quantile(.75)-numeric.quantile(.25)).replace(0,1).fillna(1)
        self.vocab={c:{v:i+1 for i,v in enumerate(sorted(x[c].fillna("<MISSING>").astype(str).unique()))} for c in CATEGORICAL}; return self
    def numeric(self,p):
        numeric = self._float_frame(p)
        clipped = numeric.clip(lower=self.q1, upper=self.q99, axis="columns")
        return ((clipped.fillna(self.med)-self.center)/self.scale).replace([np.inf,-np.inf],0).fillna(0).to_numpy(np.float32)
    def cats(self,p): return np.stack([p[c].fillna("<MISSING>").astype(str).map(self.vocab[c]).fillna(0).to_numpy() for c in CATEGORICAL],1).astype(np.int64)


def sequences(panel,prep):
    z=panel.set_index(["platform_id","year"]); rows=[]; xn=[]; xc=[]
    for r in panel[panel.year.between(2014,2024)].itertuples():
        keys=[(r.platform_id,r.year-k) for k in (2,1,0)]
        if all(k in z.index for k in keys):
            q=z.loc[keys].reset_index(); xn.append(prep.numeric(q)); xc.append(prep.cats(q).reshape(-1))
            rows.append((r.platform_id,r.year,r.target,r.role))
    meta=pd.DataFrame(rows,columns=["platform_id","year","target","role"])
    return np.stack(xn),np.stack(xc),meta


class TECR(nn.Module):
    def __init__(self,nnum,vocabs,hidden=48,drop=.25):
        super().__init__(); self.gate=nn.Sequential(nn.Linear(nnum,nnum),nn.Sigmoid())
        self.gru=nn.GRU(nnum,hidden,batch_first=True,bidirectional=True)
        self.att=nn.Linear(hidden*2,1); self.emb=nn.ModuleList([nn.Embedding(v+1,min(12,max(3,int(math.sqrt(v+1))))) for v in vocabs])
        ed=sum(e.embedding_dim for e in self.emb)*3
        self.head=nn.Sequential(nn.Linear(hidden*2+ed,64),nn.ReLU(),nn.Dropout(drop),nn.Linear(64,1))
    def forward(self,x,c):
        h,_=self.gru(x*self.gate(x)); w=torch.softmax(self.att(h),1); h=(w*h).sum(1)
        e=torch.cat([emb(c[:,i::len(self.emb)]).flatten(1) for i,emb in enumerate(self.emb)],1)
        return self.head(torch.cat([h,e],1)).squeeze(1)


def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def train_tecr(x,c,y,fit,tune,prep,cfg,seed,fixed_epochs=None):
    seed_all(seed); dev=torch.device(cfg.device); m=TECR(x.shape[2],[len(prep.vocab[k]) for k in CATEGORICAL]).to(dev)
    pos=max(1,y[fit].sum()); loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(fit)-pos)/pos,device=dev))
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    ds=TensorDataset(torch.from_numpy(x[fit]),torch.from_numpy(c[fit]),torch.from_numpy(y[fit].astype(np.float32)))
    loader=DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(dev.type == "cuda"),
    )
    best=(-np.inf,None,0); limit=fixed_epochs or cfg.epochs
    started = time.perf_counter()
    for epoch in range(limit):
        m.train(); epoch_losses=[]
        for a,b,d in loader:
            a=a.to(dev,non_blocking=True); b=b.to(dev,non_blocking=True); d=d.to(dev,non_blocking=True)
            opt.zero_grad(set_to_none=True); l=loss(m(a,b),d); l.backward(); nn.utils.clip_grad_norm_(m.parameters(),5); opt.step(); epoch_losses.append(float(l.detach().cpu()))
        if fixed_epochs is None:
            pr=predict_tecr(m,x[tune],c[tune],dev); score=average_precision_score(y[tune],pr)
            if score>best[0]+1e-5: best=(score,{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},epoch+1)
            if epoch-best[2]>=cfg.patience: break
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == limit:
            elapsed = time.perf_counter() - started
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan
            if fixed_epochs is None:
                LOG.info("Seed %s epoch %d/%d loss %.5f tuning PR-AUC %.5f elapsed %.1fs", seed, epoch+1, limit, mean_loss, score, elapsed)
            else:
                eta = elapsed / (epoch + 1) * (limit - epoch - 1)
                LOG.info("Seed %s epoch %d/%d loss %.5f elapsed %.1fs ETA %.1fs", seed, epoch+1, limit, mean_loss, elapsed, eta)
    if fixed_epochs is None: m.load_state_dict(best[1])
    return m,best[2] or limit
def predict_tecr(m,x,c,dev):
    m.eval(); ans=[]
    with torch.no_grad():
        for i in range(0,len(x),2048):
            xb=torch.from_numpy(x[i:i+2048]).to(dev,non_blocking=True)
            cb=torch.from_numpy(c[i:i+2048]).to(dev,non_blocking=True)
            ans.append(m(xb,cb).cpu().numpy())
    return np.concatenate(ans) if ans else np.array([])


def calibrator(logits,y):
    m=LogisticRegression(C=1e6,solver="lbfgs",max_iter=2000).fit(np.asarray(logits).reshape(-1,1),y); return m
def choose_threshold(y,p):
    ts=np.unique(np.quantile(p,np.linspace(0,1,501))); scores=[f1_score(y,p>=t,zero_division=0) for t in ts]; return float(ts[int(np.argmax(scores))])
def metrics(y,p,t,ref):
    pred=p>=t; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    safe=lambda f: float(f()) if len(np.unique(y))>1 else np.nan
    return {"n":len(y),"positives":int(y.sum()),"prevalence":float(y.mean()),"average_precision":safe(lambda:average_precision_score(y,p)),
        "roc_auc":safe(lambda:roc_auc_score(y,p)),"f1":f1_score(y,pred,zero_division=0),"precision":precision_score(y,pred,zero_division=0),
        "recall":recall_score(y,pred,zero_division=0),"specificity":tn/max(1,tn+fp),"balanced_accuracy":balanced_accuracy_score(y,pred),
        "mcc":matthews_corrcoef(y,pred),"brier":brier_score_loss(y,p),"brier_skill":1-brier_score_loss(y,p)/(ref*(1-ref)),
        "log_loss":log_loss(y,p,labels=[0,1]),"threshold":t}


def cluster_ci(frame,reps,seed=2026):
    """Platform-clustered bootstrap without repeated DataFrame scans."""
    rng=np.random.default_rng(seed)
    work=frame.reset_index(drop=True)
    ids=work.platform_id.drop_duplicates().to_numpy()
    grouped_indices={
        platform_id: group.index.to_numpy(dtype=np.int64)
        for platform_id,group in work.groupby("platform_id",sort=False)
    }
    y_all=work.target.to_numpy(dtype=np.int8)
    p_all=work.probability.to_numpy(dtype=np.float64)
    threshold=float(work.threshold.iloc[0])
    reference=float(work.reference_prevalence.iloc[0])
    vals=[]
    LOG.info("Starting %d-replicate platform-clustered bootstrap", reps)
    for replicate in range(reps):
        sampled_ids=rng.choice(ids,size=len(ids),replace=True)
        sampled_rows=np.concatenate([grouped_indices[u] for u in sampled_ids])
        sampled_y=y_all[sampled_rows]
        if np.unique(sampled_y).size>1:
            vals.append(metrics(sampled_y,p_all[sampled_rows],threshold,reference))
        if (replicate+1)%100==0 or replicate+1==reps:
            LOG.info("Bootstrap progress: %d/%d", replicate+1, reps)
    rows=[]
    for k in ["average_precision","roc_auc","f1","brier","mcc"]:
        a=np.array([v[k] for v in vals],float); rows.append({"metric":k,"lower_95":np.nanquantile(a,.025),"upper_95":np.nanquantile(a,.975),"bootstrap_unit":"platform"})
    return pd.DataFrame(rows)


def annual_budgets(pred):
    rows=[]
    for year,q in pred.groupby("year"):
        assert q.platform_id.is_unique
        for k in (50,100,200):
            n=min(k,len(q)); top=q.nlargest(n,"probability"); rows.append({"year":year,"requested_k":k,"actual_k":n,"precision_at_k":top.target.mean(),"recall_at_k":top.target.sum()/max(1,q.target.sum()),"ranking_unit":"unique platform within year"})
    return pd.DataFrame(rows)


def main(cfg):
    out=Path(cfg.output_dir); out.mkdir(parents=True,exist_ok=True)
    LOG.info("Locating required BSEE source files below %s", cfg.raw_dir)
    files=locate(cfg.raw_dir)
    manifest={k:{"path":str(v),"sha256":sha256(v),"bytes":v.stat().st_size} for k,v in files.items()}
    LOG.info("Building the annual platform risk set")
    panel=build_panel(files,cfg,out); panel.to_csv(out/"annual_risk_set.csv",index=False)
    LOG.info("Annual risk set: %s rows", f"{len(panel):,}")
    LOG.info("Fitting training-period preprocessing")
    prep=Prep().fit(panel); x,c,meta=sequences(panel,prep); y=meta.target.to_numpy()
    LOG.info("Constructed %s exact consecutive sequences", f"{len(meta):,}")
    meta.to_csv(out/"sequence_index.csv",index=False)
    masks={r:np.flatnonzero(meta.role.eq(r)) for r in meta.role.unique()}
    required=["inner_fit","inner_tuning","calibration","exploratory_previously_inspected"]
    if any(len(masks.get(r,[]))==0 for r in required): raise RuntimeError(f"Empty split: { {r:len(masks.get(r,[])) for r in required} }")
    # Architecture/epoch selection uses only 2014-2019.
    tune_models=[]; epochs=[]
    for s in SEEDS[:3]:
        LOG.info("Selecting epoch count: tuning seed %s of %s", s, SEEDS[:3])
        m,e=train_tecr(x,c,y,masks["inner_fit"],masks["inner_tuning"],prep,cfg,s); tune_models.append(m); epochs.append(e)
    chosen=max(5,int(np.median(epochs))); dev_idx=np.flatnonzero(meta.year.between(2014,2019)); all_logits=[]; models=[]
    LOG.info("Selected fixed epoch count: %d", chosen)
    for s in SEEDS:
        LOG.info("Final %s fit: seed %s (%d/%d)", cfg.device.upper(), s, SEEDS.index(s)+1, len(SEEDS))
        m,_=train_tecr(x,c,y,dev_idx,masks["inner_tuning"],prep,cfg,s,fixed_epochs=chosen); models.append(m); all_logits.append(predict_tecr(m,x,c,torch.device(cfg.device)))
    raw=np.mean(all_logits,axis=0); cal_idx=masks["calibration"]; cal=calibrator(raw[cal_idx],y[cal_idx]); prob=cal.predict_proba(raw.reshape(-1,1))[:,1]
    threshold=choose_threshold(y[cal_idx],prob[cal_idx]); ref=float(y[dev_idx].mean())
    pred=meta.copy(); pred["raw_ensemble_logit"]=raw; pred["probability"]=prob; pred["threshold"]=threshold; pred["reference_prevalence"]=ref
    pred.to_csv(out/"predictions.csv",index=False)
    rows=[]
    for role in ["calibration","exploratory_previously_inspected"]:
        q=pred[pred.role.eq(role)]; z=metrics(q.target.to_numpy(),q.probability.to_numpy(),threshold,ref); z.update(model="TECR-Net corrected",role=role); rows.append(z)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    ex=pred[pred.role.eq("exploratory_previously_inspected")]
    annual_budgets(ex).to_csv(out/"annual_unique_platform_precision_at_k.csv",index=False)
    cluster_ci(ex,cfg.bootstrap_replicates).to_csv(out/"exploratory_platform_cluster_bootstrap_ci.csv",index=False)
    # Simple benchmarks, trained/tuned only on development and calibrated identically.
    flat=np.concatenate([x.reshape(len(x),-1),c.astype(np.float32)],axis=1)
    baselines={"logistic":Pipeline([("impute",SimpleImputer()),("scale",RobustScaler()),("model",LogisticRegression(C=.1,max_iter=3000,class_weight="balanced"))]),
               "hist_gradient_boosting":HistGradientBoostingClassifier(max_iter=250,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1,class_weight="balanced",random_state=11)}
    base_rows=[]
    for name,m in baselines.items():
        LOG.info("Fitting baseline: %s", name)
        m.fit(flat[dev_idx],y[dev_idx]); pp=m.predict_proba(flat)[:,1]; lg=np.log(np.clip(pp,1e-6,1-1e-6)/np.clip(1-pp,1e-6,1)); cm=calibrator(lg[cal_idx],y[cal_idx]); cp=cm.predict_proba(lg.reshape(-1,1))[:,1]; th=choose_threshold(y[cal_idx],cp[cal_idx])
        for role in ["calibration","exploratory_previously_inspected"]:
            ix=masks[role]; z=metrics(y[ix],cp[ix],th,ref); z.update(model=name,role=role); base_rows.append(z)
        joblib.dump(m,out/f"{name}.joblib")
    pd.concat([pd.read_csv(out/"metrics.csv"),pd.DataFrame(base_rows)],ignore_index=True).to_csv(out/"metrics.csv",index=False)
    torch.save({"states":[m.state_dict() for m in models],"numeric_features":NUMERIC,"categorical_features":CATEGORICAL,"epochs":chosen},out/"tecr_net_7seed.pt")
    joblib.dump({"preprocessor":prep,"calibrator":cal,"threshold":threshold},out/"preprocessing_and_calibration.joblib")
    pd.DataFrame([{"role":r,"feature_years":f"{meta[meta.role.eq(r)].year.min()}-{meta[meta.role.eq(r)].year.max()}","n":len(masks[r]),"positives":int(y[masks[r]].sum())} for r in required]).to_csv(out/"split_summary.csv",index=False)
    run={"config":asdict(cfg),"sources":manifest,"versions":{"python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__,"torch":torch.__version__},
         "selected_epochs":chosen,"calibration_method":"logistic recalibration of ensemble logits on 2020-2021","confirmatory_test_available":False,
         "interpretation":"2022-2024 was previously inspected and is exploratory only; obtain a genuinely unseen temporal or external cohort for confirmation.",
         "limitations":["Static structure snapshot predictors excluded pending dated histories.","Accident data not assigned to platforms because the supplied table lacks a guaranteed time-varying platform key.","Absence of an INC row is treated as zero only within the platform INC registry coverage."]}
    (out/"run_manifest.json").write_text(json.dumps(run,indent=2,default=str),encoding="utf-8")
    print(f"Completed. Outputs: {out}\nIMPORTANT: 2022-2024 results are exploratory, not independent test evidence.")


def parse_args(argv=None):
    """Parse CLI options without consuming Kaggle/Jupyter kernel arguments.

    Kaggle injects ``-f <kernel.json>`` when a complete script is pasted into a
    notebook cell.  Normal ``python script.py`` execution remains strict, so a
    misspelled user option still raises an error instead of being ignored.
    """
    ap = argparse.ArgumentParser(
        description="Leakage-conscious BSEE TECR-Net modelling pipeline"
    )
    ap.add_argument("--raw-dir", default="/kaggle/input")
    ap.add_argument(
        "--output-dir",
        default="/kaggle/working/tecr_net_corrected",
    )
    ap.add_argument("--chunksize", type=int, default=250_000)
    ap.add_argument("--bootstrap-replicates", type=int, default=1_000)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # A pasted Kaggle/Jupyter cell must not parse the kernel's ``-f`` option.
    # A separately executed .py file receives its real command-line options.
    if argv is None and "ipykernel" in sys.modules:
        argv = []
    return ap.parse_args(argv)


def notebook_entrypoint():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        LOG.warning("CUDA was requested but is unavailable; using CPU.")
        args.device = "cpu"
    if args.device == "cuda":
        LOG.info("Using GPU: %s", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    else:
        LOG.info("Using CPU")
    main(Config(**vars(args)))


if __name__ == "__main__":
    notebook_entrypoint()
