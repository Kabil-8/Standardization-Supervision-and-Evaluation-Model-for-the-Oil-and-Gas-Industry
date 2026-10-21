"""Corrected TECR-Net Experiment B: disjoint model selection, threshold, calibration,
previously-inspected exploratory evaluation, and byte-level inference replay.

Kaggle: %run /kaggle/working/BSEE_Experiment_B_TECR_P5_EndToEnd.py
INPUT: /kaggle/working/BSEE_Experiment_B_Reconstruction/risk_set_all_platform_years.csv.gz
CAUTION: outcome completeness and source as-of availability are NOT verified.
"""
from pathlib import Path
import json, hashlib, random, copy
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score,roc_auc_score,brier_score_loss,
                             log_loss,f1_score,precision_score,recall_score,confusion_matrix)

ROOT=Path('/kaggle/working/BSEE_Experiment_B_Reconstruction')
OUT=Path('/kaggle/working/BSEE_Experiment_B_Model')
OUT.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT/'risk_set_all_platform_years.csv.gz'
if not SOURCE.exists(): raise FileNotFoundError(f'Run panel reconstruction first: {SOURCE}')
SEEDS=[11,22,33,44,55,66,77]
EPOCHS=60
PATIENCE=10
BATCH=256
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
# Include the current-year INC observation and prior histories; all represent proxy data,
# not proven published or available at Dec 31 t.
NUMERIC=['active_months','has_production_record','producing_wells_mean',
 'producing_wells_max','bopd_mean','bopd_max','mcfd_mean','mcfd_max',
 'boepd_mean','bwpd_mean','platform_age','inc_count_for_history',
 'inc_enforcement_types','inc_count_tminus1','inc_count_tminus2',
 'inc_count_tminus3','inc_count_previous_3_calendar_years',
 'operator_facilities_inspected','operator_inspection_records',
 'operator_inspection_types','operator_warning_incs',
 'operator_component_shutin_incs','operator_facility_shutin_incs']
CATEGORICAL=['area_code','region_code']
SPLITS={'fit':(2014,2017),'architecture_selection':(2018,2018),
        'threshold_selection':(2019,2019),'calibration':(2020,2021),
        'previously_inspected_exploratory':(2022,2024)}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''): h.update(block)
 return h.hexdigest()

def set_seed(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def mask_flags(s):return s.astype(str).str.lower().eq('true')

print('STEP 1/6: Cohorts, chronological sequence keys, and fitted-only preprocessing')
panel=pd.read_csv(SOURCE,dtype={'platform_id':str},low_memory=False)
assert panel[['platform_id','year']].duplicated().sum()==0
assert (panel.target_year==panel.year+1).all()
missing=set(NUMERIC+CATEGORICAL+['provisional_binary_target','calendar_3year_available'])-set(panel.columns)
if missing: raise ValueError(f'Missing columns: {missing}')
panel=panel.sort_values(['platform_id','year']).reset_index(drop=True)
panel['year']=panel.year.astype(int)
for c in NUMERIC:
 panel[c]=pd.to_numeric(panel[c],errors='coerce').replace([np.inf,-np.inf],np.nan).astype('float64')
for c in CATEGORICAL:panel[c]=panel[c].astype('string').fillna('<MISSING>')
panel['target']=pd.to_numeric(panel.provisional_binary_target,errors='coerce')
assert set(panel.target.dropna().unique()).issubset({0.,1.})
# Build exactly 3 years t-2, t-1, t; do not use sequence indices as calendar lags.
idx=pd.MultiIndex.from_frame(panel[['platform_id','year']])
assert idx.is_unique
row_lookup=pd.Series(np.arange(len(panel)),index=idx)
meta=[]; seq_indices=[]
for r in panel.itertuples(index=False):
 if not (2014<=r.year<=2024) or pd.isna(r.target):continue
 keys=[(r.platform_id,r.year-2),(r.platform_id,r.year-1),(r.platform_id,r.year)]
 if not all(k in row_lookup.index for k in keys): continue
 if not bool(r.calendar_3year_available):continue
 seq_indices.append(row_lookup.loc[keys].to_numpy(dtype=int))
 meta.append((r.platform_id,int(r.year),int(r.year+1),int(r.target)))
meta=pd.DataFrame(meta,columns=['platform_id','feature_year','target_year','y_true'])
seq_indices=np.asarray(seq_indices,dtype=np.int64)
assert len(meta)>0 and seq_indices.shape==(len(meta),3)
assert not meta.duplicated(['platform_id','feature_year']).any()
meta['split']='excluded'
for name,(a,b) in SPLITS.items():meta.loc[meta.feature_year.between(a,b),'split']=name
assert (meta.split!='excluded').all()
for name in SPLITS:
 subset=meta[meta.split==name]
 if not len(subset) or subset.y_true.nunique()!=2:raise ValueError(f'{name} has insufficient classes: {subset.y_true.value_counts()}')
print(pd.crosstab(meta.split,meta.y_true).to_string())

fit_seq=seq_indices[meta.split.eq('fit').to_numpy()]
# Fit imputers/scaler only on observations in fitting sequences, no 2018+ data.
fit_rows=np.unique(fit_seq.ravel())
fit_numeric=panel.loc[fit_rows,NUMERIC].to_numpy(dtype=np.float64)
imputer=SimpleImputer(strategy='median',keep_empty_features=True)
scaler=RobustScaler()
scaler.fit(imputer.fit_transform(fit_numeric))
num=scaler.transform(imputer.transform(panel[NUMERIC].to_numpy(dtype=np.float64)))
num=np.asarray(np.clip(num,-30,30),dtype=np.float32)
vocab={}
cat_arrays=[]
for c in CATEGORICAL:
 words=panel[c].astype(str)
 fit_v=sorted(set(words.iloc[fit_rows]))
 vocab[c]={v:i+1 for i,v in enumerate(fit_v)}
 cat_arrays.append(words.map(vocab[c]).fillna(0).to_numpy(dtype=np.int64))
cat=np.stack(cat_arrays,axis=1)
X=num[seq_indices]
C=cat[seq_indices]
y=meta.y_true.to_numpy(dtype=np.float32)
indices={name:np.flatnonzero(meta.split.to_numpy()==name) for name in SPLITS}

class TECR(nn.Module):
 def __init__(self,n_features,cat_sizes,hidden=64,dropout=.25):
  super().__init__()
  self.gate=nn.Parameter(torch.zeros(n_features))
  self.gru=nn.GRU(n_features,hidden,batch_first=True,bidirectional=True)
  self.att=nn.Sequential(nn.Linear(2*hidden,64),nn.Tanh(),nn.Linear(64,1))
  self.emb=nn.ModuleList([nn.Embedding(size+1,min(12,max(4,int(np.sqrt(size+1))))) for size in cat_sizes])
  total_emb=3*sum(e.embedding_dim for e in self.emb)
  self.head=nn.Sequential(nn.Linear(2*hidden+total_emb,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,1))
 def forward(self,x,c):
  states,_=self.gru(x*torch.sigmoid(self.gate)[None,None,:])
  attention=torch.softmax(self.att(states),dim=1)
  pooled=(attention*states).sum(dim=1)
  # c shape: batch x 3 x number_of_categoricals
  embeddings=torch.cat([emb(c[:,:,j]).flatten(1) for j,emb in enumerate(self.emb)],dim=1)
  return self.head(torch.cat((pooled,embeddings),dim=1)).squeeze(-1)

ARCH={'n_features':len(NUMERIC),'cat_sizes':[len(vocab[c]) for c in CATEGORICAL], 'hidden':64,'dropout':.25}

def predict_single(model,positions):
 model.eval(); out=[]
 with torch.no_grad():
  for start in range(0,len(positions),1024):
   ids=positions[start:start+1024]
   xb=torch.as_tensor(X[ids],device=DEVICE)
   cb=torch.as_tensor(C[ids],device=DEVICE)
   out.append(torch.sigmoid(model(xb,cb)).cpu().numpy())
 return np.concatenate(out) if out else np.array([],dtype=float)

print('STEP 2/6: Fit seven TECR-Net checkpoints; 2018 early-stopping only')
checkpoints=[]; seed_log=[]
for seed in SEEDS:
 set_seed(seed)
 model=TECR(**ARCH).to(DEVICE)
 pos=float(y[indices['fit']].sum()); neg=float(len(indices['fit'])-pos)
 criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/max(pos,1)],device=DEVICE))
 optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 ds=TensorDataset(torch.from_numpy(X[indices['fit']]),torch.from_numpy(C[indices['fit']]),
                  torch.from_numpy(y[indices['fit']]))
 loader=DataLoader(ds,batch_size=BATCH,shuffle=True)
 best_score=-float('inf'); best_epoch=0; best_state=None
 for epoch in range(1,EPOCHS+1):
  model.train()
  for xb,cb,yb in loader:
   xb,cb,yb=xb.to(DEVICE),cb.to(DEVICE),yb.to(DEVICE)
   optimizer.zero_grad(); logits=model(xb,cb)
   loss=criterion(logits,yb);loss.backward()
   nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
  v=predict_single(model,indices['architecture_selection'])
  ap=average_precision_score(y[indices['architecture_selection']],v)
  if ap>best_score+1e-7:
   best_score=float(ap);best_epoch=epoch
   best_state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
  if epoch-best_epoch>=PATIENCE:break
 assert best_state is not None
 file=OUT/f'corrected_tecr_seed_{seed}.pt'
 torch.save({'state_dict':best_state,'arch':ARCH,'seed':seed,'best_epoch':best_epoch,
             'selection_pr_auc':best_score,'model_version':'tecr_b_provisional_v1'},file)
 checkpoints.append(file)
 seed_log.append({'seed':seed,'best_epoch':best_epoch,'selection_pr_auc':best_score,'checkpoint_sha256':sha(file)})
 print(f'Seed {seed}: epoch {best_epoch} / selection AP {best_score:.5f}')
pd.DataFrame(seed_log).to_csv(OUT/'seven_seed_selection_log.csv',index=False)

print('STEP 3/6: Frozen seven-seed raw probabilities and disjoint 2019 threshold selection')
score_positions=np.flatnonzero(meta.split.ne('fit').to_numpy())
ensemble=np.zeros(len(score_positions),dtype=float)
for file in checkpoints:
 obj=torch.load(file,map_location='cpu',weights_only=True)
 mdl=TECR(**obj['arch']).to(DEVICE)
 mdl.load_state_dict(obj['state_dict']); ensemble+=predict_single(mdl,score_positions)/len(checkpoints)
raw=np.full(len(meta),np.nan)
raw[score_positions]=ensemble
# Threshold selected only using 2019 raw probabilities; the 2020-21 calibrator is not fit yet.
threshold_indices=indices['threshold_selection']
r=raw[threshold_indices]; truth=y[threshold_indices].astype(int)
candidates=np.unique(np.r_[0.,r,1.])
selection_f1=np.array([f1_score(truth,r>=t,zero_division=0) for t in candidates])
best_idx=int(selection_f1.argmax()); raw_threshold=float(candidates[best_idx])
print('2019 raw threshold:',raw_threshold,'F1:',selection_f1[best_idx])

print('STEP 4/6: Platt calibration on 2020-2021 only; apply unchanged elsewhere')
cal_idx=indices['calibration']
# Calibrate logit of the raw ensemble probability with a 2-parameter sigmoid.
def raw_logit(probs):
 p=np.clip(np.asarray(probs,dtype=float),1e-6,1-1e-6)
 return np.log(p/(1-p)).reshape(-1,1)
calibrator=LogisticRegression(C=1e6,max_iter=3000,random_state=17)
calibrator.fit(raw_logit(raw[cal_idx]),y[cal_idx].astype(int))
# A Platt mapping is monotone when its fitted slope is positive.
slope=float(calibrator.coef_.ravel()[0])
if slope<=0:raise RuntimeError('Platt slope non-positive. Raw threshold cannot be mapped monotonically.')
calibrated=np.full(len(meta),np.nan)
calibrated[score_positions]=calibrator.predict_proba(raw_logit(raw[score_positions]))[:,1]
cal_threshold=float(calibrator.predict_proba(raw_logit([raw_threshold]))[0,1])
# Invariant: raw and calibrated decisions match EXACTLY due to monotonic calibration.
assert np.array_equal(raw[score_positions]>=raw_threshold,
                      calibrated[score_positions]>=cal_threshold)
joblib.dump({'imputer':imputer,'scaler':scaler,'vocab':vocab,'numeric':NUMERIC,'categorical':CATEGORICAL},OUT/'preprocessing.joblib')
joblib.dump(calibrator,OUT/'platt_calibrator.joblib')

print('STEP 5/6: Metrics, reliability, predictions and inference-ready manifest')
rows=[]
for name in ('architecture_selection','threshold_selection','calibration','previously_inspected_exploratory'):
 ids=indices[name]; yy=y[ids].astype(int);p=calibrated[ids];pred=(p>=cal_threshold).astype(int)
 tn,fp,fn,tp=confusion_matrix(yy,pred,labels=[0,1]).ravel()
 rows.append({'split':name,'n':len(ids),'positives':int(yy.sum()),'prevalence':float(yy.mean()),
  'pr_auc':float(average_precision_score(yy,p)),'roc_auc':float(roc_auc_score(yy,p)),
  'brier':float(brier_score_loss(yy,p)),'log_loss':float(log_loss(yy,p,labels=[0,1])),
  'f1':float(f1_score(yy,pred,zero_division=0)),
  'precision':float(precision_score(yy,pred,zero_division=0)),
  'recall':float(recall_score(yy,pred,zero_division=0)),
  'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
  'interpretation':('calibration_apparent' if name=='calibration' else
                    'previously_inspected_not_independent' if name=='previously_inspected_exploratory' else
                    'development_not_independent')})
report=pd.DataFrame(rows);report.to_csv(OUT/'p5_calibration_metrics.csv',index=False)
preds=meta.iloc[score_positions].copy()
preds['raw_probability']=raw[score_positions]
preds['calibrated_probability']=calibrated[score_positions]
preds['raw_threshold']=raw_threshold
preds['decision_threshold']=cal_threshold
preds['predicted_positive']=(preds.calibrated_probability>=cal_threshold).astype(int)
preds['model_family']='TECR-Net';preds['model_version']='tecr_b_provisional_v1'
preds['fit_train_max_feature_year']=2017
preds['predictions_are_out_of_sample']=True
preds.to_csv(OUT/'corrected_oof_predictions.csv',index=False)
bins=[]
for name in ('calibration','previously_inspected_exploratory'):
 ids=indices[name]; temp=pd.DataFrame({'prob':calibrated[ids],'label':y[ids].astype(int)})
 temp['bin']=pd.cut(temp.prob,np.linspace(0,1,11),include_lowest=True)
 tmp=temp.groupby('bin',observed=False).agg(n=('label','size'),mean_probability=('prob','mean'),observed_rate=('label','mean')).reset_index()
 tmp['bin']=tmp['bin'].astype(str);tmp['split']=name;bins.append(tmp)
pd.concat(bins,ignore_index=True).to_csv(OUT/'reliability_bins.csv',index=False)
manifest={'model_version':'tecr_b_provisional_v1','device_used_for_training':DEVICE,
 'seeds':SEEDS,'architecture':ARCH,'numeric_features':NUMERIC,'categorical_features':CATEGORICAL,
 'chronological_splits':{k:list(v) for k,v in SPLITS.items()},
 'threshold_selection_rule':'maximize F1 on raw 2019 ensemble predictions, before calibration fitting',
 'raw_threshold_2019':raw_threshold,'calibrated_threshold':cal_threshold,
 'calibrator_fit_years':[2020,2021],'calibrator_slope':slope,
 'calibration_fit_is_apparent':True,'later_years_previously_inspected':True,
 'outcome_completeness_verified':False,'historical_publication_asof_verified':False,
 'independent_mature_outcome_evaluation_pass':False,
 'calibration_or_test_independence_claim':'None: chronological separation only; data and labels already investigated',
 'checkpoint_sha256':{p.name:sha(p) for p in checkpoints},
 'input_panel_sha256':sha(SOURCE)}
for name in ('preprocessing.joblib','platt_calibrator.joblib','corrected_oof_predictions.csv','p5_calibration_metrics.csv'):
 manifest.setdefault('artifact_sha256',{})[name]=sha(OUT/name)
(OUT/'inference_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(report.to_string(index=False))

print('STEP 6/6: Independently reload checkpoints + preprocessing + calibrator and replay inference')
saved=joblib.load(OUT/'preprocessing.joblib')
saved_cal=joblib.load(OUT/'platt_calibrator.joblib')
assert saved['numeric']==NUMERIC and saved['categorical']==CATEGORICAL
# Replay preprocessing from original unmodified panel columns.
re_num=saved['scaler'].transform(saved['imputer'].transform(panel[NUMERIC].to_numpy(dtype=np.float64)))
re_num=np.clip(re_num,-30,30).astype(np.float32)
assert np.array_equal(re_num,num)
re_c=np.stack([panel[c].astype(str).map(saved['vocab'][c]).fillna(0).to_numpy(np.int64) for c in CATEGORICAL],axis=1)
assert np.array_equal(re_c,cat)
replayed=np.zeros(len(score_positions),dtype=float)
for file in checkpoints:
 obj=torch.load(file,map_location='cpu',weights_only=True)
 mdl=TECR(**obj['arch']).to(DEVICE);mdl.load_state_dict(obj['state_dict'])
 replayed+=predict_single(mdl,score_positions)/len(checkpoints)
replay_cal=saved_cal.predict_proba(raw_logit(replayed))[:,1]
max_err=float(np.max(np.abs(replay_cal-calibrated[score_positions])))
assert max_err<1e-7,('Round-trip probability mismatch',max_err)
assert np.array_equal(replay_cal>=cal_threshold,preds.predicted_positive.to_numpy())
assert preds[['platform_id','feature_year']].duplicated().sum()==0
print('REPLAY MAX ABS ERROR:',max_err)
print('THRESHOLD CONSISTENCY: PASS')
print('ARTIFACT RELOAD + PROBABILITY REPLAY: PASS')
print('MODEL: Corrected 7-seed TECR-Net, proxy outcome only')
print('INDEPENDENT CONFIRMATORY EVALUATION: PENDING')
print('OUTPUT:',OUT)
