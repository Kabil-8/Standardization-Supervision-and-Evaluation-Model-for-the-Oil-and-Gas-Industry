"""Frozen corrected TECR-Net inference. Imports neither training code nor labels.
Kaggle example:
%run /kaggle/working/BSEE_Experiment_B_TECR_Inference.py --platform 10042::1 --year 2024
Only a research-proxy output; not a validated regulatory forecast.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import joblib,torch
from torch import nn

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
  states,_=self.gru(x*torch.sigmoid(self.gate)[None,None,:]); weights=torch.softmax(self.att(states),dim=1)
  pooled=(weights*states).sum(dim=1)
  embedding=torch.cat([emb(c[:,:,j]).flatten(1) for j,emb in enumerate(self.emb)],dim=1)
  return self.head(torch.cat((pooled,embedding),dim=1)).squeeze(-1)

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''): h.update(block)
 return h.hexdigest()

def predict(platform_id,feature_year,artifact_dir='/kaggle/working/BSEE_Experiment_B_Model',
            panel_path='/kaggle/working/BSEE_Experiment_B_Reconstruction/risk_set_all_platform_years.csv.gz'):
 root=Path(artifact_dir); manifest=json.loads((root/'inference_manifest.json').read_text())
 assert manifest['model_version']=='tecr_b_provisional_v1'
 for filename, expected in manifest['artifact_sha256'].items():
  if (root/filename).exists() and sha(root/filename)!=expected:raise RuntimeError(f'Artifact hash mismatch: {filename}')
 for filename,expected in manifest['checkpoint_sha256'].items():
  if sha(root/filename)!=expected:raise RuntimeError(f'Checkpoint hash mismatch: {filename}')
 prep=joblib.load(root/'preprocessing.joblib'); cal=joblib.load(root/'platt_calibrator.joblib')
 panel=pd.read_csv(panel_path,dtype={'platform_id':str},low_memory=False)
 sub=panel.loc[panel.platform_id.eq(str(platform_id)) & panel.year.isin([feature_year-2,feature_year-1,feature_year])].sort_values('year')
 if list(sub.year)!=[feature_year-2,feature_year-1,feature_year]:
  raise ValueError('Incomplete three-calendar-year input sequence')
 if feature_year>2024:raise ValueError('2025+ feature years are unsupported by this reconstructed panel')
 values=sub[prep['numeric']].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).to_numpy(np.float64)
 x=np.clip(prep['scaler'].transform(prep['imputer'].transform(values)),-30,30).astype(np.float32)[None,:,:]
 c=np.stack([sub[name].astype('string').fillna('<MISSING>').astype(str).map(prep['vocab'][name]).fillna(0).to_numpy(np.int64)
             for name in prep['categorical']],axis=1)[None,:,:]
 device='cpu'; raw=0.0
 for seed in manifest['seeds']:
  obj=torch.load(root/f'corrected_tecr_seed_{seed}.pt',map_location='cpu',weights_only=True)
  model=TECR(**obj['arch']).to(device);model.load_state_dict(obj['state_dict']);model.eval()
  with torch.no_grad():
   raw+=float(torch.sigmoid(model(torch.from_numpy(x),torch.from_numpy(c))).item())/len(manifest['seeds'])
 logit=np.log(np.clip(raw,1e-6,1-1e-6)/(1-np.clip(raw,1e-6,1-1e-6)))
 calibrated=float(cal.predict_proba(np.array([[logit]],dtype=float))[0,1])
 threshold=float(manifest['calibrated_threshold'])
 assert (calibrated>=threshold)==(raw>=manifest['raw_threshold_2019'])
 return {'platform_id':str(platform_id),'feature_year':int(feature_year),'target_year':int(feature_year+1),
  'model_version':manifest['model_version'],'raw_probability':raw,
  'calibrated_probability':calibrated,'decision_threshold':threshold,
  'flag':bool(calibrated>=threshold),'outcome':'unverified recorded_INC_proxy',
  'independent_validation':False,'publication_asof_verified':False}

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--platform',required=True)
 parser.add_argument('--year',required=True,type=int);args=parser.parse_args()
 print(json.dumps(predict(args.platform,args.year),indent=2))
