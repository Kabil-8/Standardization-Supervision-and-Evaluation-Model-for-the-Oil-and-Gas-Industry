"""TECR-Net Experiment A: frozen archived model, 2025 predictors, interim 2026 INC surveillance.

Requires tecrnet_inference.py (provided in companion ZIP), exact seven frozen checkpoints
and their preprocessing.json under a single Kaggle directory. No fitting/threshold tuning.
Outputs predictions BEFORE reading the 2026 INC file. Does NOT compute classification metrics.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys
import numpy as np
import pandas as pd

# ========= CONFIG (change only if Kaggle dataset layout differs) =========
ARCHIVED_PANEL = Path('/kaggle/input/datasets/kabil908/bsee-preprocessed/bsee_platform_year_full.csv')
NEW_RAW_ROOT = Path('/kaggle/input/datasets/kabil908/p4-dataset')
OUT = Path('/kaggle/working/BSEE_Experiment_A_2026_Interim')
THRESHOLD = 0.04908945831718186  # archived ensemble development-selected threshold; not independently validated
# If automatic discovery fails, set the directory containing preprocessing.json and seven tecrnet_seed_*.pt:
FROZEN_ARTIFACTS = None
# ===============================================================

try:
    import tecrnet_inference as T
except ImportError as e:
    raise RuntimeError('Place tecrnet_inference.py beside this script, or add its directory to sys.path') from e

SEEDS = [11,22,33,44,55,66,77]
OUT.mkdir(parents=True,exist_ok=True)

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def locate_one(root,name):
    matches = [p for p in root.rglob(name) if p.is_file()]
    if len(matches)!=1:
        raise RuntimeError(f'Expected exactly one {name} in {root}; found {len(matches)}: {matches[:5]}')
    return matches[0]

def locate_artifacts():
    if FROZEN_ARTIFACTS:
        folders=[Path(FROZEN_ARTIFACTS)]
    else:
        folders=[p.parent for p in Path('/kaggle/input').rglob('preprocessing.json')]
    matches=[d for d in folders if (d/'preprocessing.json').is_file() and all((d/f'tecrnet_seed_{s}.pt').is_file() for s in SEEDS)]
    if len(matches)!=1:
        raise RuntimeError(
            'Expected exactly one full_model artifact folder containing preprocessing.json and all seven checkpoints. '
            f'Found: {[str(x) for x in matches]}. Set FROZEN_ARTIFACTS explicitly.'
        )
    return matches[0]

if not ARCHIVED_PANEL.exists():
    raise FileNotFoundError(f'Missing archived full panel: {ARCHIVED_PANEL}')
if not NEW_RAW_ROOT.exists():
    raise FileNotFoundError(f'Missing new official source dataset: {NEW_RAW_ROOT}')
art = locate_artifacts()
prep=json.loads((art/'preprocessing.json').read_text(encoding='utf-8'))
panel=T.load_panel(ARCHIVED_PANEL)
panel['platform_id']=panel['platform_id'].astype(str).str.strip()
panel['year']=pd.to_numeric(panel['year'],errors='raise').astype(int)
assert not panel.duplicated(['platform_id','year']).any()

# Build three consecutive calendar-year windows; no future-year condition.
key_to_row={(str(pid),int(y)):i for i,(pid,y) in enumerate(zip(panel.platform_id,panel.year))}
index=[]
for pid, y in sorted(key_to_row):
    if y!=2025: continue
    keys=[(pid,2023),(pid,2024),(pid,2025)]
    if all(k in key_to_row for k in keys):
        index.append(dict(platform_id=pid,feature_year=2025,target_year=2026,
                          row_t_minus_2=key_to_row[keys[0]],row_t_minus_1=key_to_row[keys[1]],row_t=key_to_row[keys[2]]))
idx=pd.DataFrame(index)
if idx.empty: raise RuntimeError('No 2025 feature-year candidates found')
print('Three-year-history candidates:',len(idx),'(anticipated 714 on archived full panel)')

# Validate all columns without using target-year labels.
needed=prep['dynamic_columns']+prep['static_columns']+prep['categorical_columns']
missing=[c for c in needed if c not in panel.columns]
if missing: raise RuntimeError(f'Frozen model input columns unavailable: {missing}')
arrays=T.build_arrays(panel,idx,prep)
score=T.score(arrays,prep,art,device='cpu')
if len(score)!=len(idx): raise AssertionError('Prediction count mismatch')
if not np.isfinite(score['ensemble_probability']).all(): raise RuntimeError('Nonfinite scores')
if not score['ensemble_probability'].between(0,1).all():raise RuntimeError('Invalid scores')

pred=pd.concat([idx[['platform_id','feature_year','target_year']].reset_index(drop=True),score.reset_index(drop=True)],axis=1)
pred['archived_development_threshold']=THRESHOLD
pred['flag_at_archived_threshold']=(pred['ensemble_probability']>=THRESHOLD).astype(int)
pred['source_feature_status']='ARCHIVED_PREPROCESSED_2023_TO_2025_NOT_YET_REFRESHED_OR_ASOF_VERIFIED'
pred['outcome_status']='UNLABELED_FULL_YEAR_2026_INCOMPLETE'
predpath=OUT/'2025_candidates_frozen_predictions_NO_LABELS.csv'
if predpath.exists():raise FileExistsError(f'Refusing to replace archived score file: {predpath}')
pred.to_csv(predpath,index=False)
print('Frozen predictions saved first:',predpath)
print('Predicted flags at archived threshold:',int(pred.flag_at_archived_threshold.sum()))

# Only now inspect partial 2026 events. Source ID_NUMBER::STRUCTURE_NUMBER mapping
# is a provisional join. We do not assume every enforcement-code line is an independent event.
incpath=locate_one(NEW_RAW_ROOT,'mv_platstruc_platformincs.txt')
incs=pd.read_csv(incpath,dtype=str,low_memory=False)
required=['ID_NUMBER','STRUCTURE_NUMBER','INC_ISSUE_DATE','INC_NUMBER']
if any(c not in incs for c in required):raise RuntimeError('Unexpected INC source fields')
for col in ('ID_NUMBER','STRUCTURE_NUMBER','INC_NUMBER'):
    incs[col]=incs[col].astype('string').str.strip()
incs['event_date']=pd.to_datetime(incs['INC_ISSUE_DATE'],format='mixed',errors='coerce')
cutoff=pd.Timestamp('2026-10-17 23:59:59')
partial=incs[incs.event_date.between(pd.Timestamp('2026-01-01'),cutoff,inclusive='both')].copy()
partial['platform_id']=partial['ID_NUMBER'].astype(str)+'::'+partial['STRUCTURE_NUMBER'].astype(str)
matched=partial[partial.platform_id.isin(set(idx.platform_id))].copy()

# Event records may represent multiple enforcement codes for the same inspection.
counts=matched.groupby('platform_id').agg(
    matched_2026_inc_code_lines=('INC_NUMBER','size'),
    first_recorded_2026_inc=('event_date','min'),
    last_recorded_2026_inc=('event_date','max')
).reset_index()
joined=pred[['platform_id','feature_year','target_year','ensemble_probability','flag_at_archived_threshold']].merge(
    counts,on='platform_id',how='left',validate='one_to_one')
joined['matched_2026_inc_code_lines']=joined['matched_2026_inc_code_lines'].fillna(0).astype(int)
joined['has_observed_inc_by_cutoff']=(joined.matched_2026_inc_code_lines>0).astype(int)
joined['no_matched_inc_is_confirmed_negative']=False
joined['year_complete']=False
joined['join_verification']='PROVISIONAL_ID_NUMBER_STRUCTURE_NUMBER_KEY'
joined['last_included_event_date']='2026-10-17'
joinedpath=OUT/'2026_partial_inc_event_matching_EXPLORATORY.csv'
if joinedpath.exists():raise FileExistsError(f'Refusing to overwrite: {joinedpath}')
joined.to_csv(joinedpath,index=False)

receipt={
 'experiment':'A_frozen_TECRNet_partial_2026_descriptive_only',
 'created_utc':datetime.now(timezone.utc).isoformat(),
 'dataset':str(ARCHIVED_PANEL), 'archived_panel_sha256':sha256(ARCHIVED_PANEL),
 'new_inc_source':str(incpath),'new_inc_source_sha256':sha256(incpath),
 'frozen_artifacts':str(art),
 'preprocessing_sha256':sha256(art/'preprocessing.json'),
 'checkpoint_sha256':{str(s):sha256(art/f'tecrnet_seed_{s}.pt') for s in SEEDS},
 'threshold':THRESHOLD,'threshold_selection':'archived development F1; not independent',
 'prediction_csv_sha256':sha256(predpath),
 'matched_partial_inc_csv_sha256':sha256(joinedpath),
 'candidate_count':len(idx),
 'flagged_count':int(pred.flag_at_archived_threshold.sum()),
 'candidates_with_at_least_one_matched_inc_line_by_cutoff':int(joined.has_observed_inc_by_cutoff.sum()),
 'matched_inc_code_lines':len(matched),'all_2026_inc_code_lines_through_cutoff':len(partial),
 'run_inference_before_reading_outcomes':True,
 'excluded_target_year_production_requirement':True,
 'model_refit':False,'calibration_refit':False,'threshold_retuned':False,
 'important_limitations':[
  '2026 outcomes are partial and previously inspected; NOT untouched mature validation',
  'Historical 2023-2025 feature values are archived preprocessing outputs; not recomputed from new source snapshot',
  'Historical availability and 2025 forecast issuance cutoff NOT verified',
  'Platform-to-INC join is provisional and must be independently verified',
  'No-record platforms are NOT verified negative cases',
  'No AP, ROC-AUC, Brier, F1, or other evaluation metric is justified here',
  'Original Platt calibration may have been tuned on an already reported validation set']
}
receipt_path=OUT/'experiment_A_receipt.json'
if receipt_path.exists():raise FileExistsError(f'Refusing overwrite: {receipt_path}')
receipt_path.write_text(json.dumps(receipt,indent=2,default=str),encoding='utf-8')
print('Partial 2026 source code lines through cutoff:',len(partial))
print('Matched code lines for 2025 candidates:',len(matched))
print('Candidate platforms with observed INC by cutoff:',int(joined.has_observed_inc_by_cutoff.sum()))
print('CSV and audit receipt saved at:',OUT)
print('STATUS: exploratory surveillance ONLY; NO independent evaluation, NO negative labels, NO metrics')
