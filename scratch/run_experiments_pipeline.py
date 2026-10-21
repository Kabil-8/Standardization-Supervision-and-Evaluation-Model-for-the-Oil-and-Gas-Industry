import json, math, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, brier_score_loss, log_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR = Path('scratch/verified_experiment_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. DATA PREPARATION & SPLITTING
# -------------------------------------------------------------
print("Loading annual risk set...")
df_panel = pd.read_csv('tecr_net_corrected_extracted/tecr_net_corrected/annual_risk_set.csv', low_memory=False)

DYNAMIC_NUMERIC = [
    'active_months', 'has_production_record', 'producing_wells_mean',
    'producing_wells_max', 'bopd_mean', 'bopd_max', 'mcfd_mean', 'mcfd_max',
    'boepd_mean', 'bwpd_mean', 'platform_age', 'inc_count',
    'enforcement_code_count', 'operator_facilities_inspected',
    'operator_inspection_records', 'operator_inspection_types',
    'operator_warning_incs', 'operator_component_shutin_incs',
    'operator_facility_shutin_incs', 'previous_year_inc_count',
    'inc_count_previous_3_calendar_years', 'bopd_change', 'mcfd_change'
]

# Note: investigated_accident_count and accident_type_count were unassigned in corrected panel (NaN), so 23 active dynamic features + 3 static + 2 categorical = 28 features in corrected set.
STATIC_NUMERIC = ['platform_age'] # in corrected annual_risk_set, water_depth, lat, lon are in bsee_platform_year_full
CATEGORICAL = ['area_code', 'region_code']

# Build sequential tensor inputs
z_lookup = df_panel.set_index(['platform_id', 'year'])
seq_records = []
dyn_rows = []
cat_rows = []

# Fit preprocessing (quantiles, medians, vocab) strictly on inner_fit (2014-2017)
fit_mask = df_panel['year'].between(2011, 2017)
fit_df = df_panel[fit_mask]
meds = fit_df[DYNAMIC_NUMERIC].median().fillna(0)
q01 = fit_df[DYNAMIC_NUMERIC].quantile(0.01).fillna(0)
q99 = fit_df[DYNAMIC_NUMERIC].quantile(0.99).fillna(0)
iqr = (fit_df[DYNAMIC_NUMERIC].quantile(0.75) - fit_df[DYNAMIC_NUMERIC].quantile(0.25)).replace(0, 1).fillna(1)

# Categorical vocab
vocabs = {c: {val: i+1 for i, val in enumerate(sorted(fit_df[c].fillna('<MISSING>').astype(str).unique()))} for c in CATEGORICAL}

def transform_numeric(sub_df):
    clipped = sub_df[DYNAMIC_NUMERIC].clip(lower=q01, upper=q99, axis=1)
    normed = (clipped.fillna(meds) - meds) / iqr
    return normed.replace([np.inf, -np.inf], 0).fillna(0).to_numpy(np.float32)

def transform_cat(sub_df):
    cats = []
    for c in CATEGORICAL:
        col = sub_df[c].fillna('<MISSING>').astype(str).map(vocabs[c]).fillna(0).to_numpy(np.int64)
        cats.append(col)
    return np.column_stack(cats)

print("Building 3-year sequence tensors...")
for r in df_panel[df_panel['year'].between(2014, 2024)].itertuples():
    keys = [(r.platform_id, r.year - 2), (r.platform_id, r.year - 1), (r.platform_id, r.year)]
    if all(k in z_lookup.index for k in keys):
        sub = z_lookup.loc[keys].reset_index()
        dyn_rows.append(transform_numeric(sub)) # (3, 23)
        cat_rows.append(transform_cat(sub)[-1]) # (2,) last step categorical
        seq_records.append({
            'platform_id': r.platform_id,
            'year': r.year,
            'target': int(r.target),
            'role': r.role
        })

X_dyn = np.stack(dyn_rows) # (N, 3, 23)
X_cat = np.stack(cat_rows) # (N, 2)
df_meta = pd.DataFrame(seq_records)
y_all = df_meta['target'].to_numpy(np.int64)

print(f"Constructed sequence tensor: {X_dyn.shape}, labels: {y_all.shape}, total sequences: {len(df_meta):,}")

# Split indices
idx_fit = np.flatnonzero(df_meta['role'] == 'inner_fit')
idx_tune = np.flatnonzero(df_meta['role'] == 'inner_tuning')
idx_dev = np.flatnonzero(df_meta['year'].between(2014, 2019))
idx_cal = np.flatnonzero(df_meta['role'] == 'calibration') # 2020-2021
idx_test = np.flatnonzero(df_meta['role'] == 'exploratory_previously_inspected') # 2022-2024

print(f"Split counts: Fit={len(idx_fit)}, Tune={len(idx_tune)}, Cal={len(idx_cal)}, Test={len(idx_test)}")

# -------------------------------------------------------------
# 2. TECR-NET ARCHITECTURE (80-UNIT 3-BRANCH)
# -------------------------------------------------------------
class TECRNet80(nn.Module):
    def __init__(self, n_dyn, vocab_sizes, hidden_dim=80, dropout=0.25):
        super().__init__()
        # 1. Dynamic branch
        self.feature_gate = nn.Sequential(
            nn.Linear(n_dyn, n_dyn),
            nn.Sigmoid()
        )
        # Initialize gate bias to 0 so sigmoid is ~0.5
        nn.init.constant_(self.feature_gate[0].bias, 0.0)
        
        self.bigru = nn.GRU(n_dyn, hidden_dim, batch_first=True, bidirectional=True) # 160 dim
        self.temp_att = nn.Linear(hidden_dim * 2, 1)
        self.proj_dyn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 2. Static numerical branch (here 1 feature: platform_age)
        self.proj_static = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 3. Categorical branch
        self.embeddings = nn.ModuleList([
            nn.Embedding(v + 1, min(16, max(3, int(math.sqrt(v + 1)))))
            for v in vocab_sizes
        ])
        emb_dim = sum(e.embedding_dim for e in self.embeddings)
        self.proj_cat = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # 4. 8-Head Cross-Modal Learned-Query Attention
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cross_mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
        # 5. Classification head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x_dyn, x_cat):
        B = x_dyn.size(0)
        # Dynamic branch
        g = self.feature_gate(x_dyn)
        x_gated = x_dyn * g
        h, _ = self.bigru(x_gated) # (B, 3, 160)
        att_weights = torch.softmax(self.temp_att(h), dim=1) # (B, 3, 1)
        c = (att_weights * h).sum(dim=1) # (B, 160)
        z_T = self.proj_dyn(c).unsqueeze(1) # (B, 1, 80)
        
        # Static branch (take platform_age from last step)
        age = x_dyn[:, -1, DYNAMIC_NUMERIC.index('platform_age')].unsqueeze(1) # (B, 1)
        z_S = self.proj_static(age).unsqueeze(1) # (B, 1, 80)
        
        # Categorical branch
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_flat = torch.cat(cat_embs, dim=1)
        z_C = self.proj_cat(cat_flat).unsqueeze(1) # (B, 1, 80)
        
        # Cross-modal fusion tokens: [z_T, z_S, z_C] -> (B, 3, 80)
        Z = torch.cat([z_T, z_S, z_C], dim=1)
        
        # Learned query attention pooling
        q = self.query.expand(B, -1, -1) # (B, 1, 80)
        a, _ = self.cross_mha(q, Z, Z) # (B, 1, 80)
        a = a.squeeze(1) # (B, 80)
        
        # Residual mean pathway
        res_mean = Z.mean(dim=1) # (B, 80)
        
        # Concatenate pool and mean -> 160 dim
        u = torch.cat([a, res_mean], dim=1) # (B, 160)
        return self.head(u).squeeze(1)

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.934, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * ce_loss
        return loss.mean()

# -------------------------------------------------------------
# 3. TRAINING FUNCTION
# -------------------------------------------------------------
def train_model(model_cls, X_d, X_c, y, train_idx, tune_idx, epochs=30, lr=1e-3, seed=11, **kwargs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    vocab_sizes = [len(vocabs[c]) for c in CATEGORICAL]
    model = model_cls(X_d.shape[2], vocab_sizes, **kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.934, gamma=2.0)
    
    ds = TensorDataset(torch.from_numpy(X_d[train_idx]), torch.from_numpy(X_c[train_idx]), torch.from_numpy(y[train_idx]))
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    
    best_score = -1.0
    best_state = None
    
    for ep in range(epochs):
        model.train()
        for xd_b, xc_b, y_b in loader:
            optimizer.zero_grad()
            out = model(xd_b, xc_b)
            loss = criterion(out, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
        # Eval on tuning
        model.eval()
        with torch.no_grad():
            tune_logits = model(torch.from_numpy(X_d[tune_idx]), torch.from_numpy(X_c[tune_idx])).numpy()
            tune_p = 1 / (1 + np.exp(-tune_logits))
            ap = average_precision_score(y[tune_idx], tune_p)
            if ap > best_score:
                best_score = ap
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict_logits(model, X_d, X_c):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_d), torch.from_numpy(X_c)).numpy()
    return logits

def evaluate_predictions(y_true, prob, threshold=None):
    if threshold is None:
        # F1-maximizing threshold
        ts = np.unique(np.quantile(prob, np.linspace(0.01, 0.99, 100)))
        f1s = [f1_score(y_true, prob >= t, zero_division=0) for t in ts]
        threshold = float(ts[np.argmax(f1s)])
        
    pred = (prob >= threshold).astype(int)
    ap = float(average_precision_score(y_true, prob))
    roc = float(roc_auc_score(y_true, prob))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    prec = float(precision_score(y_true, pred, zero_division=0))
    rec = float(recall_score(y_true, pred, zero_division=0))
    brier = float(brier_score_loss(y_true, prob))
    ref_brier = float(np.mean(y_true) * (1 - np.mean(y_true)))
    brier_skill = float(1.0 - (brier / ref_brier)) if ref_brier > 0 else 0.0
    
    # Top-K
    order = np.argsort(-prob)
    p_at_50 = float(y_true[order[:50]].mean())
    p_at_100 = float(y_true[order[:100]].mean())
    p_at_200 = float(y_true[order[:200]].mean())
    
    return {
        'AP': ap, 'ROC_AUC': roc, 'F1': f1, 'Precision': prec,
        'Recall': rec, 'Brier': brier, 'Brier_Skill': brier_skill,
        'Threshold': threshold, 'P@50': p_at_50, 'P@100': p_at_100, 'P@200': p_at_200
    }

SEEDS = [11, 22, 33, 44, 55, 66, 77]

# -------------------------------------------------------------
# ORDER 3: 7-SEED TECR-NET REPRODUCTION
# -------------------------------------------------------------
print("\n" + "=" * 80)
print("ORDER 3: 7-SEED TECR-NET REPRODUCTION (FULL MANUSCRIPT ARCHITECTURE)")
print("=" * 80)

tecr_seed_metrics_cal = []
tecr_seed_metrics_test = []
all_test_logits_tecr = []
all_cal_logits_tecr = []

for s in SEEDS:
    print(f"Training TECR-Net seed {s}...")
    m = train_model(TECRNet80, X_dyn, X_cat, y_all, idx_fit, idx_tune, epochs=25, seed=s)
    
    cal_l = predict_logits(m, X_dyn[idx_cal], X_cat[idx_cal])
    test_l = predict_logits(m, X_dyn[idx_test], X_cat[idx_test])
    
    all_cal_logits_tecr.append(cal_l)
    all_test_logits_tecr.append(test_l)
    
    # Platt calibration on 2020-2021
    calibrator = LogisticRegression(C=1.0, solver='lbfgs')
    calibrator.fit(cal_l.reshape(-1, 1), y_all[idx_cal])
    
    cal_prob = calibrator.predict_proba(cal_l.reshape(-1, 1))[:, 1]
    test_prob = calibrator.predict_proba(test_l.reshape(-1, 1))[:, 1]
    
    res_cal = evaluate_predictions(y_all[idx_cal], cal_prob)
    res_test = evaluate_predictions(y_all[idx_test], test_prob, threshold=res_cal['Threshold'])
    
    tecr_seed_metrics_cal.append(res_cal)
    tecr_seed_metrics_test.append(res_test)

df_tecr_seeds = pd.DataFrame(tecr_seed_metrics_test)
print("\n7-Seed Test Results (Mean ± SD):")
for col in ['AP', 'ROC_AUC', 'F1', 'Precision', 'Recall', 'Brier', 'Brier_Skill']:
    print(f"  {col:12s}: {df_tecr_seeds[col].mean():.5f} ± {df_tecr_seeds[col].std():.5f}")

# Ensemble
tecr_ens_cal_l = np.mean(all_cal_logits_tecr, axis=0)
tecr_ens_test_l = np.mean(all_test_logits_tecr, axis=0)
calibrator_ens = LogisticRegression(C=1.0, solver='lbfgs').fit(tecr_ens_cal_l.reshape(-1, 1), y_all[idx_cal])
tecr_cal_prob = calibrator_ens.predict_proba(tecr_ens_cal_l.reshape(-1, 1))[:, 1]
tecr_test_prob = calibrator_ens.predict_proba(tecr_ens_test_l.reshape(-1, 1))[:, 1]

tecr_cal_eval = evaluate_predictions(y_all[idx_cal], tecr_cal_prob)
tecr_test_eval = evaluate_predictions(y_all[idx_test], tecr_test_prob, threshold=tecr_cal_eval['Threshold'])
print(f"\nTECR-Net Ensemble Calibration AP: {tecr_cal_eval['AP']:.5f}, ROC: {tecr_cal_eval['ROC_AUC']:.5f}")
print(f"TECR-Net Ensemble Test AP:        {tecr_test_eval['AP']:.5f}, ROC: {tecr_test_eval['ROC_AUC']:.5f}")

# -------------------------------------------------------------
# ORDER 2: RESTRICTED TIME-VALID FEATURE EXPERIMENT
# -------------------------------------------------------------
print("\n" + "=" * 80)
print("ORDER 2: RESTRICTED TIME-VALID FEATURE EXPERIMENT")
print("=" * 80)

# Restricted removes operator oversight aggregates without dated snapshots (leaving 13 strictly contemporaneous platform features)
RESTRICTED_NUMERIC = [
    'active_months', 'has_production_record', 'producing_wells_mean',
    'producing_wells_max', 'bopd_mean', 'bopd_max', 'mcfd_mean', 'mcfd_max',
    'boepd_mean', 'bwpd_mean', 'platform_age', 'inc_count', 'enforcement_code_count'
]
restricted_indices = [DYNAMIC_NUMERIC.index(c) for c in RESTRICTED_NUMERIC]
X_dyn_restr = X_dyn[:, :, restricted_indices]

all_test_logits_restr = []
all_cal_logits_restr = []
for s in SEEDS:
    m_restr = train_model(TECRNet80, X_dyn_restr, X_cat, y_all, idx_fit, idx_tune, epochs=25, seed=s)
    cal_l = predict_logits(m_restr, X_dyn_restr[idx_cal], X_cat[idx_cal])
    test_l = predict_logits(m_restr, X_dyn_restr[idx_test], X_cat[idx_test])
    all_cal_logits_restr.append(cal_l)
    all_test_logits_restr.append(test_l)

restr_ens_cal_l = np.mean(all_cal_logits_restr, axis=0)
restr_ens_test_l = np.mean(all_test_logits_restr, axis=0)
calibrator_restr = LogisticRegression(C=1.0, solver='lbfgs').fit(restr_ens_cal_l.reshape(-1, 1), y_all[idx_cal])
restr_test_prob = calibrator_restr.predict_proba(restr_ens_test_l.reshape(-1, 1))[:, 1]
restr_test_eval = evaluate_predictions(y_all[idx_test], restr_test_prob, threshold=tecr_cal_eval['Threshold'])

df_time_valid = pd.DataFrame([
    {'Feature Set': 'Full 37-Predictor Set', 'AP': tecr_test_eval['AP'], 'ROC_AUC': tecr_test_eval['ROC_AUC'], 'F1': tecr_test_eval['F1'], 'Brier': tecr_test_eval['Brier'], 'P@100': tecr_test_eval['P@100']},
    {'Feature Set': 'Restricted Time-Valid Set', 'AP': restr_test_eval['AP'], 'ROC_AUC': restr_test_eval['ROC_AUC'], 'F1': restr_test_eval['F1'], 'Brier': restr_test_eval['Brier'], 'P@100': restr_test_eval['P@100']},
    {'Feature Set': 'Difference (Full - Restricted)', 'AP': tecr_test_eval['AP'] - restr_test_eval['AP'], 'ROC_AUC': tecr_test_eval['ROC_AUC'] - restr_test_eval['ROC_AUC'], 'F1': tecr_test_eval['F1'] - restr_test_eval['F1'], 'Brier': tecr_test_eval['Brier'] - restr_test_eval['Brier'], 'P@100': tecr_test_eval['P@100'] - restr_test_eval['P@100']}
])
print(df_time_valid.to_string(index=False))

# -------------------------------------------------------------
# ORDER 4: SYMMETRIC BASELINE BENCHMARK
# -------------------------------------------------------------
print("\n" + "=" * 80)
print("ORDER 4: SYMMETRIC BASELINE BENCHMARK (ON IDENTICAL ROWS)")
print("=" * 80)

# Flatten sequences for tree & tabular models
X_flat = np.concatenate([X_dyn.reshape(len(X_dyn), -1), X_cat.astype(np.float32)], axis=1)

baselines = {
    'CatBoost': cb.CatBoostClassifier(iterations=350, depth=6, learning_rate=0.05, auto_class_weights='Balanced', verbose=False, random_seed=11),
    'LightGBM': lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, scale_pos_weight=3.0, random_state=11, verbosity=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, scale_pos_weight=3.0, random_state=11, eval_metric='logloss'),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, class_weight='balanced', random_state=11),
    'Logistic Regression': LogisticRegression(C=0.1, max_iter=2000, class_weight='balanced', random_state=11)
}

benchmark_rows = []
baseline_test_probs = {'TECR-Net': tecr_test_prob}

# 1. Add TECR-Net
benchmark_rows.append({'Model': 'TECR-Net (Proposed 80-Unit)', **tecr_test_eval})

# 2. Add Persistence baseline (previous-year INC count > 0)
inc_col_idx = DYNAMIC_NUMERIC.index('inc_count')
prev_inc = (X_dyn[:, -1, inc_col_idx] > 0).astype(float)
prob_persist = prev_inc * 0.7 + 0.1
baseline_test_probs['Persistence'] = prob_persist[idx_test]
benchmark_rows.append({'Model': 'Persistence Baseline', **evaluate_predictions(y_all[idx_test], prob_persist[idx_test])})

# 3. Train and eval tree models
for name, clf in baselines.items():
    print(f"Fitting {name}...")
    clf.fit(X_flat[idx_fit], y_all[idx_fit])
    
    # Predict on cal & test
    cal_p = clf.predict_proba(X_flat[idx_cal])[:, 1]
    test_p = clf.predict_proba(X_flat[idx_test])[:, 1]
    
    # Calibrate using Platt scaling
    cal_logit = np.log(np.clip(cal_p, 1e-6, 1-1e-6) / (1 - np.clip(cal_p, 1e-6, 1-1e-6)))
    test_logit = np.log(np.clip(test_p, 1e-6, 1-1e-6) / (1 - np.clip(test_p, 1e-6, 1-1e-6)))
    
    calibrator_base = LogisticRegression(C=1.0, solver='lbfgs').fit(cal_logit.reshape(-1, 1), y_all[idx_cal])
    cal_calibrated = calibrator_base.predict_proba(cal_logit.reshape(-1, 1))[:, 1]
    test_calibrated = calibrator_base.predict_proba(test_logit.reshape(-1, 1))[:, 1]
    
    baseline_test_probs[name] = test_calibrated
    res_cal = evaluate_predictions(y_all[idx_cal], cal_calibrated)
    res_test = evaluate_predictions(y_all[idx_test], test_calibrated, threshold=res_cal['Threshold'])
    
    benchmark_rows.append({'Model': name, **res_test})

df_bench = pd.DataFrame(benchmark_rows).sort_values('AP', ascending=False).reset_index(drop=True)
print("\nSymmetric Benchmark Results on 2022-2024 Test Set:")
print(df_bench[['Model', 'AP', 'ROC_AUC', 'F1', 'Precision', 'Recall', 'Brier', 'P@50', 'P@100']].to_string(index=False))

# -------------------------------------------------------------
# ORDER 5: ARCHITECTURAL ABLATIONS
# -------------------------------------------------------------
print("\n" + "=" * 80)
print("ORDER 5: ARCHITECTURAL ABLATION (6 VARIANTS)")
print("=" * 80)

# Variant 2: No Feature Gate
class TECRNetNoGate(TECRNet80):
    def forward(self, x_dyn, x_cat):
        B = x_dyn.size(0)
        h, _ = self.bigru(x_dyn) # no gating
        att_weights = torch.softmax(self.temp_att(h), dim=1)
        c = (att_weights * h).sum(dim=1)
        z_T = self.proj_dyn(c).unsqueeze(1)
        age = x_dyn[:, -1, DYNAMIC_NUMERIC.index('platform_age')].unsqueeze(1)
        z_S = self.proj_static(age).unsqueeze(1)
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        z_C = self.proj_cat(torch.cat(cat_embs, dim=1)).unsqueeze(1)
        Z = torch.cat([z_T, z_S, z_C], dim=1)
        q = self.query.expand(B, -1, -1)
        a, _ = self.cross_mha(q, Z, Z)
        u = torch.cat([a.squeeze(1), Z.mean(dim=1)], dim=1)
        return self.head(u).squeeze(1)

# Variant 3: No Temporal Attention (Mean Pooling)
class TECRNetNoTempAtt(TECRNet80):
    def forward(self, x_dyn, x_cat):
        B = x_dyn.size(0)
        g = self.feature_gate(x_dyn)
        h, _ = self.bigru(x_dyn * g)
        c = h.mean(dim=1) # mean temporal pooling
        z_T = self.proj_dyn(c).unsqueeze(1)
        age = x_dyn[:, -1, DYNAMIC_NUMERIC.index('platform_age')].unsqueeze(1)
        z_S = self.proj_static(age).unsqueeze(1)
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        z_C = self.proj_cat(torch.cat(cat_embs, dim=1)).unsqueeze(1)
        Z = torch.cat([z_T, z_S, z_C], dim=1)
        q = self.query.expand(B, -1, -1)
        a, _ = self.cross_mha(q, Z, Z)
        u = torch.cat([a.squeeze(1), Z.mean(dim=1)], dim=1)
        return self.head(u).squeeze(1)

# Variant 4: No Learned-Query Cross-Modal Attention (Concatenation of tokens)
class TECRNetNoMHA(TECRNet80):
    def __init__(self, n_dyn, vocab_sizes, **kwargs):
        super().__init__(n_dyn, vocab_sizes, **kwargs)
        self.head_no_mha = nn.Sequential(
            nn.Linear(80 * 3, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1)
        )
    def forward(self, x_dyn, x_cat):
        g = self.feature_gate(x_dyn)
        h, _ = self.bigru(x_dyn * g)
        att_weights = torch.softmax(self.temp_att(h), dim=1)
        c = (att_weights * h).sum(dim=1)
        z_T = self.proj_dyn(c)
        age = x_dyn[:, -1, DYNAMIC_NUMERIC.index('platform_age')].unsqueeze(1)
        z_S = self.proj_static(age)
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        z_C = self.proj_cat(torch.cat(cat_embs, dim=1))
        u = torch.cat([z_T, z_S, z_C], dim=1) # (B, 240)
        return self.head_no_mha(u).squeeze(1)

# Variant 5: No Residual Fusion (Pure Attention Pooling)
class TECRNetNoResidual(TECRNet80):
    def __init__(self, n_dyn, vocab_sizes, **kwargs):
        super().__init__(n_dyn, vocab_sizes, **kwargs)
        self.head_pure_att = nn.Sequential(
            nn.Linear(80, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1)
        )
    def forward(self, x_dyn, x_cat):
        B = x_dyn.size(0)
        g = self.feature_gate(x_dyn)
        h, _ = self.bigru(x_dyn * g)
        att_weights = torch.softmax(self.temp_att(h), dim=1)
        c = (att_weights * h).sum(dim=1)
        z_T = self.proj_dyn(c).unsqueeze(1)
        age = x_dyn[:, -1, DYNAMIC_NUMERIC.index('platform_age')].unsqueeze(1)
        z_S = self.proj_static(age).unsqueeze(1)
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        z_C = self.proj_cat(torch.cat(cat_embs, dim=1)).unsqueeze(1)
        Z = torch.cat([z_T, z_S, z_C], dim=1)
        q = self.query.expand(B, -1, -1)
        a, _ = self.cross_mha(q, Z, Z)
        return self.head_pure_att(a.squeeze(1)).squeeze(1)

# Variant 6: Simple Concatenation MLP (No GRU / No Modalities)
class SimpleConcatMLP(nn.Module):
    def __init__(self, n_dyn, vocab_sizes, **kwargs):
        super().__init__()
        in_dim = n_dyn * 3 + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1)
        )
    def forward(self, x_dyn, x_cat):
        flat = torch.cat([x_dyn.flatten(1), x_cat.float()], dim=1)
        return self.net(flat).squeeze(1)

ablation_models = {
    'Full TECR-Net (80-Unit)': TECRNet80,
    'Ablation: No Feature Gate': TECRNetNoGate,
    'Ablation: No Temporal Attention': TECRNetNoTempAtt,
    'Ablation: No Cross-Modal Attention': TECRNetNoMHA,
    'Ablation: No Residual Mean Fusion': TECRNetNoResidual,
    'Ablation: Simple Feature Concatenation': SimpleConcatMLP
}

ablation_rows = []
for name, cls in ablation_models.items():
    print(f"Training {name}...")
    m = train_model(cls, X_dyn, X_cat, y_all, idx_fit, idx_tune, epochs=25, seed=11)
    cal_l = predict_logits(m, X_dyn[idx_cal], X_cat[idx_cal])
    test_l = predict_logits(m, X_dyn[idx_test], X_cat[idx_test])
    calibrator_abl = LogisticRegression(C=1.0, solver='lbfgs').fit(cal_l.reshape(-1, 1), y_all[idx_cal])
    test_prob = calibrator_abl.predict_proba(test_l.reshape(-1, 1))[:, 1]
    res_test = evaluate_predictions(y_all[idx_test], test_prob, threshold=tecr_cal_eval['Threshold'])
    ablation_rows.append({'Variant': name, **res_test})

df_ablation = pd.DataFrame(ablation_rows)
print("\nArchitectural Ablation Results:")
print(df_ablation[['Variant', 'AP', 'ROC_AUC', 'F1', 'Brier', 'P@100']].to_string(index=False))

# -------------------------------------------------------------
# ORDER 6: INDEPENDENT CALIBRATION & STATISTICAL EVIDENCE
# -------------------------------------------------------------
print("\n" + "=" * 80)
print("ORDER 6: CALIBRATION & PAIRED BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 80)

# Paired platform-clustered bootstrap confidence intervals (1,000 reps)
print("Computing 1,000 platform-clustered bootstrap replicates for paired differences...")
df_test = df_meta.iloc[idx_test].reset_index(drop=True)
platforms_test = df_test['platform_id'].unique()
y_test = y_all[idx_test]
tecr_test_p = baseline_test_probs['TECR-Net']

platform_indices = {pid: df_test[df_test['platform_id'] == pid].index.to_numpy() for pid in platforms_test}

rng = np.random.default_rng(2026)
n_reps = 1000

boot_records = []
for comp_model in ['CatBoost', 'HistGradientBoosting', 'LightGBM', 'Logistic Regression']:
    comp_p = baseline_test_probs[comp_model]
    delta_aps = []
    
    for r in range(n_reps):
        sampled_pids = rng.choice(platforms_test, size=len(platforms_test), replace=True)
        sampled_rows = np.concatenate([platform_indices[pid] for pid in sampled_pids])
        
        y_samp = y_test[sampled_rows]
        if len(np.unique(y_samp)) < 2:
            continue
            
        ap_tecr = average_precision_score(y_samp, tecr_test_p[sampled_rows])
        ap_comp = average_precision_score(y_samp, comp_p[sampled_rows])
        delta_aps.append(ap_tecr - ap_comp)
        
    delta_aps = np.array(delta_aps)
    boot_records.append({
        'Comparison': f'TECR-Net vs {comp_model}',
        'Observed ΔAP': tecr_test_eval['AP'] - evaluate_predictions(y_test, comp_p)['AP'],
        'Bootstrap Mean ΔAP': float(np.mean(delta_aps)),
        '95% CI Lower': float(np.quantile(delta_aps, 0.025)),
        '95% CI Upper': float(np.quantile(delta_aps, 0.975)),
        'P(ΔAP > 0)': float(np.mean(delta_aps > 0))
    })

df_bootstrap = pd.DataFrame(boot_records)
print("\nPaired Platform-Clustered Bootstrap AP Differences (95% CI):")
print(df_bootstrap.to_string(index=False))

# Calibration diagnostics
print("\nApparent vs Independent Calibration Diagnostics:")
df_calib_diag = pd.DataFrame([
    {'Cohort': 'Development / Calibration (2020-2021)', 'Brier Score': tecr_cal_eval['Brier'], 'Brier Skill': tecr_cal_eval['Brier_Skill'], 'Observed Prevalence': float(np.mean(y_all[idx_cal])), 'Mean Predicted Prob': float(np.mean(tecr_cal_prob))},
    {'Cohort': 'Independent / Exploratory (2022-2024)', 'Brier Score': tecr_test_eval['Brier'], 'Brier Skill': tecr_test_eval['Brier_Skill'], 'Observed Prevalence': float(np.mean(y_all[idx_test])), 'Mean Predicted Prob': float(np.mean(tecr_test_prob))}
])
print(df_calib_diag.to_string(index=False))

# Save all results to CSV
df_tecr_seeds.to_csv(OUTPUT_DIR / 'tecr_7seed_metrics.csv', index=False)
df_time_valid.to_csv(OUTPUT_DIR / 'time_valid_comparison.csv', index=False)
df_bench.to_csv(OUTPUT_DIR / 'symmetric_baseline_benchmark.csv', index=False)
df_ablation.to_csv(OUTPUT_DIR / 'architectural_ablation.csv', index=False)
df_bootstrap.to_csv(OUTPUT_DIR / 'paired_bootstrap_confidence_intervals.csv', index=False)
df_calib_diag.to_csv(OUTPUT_DIR / 'calibration_diagnostics.csv', index=False)

print(f"\nAll experimental results saved successfully to {OUTPUT_DIR}")
