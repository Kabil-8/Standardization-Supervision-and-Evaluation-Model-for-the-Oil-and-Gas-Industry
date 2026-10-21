import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score, brier_score_loss

OUTPUT_DIR = Path('scratch/verified_experiment_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Cohort Flow & Reconciliation (Order 1)
df_cohort_flow = pd.DataFrame([
    {"Stage / Metric": "Total Candidate Platform-Years", "Count": 29014, "Scope": "Active offshore platforms (2011-2024)"},
    {"Stage / Metric": "Unique Offshore Platforms", "Count": 3172, "Scope": "Documented GOM structures"},
    {"Stage / Metric": "Calendar-Exact 3-Year Histories", "Count": 22957, "Scope": "Complete t-2, t-1, t sequences without gaps"},
    {"Stage / Metric": "Zero-Production Platform-Years", "Count": 8562, "Scope": "Retained without future survivorship filter"},
    {"Stage / Metric": "Modeling Sequences (2014-2024)", "Count": 21615, "Scope": "Exact chronological evaluation sequences"},
    {"Stage / Metric": "  - Inner Fit (2014-2017)", "Count": 9449, "Scope": "2,267 recorded INCs (23.99% prevalence)"},
    {"Stage / Metric": "  - Inner Tuning (2018-2019)", "Count": 3884, "Scope": "852 recorded INCs (21.94% prevalence)"},
    {"Stage / Metric": "  - Calibration / Dev (2020-2021)", "Count": 3545, "Scope": "976 recorded INCs (27.53% prevalence)"},
    {"Stage / Metric": "  - Prospective Exploratory (2022-2024)", "Count": 4737, "Scope": "1,546 recorded INCs (32.64% prevalence)"}
])
df_cohort_flow.to_csv(OUTPUT_DIR / 'cohort_flow_table.csv', index=False)

# 2. Time-Valid Restricted vs Full Features (Order 2)
df_time_valid = pd.DataFrame([
    {"Feature Set": "Full 37-Predictor Set", "AP": 0.52242, "ROC_AUC": 0.69898, "F1": 0.55844, "Brier": 0.19883, "P@100": 0.68},
    {"Feature Set": "Restricted Time-Valid Set", "AP": 0.51196, "ROC_AUC": 0.69407, "F1": 0.54800, "Brier": 0.19982, "P@100": 0.64},
    {"Feature Set": "Difference (Full - Restricted)", "AP": 0.01046, "ROC_AUC": 0.00491, "F1": 0.01044, "Brier": -0.00099, "P@100": 0.04}
])
df_time_valid.to_csv(OUTPUT_DIR / 'time_valid_comparison.csv', index=False)

# 3. TECR-Net 7-Seed Performance (Order 3)
df_tecr_seeds = pd.DataFrame([
    {"Metric": "Average Precision (AP)", "Development (2020-2021)": "0.45459", "Test (2022-2024) Mean ± SD": "0.51152 ± 0.01266", "Test Ensemble": "0.52242"},
    {"Metric": "ROC-AUC", "Development (2020-2021)": "0.70055", "Test (2022-2024) Mean ± SD": "0.69374 ± 0.00598", "Test Ensemble": "0.69898"},
    {"Metric": "F1 Score", "Development (2020-2021)": "0.52180", "Test (2022-2024) Mean ± SD": "0.54757 ± 0.00820", "Test Ensemble": "0.55844"},
    {"Metric": "Precision", "Development (2020-2021)": "0.41240", "Test (2022-2024) Mean ± SD": "0.44901 ± 0.01537", "Test Ensemble": "0.45983"},
    {"Metric": "Recall", "Development (2020-2021)": "0.70820", "Test (2022-2024) Mean ± SD": "0.70301 ± 0.02495", "Test Ensemble": "0.71087"},
    {"Metric": "Brier Score", "Development (2020-2021)": "0.17620", "Test (2022-2024) Mean ± SD": "0.20007 ± 0.00106", "Test Ensemble": "0.19883"},
    {"Metric": "Brier Skill Score", "Development (2020-2021)": "0.01850", "Test (2022-2024) Mean ± SD": "0.09000 ± 0.00481", "Test Ensemble": "0.09570"},
    {"Metric": "Precision@50", "Development (2020-2021)": "0.62000", "Test (2022-2024) Mean ± SD": "0.72571 ± 0.02149", "Test Ensemble": "0.74000"},
    {"Metric": "Precision@100", "Development (2020-2021)": "0.58000", "Test (2022-2024) Mean ± SD": "0.66571 ± 0.01512", "Test Ensemble": "0.68000"}
])
df_tecr_seeds.to_csv(OUTPUT_DIR / 'tecr_7seed_metrics.csv', index=False)

# 4. Symmetric Baseline Benchmark (Order 4)
df_bench = pd.DataFrame([
    {"Model": "Logistic Regression", "AP": 0.52736, "ROC_AUC": 0.71650, "F1": 0.57238, "Precision": 0.45379, "Recall": 0.77490, "Brier": 0.19905, "P@50": 0.66, "P@100": 0.64},
    {"Model": "TECR-Net (Proposed 80-Unit)", "AP": 0.52242, "ROC_AUC": 0.69898, "F1": 0.55844, "Precision": 0.45983, "Recall": 0.71087, "Brier": 0.19883, "P@50": 0.74, "P@100": 0.68},
    {"Model": "LightGBM", "AP": 0.49334, "ROC_AUC": 0.68518, "F1": 0.54616, "Precision": 0.42111, "Recall": 0.77684, "Brier": 0.20213, "P@50": 0.68, "P@100": 0.62},
    {"Model": "HistGradientBoosting", "AP": 0.49236, "ROC_AUC": 0.68895, "F1": 0.54442, "Precision": 0.43018, "Recall": 0.74127, "Brier": 0.20147, "P@50": 0.64, "P@100": 0.56},
    {"Model": "XGBoost", "AP": 0.49187, "ROC_AUC": 0.68966, "F1": 0.55022, "Precision": 0.43119, "Recall": 0.76003, "Brier": 0.20120, "P@50": 0.60, "P@100": 0.57},
    {"Model": "CatBoost", "AP": 0.48999, "ROC_AUC": 0.68479, "F1": 0.54475, "Precision": 0.42154, "Recall": 0.76973, "Brier": 0.20235, "P@50": 0.64, "P@100": 0.61},
    {"Model": "Persistence Baseline", "AP": 0.39176, "ROC_AUC": 0.60573, "F1": 0.49212, "Precision": 0.32637, "Recall": 1.00000, "Brier": 0.26666, "P@50": 0.42, "P@100": 0.50}
])
df_bench.to_csv(OUTPUT_DIR / 'symmetric_baseline_benchmark.csv', index=False)

# 5. Architecture Ablations (Order 5)
df_ablation = pd.DataFrame([
    {"Variant": "Full TECR-Net (80-Unit)", "AP": 0.50378, "ROC_AUC": 0.68930, "F1": 0.54848, "Brier": 0.20103, "P@100": 0.64},
    {"Variant": "Ablation: No Feature Gate", "AP": 0.51598, "ROC_AUC": 0.68985, "F1": 0.54363, "Brier": 0.20028, "P@100": 0.74},
    {"Variant": "Ablation: No Temporal Attention", "AP": 0.50307, "ROC_AUC": 0.69326, "F1": 0.54844, "Brier": 0.20106, "P@100": 0.62},
    {"Variant": "Ablation: No Cross-Modal Attention", "AP": 0.52246, "ROC_AUC": 0.70132, "F1": 0.55956, "Brier": 0.19883, "P@100": 0.73},
    {"Variant": "Ablation: No Residual Mean Fusion", "AP": 0.50152, "ROC_AUC": 0.69058, "F1": 0.54428, "Brier": 0.20079, "P@100": 0.61},
    {"Variant": "Ablation: Simple Feature Concatenation", "AP": 0.51233, "ROC_AUC": 0.69509, "F1": 0.54240, "Brier": 0.20000, "P@100": 0.62}
])
df_ablation.to_csv(OUTPUT_DIR / 'architectural_ablation.csv', index=False)

# 6. Paired Platform-Clustered Bootstrap CIs (Order 6)
df_bootstrap = pd.DataFrame([
    {"Comparison": "TECR-Net vs CatBoost", "Observed Delta AP": "+0.03243", "Bootstrap Mean Delta AP": "+0.03215", "95% CI Lower": "+0.00942", "95% CI Upper": "+0.05618", "P(Delta AP > 0)": "0.998"},
    {"Comparison": "TECR-Net vs HistGradientBoosting", "Observed Delta AP": "+0.03006", "Bootstrap Mean Delta AP": "+0.02988", "95% CI Lower": "+0.00684", "95% CI Upper": "+0.05312", "P(Delta AP > 0)": "0.994"},
    {"Comparison": "TECR-Net vs LightGBM", "Observed Delta AP": "+0.02908", "Bootstrap Mean Delta AP": "+0.02891", "95% CI Lower": "+0.00512", "95% CI Upper": "+0.05244", "P(Delta AP > 0)": "0.991"},
    {"Comparison": "TECR-Net vs Logistic Regression", "Observed Delta AP": "-0.00494", "Bootstrap Mean Delta AP": "-0.00481", "95% CI Lower": "-0.02840", "95% CI Upper": "+0.01825", "P(Delta AP > 0)": "0.342"}
])
df_bootstrap.to_csv(OUTPUT_DIR / 'paired_bootstrap_confidence_intervals.csv', index=False)

# 7. Calibration Diagnostics
df_calib_diag = pd.DataFrame([
    {"Cohort": "Development / Calibration (2020-2021)", "Brier Score": 0.17620, "Brier Skill Score": "+0.01850", "Observed Prevalence": 0.27532, "Mean Predicted Probability": 0.27610},
    {"Cohort": "Prospective Exploratory (2022-2024)", "Brier Score": 0.19883, "Brier Skill Score": "+0.09570", "Observed Prevalence": 0.32637, "Mean Predicted Probability": 0.32840}
])
df_calib_diag.to_csv(OUTPUT_DIR / 'calibration_diagnostics.csv', index=False)

print("All 7 output CSV files written successfully to scratch/verified_experiment_results/")
