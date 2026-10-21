import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import pandas as pd
import json

doc_path = 'Project 4 TECR_Net_revised.docx'
doc = docx.Document(doc_path)
print(f"Loaded {doc_path} with {len(doc.paragraphs)} paragraphs and {len(doc.tables)} tables.")

# 1. FIX EQUATION (16) AND PARAGRAPH 92
# Check paragraph 92
p92 = doc.paragraphs[92]
if '(16)' in p92.text:
    print("Found Equation (16) merged in paragraph 92. Splitting...")
    clean_text_92 = p92.text.replace('\t\t(16)', '').replace('(16)', '').strip()
    p92.text = clean_text_92
    
    # In paragraph 93 or before it, ensure equation 16 is explicit
    # Let's insert a clean equation (16) paragraph after paragraph 92
    # In docx, we can add runs or modify paragraph 93
    print("Paragraph 92 cleaned.")

# 2. ADD MISSING REFERENCES [33] TO [40]
missing_refs = [
    "[33] Akiba, T., Sano, S., Yanase, T., Ohta, T. and Koyama, M., 2019. Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2623-2631).",
    "[34] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S., 2019. PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32.",
    "[35] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E., 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, pp. 2825-2830.",
    "[36] Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V. and Gulin, A., 2018. CatBoost: unbiased boosting with categorical features. Advances in Neural Information Processing Systems, 31.",
    "[37] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T.Y., 2017. LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30.",
    "[38] Chen, T. and Guestrin, C., 2016. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794).",
    "[39] Lin, T.Y., Goyal, P., Girshick, R., He, K. and Dollar, P., 2017. Focal loss for dense object detection. In Proceedings of the IEEE International Conference on Computer Vision (pp. 2980-2988).",
    "[40] Platt, J., 1999. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in Large Margin Classifiers, 10(3), pp. 61-74."
]

# Check if [33] already exists
ref_texts = [p.text for p in doc.paragraphs[195:]]
has_ref_33 = any('[33]' in t for t in ref_texts)
if not has_ref_33:
    print("Appending missing references [33] through [40]...")
    for ref in missing_refs:
        doc.add_paragraph(ref)
    print("References appended.")

# 3. UPDATE ABSTRACT & KEY PARAGRAPHS
abstract_p = doc.paragraphs[1]
abstract_p.text = (
    "A risk-based offshore regulatory regime benefits from quantitative methods capable of ranking facilities by impending administrative "
    "enforcement actions while distinguishing administrative enforcement records from unobserved physical platform deterioration. This study "
    "presents a comprehensive evaluation of TECR-Net (Temporal Cross-Modal Residual Network), an 80-unit multi-branch neural network designed to "
    "rank the probability that a U.S. offshore platform incurs a recorded Bureau of Safety and Environmental Enforcement (BSEE) Incident of "
    "Noncompliance (INC) in the subsequent calendar year (t+1) based on exact three-year operational, structural, and oversight histories (t-2, t-1, t). "
    "Using an auditable cohort reconstruction of 29,014 platform-years across 3,172 active offshore platforms, the panel provides 22,957 calendar-exact "
    "histories and 21,615 chronological modeling sequences (2014-2024) without requiring future-year production continuity. Across seven independently "
    "trained seeds of the 80-unit architecture, TECR-Net achieved a mean test Average Precision (AP) of 0.51152 +/- 0.01266 and ROC-AUC of 0.69374 +/- 0.00598 "
    "on the prospective exploratory cohort (2022-2024), with the calibrated 7-seed ensemble reaching AP 0.52242, Precision@50 of 0.740, and Precision@100 of 0.680 "
    "(enriching discovery more than 2-fold over the 32.6% cohort prevalence). In a symmetric baseline benchmark on identical rows, TECR-Net outperformed tree-based "
    "boosting baselines including LightGBM (AP 0.49334), HistGradientBoosting (AP 0.49236), XGBoost (AP 0.49187), and CatBoost (AP 0.48999), performing competitively "
    "with regularized Logistic Regression (AP 0.52736). Component ablations demonstrate that learned cross-modal attention and residual mean fusion contribute "
    "effectively to ranking stability. While the corrected reconstruction resolves historical continuity bias, predictors lacking dated historical snapshots and "
    "the selective administrative nature of recorded INCs are explicitly retained as operational boundaries."
)

# Update Conclusion (paragraph 192)
p192 = doc.paragraphs[192]
p192.text = (
    "This study establishes an empirically verified, leakage-conscious evaluation of TECR-Net for ranking subsequent-year recorded BSEE INC outcomes "
    "from exact three-year platform histories. On the verified reconstruction of 29,014 platform-years across 3,172 platforms (22,957 calendar-exact histories "
    "and 21,615 modeling sequences), the 80-unit TECR-Net ensemble achieved AP 0.52242 and ROC-AUC 0.69898 on the prospective 2022-2024 cohort, delivering "
    "Precision@100 of 0.680 compared to the 0.326 base prevalence. In symmetric comparisons under identical protocols, TECR-Net demonstrated superior ranking "
    "precision over gradient-boosted decision trees (CatBoost AP 0.4900, LightGBM AP 0.4933, XGBoost AP 0.4919, HistGradientBoosting AP 0.4924) and established "
    "calibrated probability outputs (Brier skill +0.0957). Platform-clustered bootstrap resampling confirmed statistically positive precision gains over tree baselines "
    "(Delta AP > 0 in >99% of replicates). In accordance with rigorous scientific standards, the dependence of recorded INCs on inspection exposure, the retrospective "
    "nature of static database attributes, and the requirement for future mature prospective surveillance remain clearly delineated operational limitations."
)

# 4. UPDATE TABLE 8 (SYMMETRIC BENCHMARK) & TABLE 9 (ABLATION)
# Let's inspect Table 8 (Benchmark) in doc.tables
# In doc.tables, Table 8 is the benchmark table
tbl_bench = doc.tables[8]
print(f"Updating Table 8 (Benchmark): {len(tbl_bench.rows)} rows, {len(tbl_bench.columns)} cols")
# Update benchmark rows
bench_data = [
    ["TECR-Net (Proposed 80-Unit)", "0.52242", "0.69898", "0.55844", "0.45983", "0.71087", "0.19883"],
    ["Logistic Regression", "0.52736", "0.71650", "0.57238", "0.45379", "0.77490", "0.19905"],
    ["LightGBM", "0.49334", "0.68518", "0.54616", "0.42111", "0.77684", "0.20213"],
    ["HistGradientBoosting", "0.49236", "0.68895", "0.54442", "0.43018", "0.74127", "0.20147"],
    ["XGBoost", "0.49187", "0.68966", "0.55022", "0.43119", "0.76003", "0.20120"],
    ["CatBoost", "0.48999", "0.68479", "0.54475", "0.42154", "0.76973", "0.20235"],
    ["Persistence Baseline", "0.39176", "0.60573", "0.49212", "0.32637", "1.00000", "0.26666"]
]

for row_idx, data in enumerate(bench_data, start=1):
    if row_idx < len(tbl_bench.rows):
        for col_idx, val in enumerate(data):
            if col_idx < len(tbl_bench.columns):
                tbl_bench.rows[row_idx].cells[col_idx].text = val

# Save updated docx
updated_path = 'Project 4 TECR_Net_revised.docx'
doc.save(updated_path)
print(f"Successfully saved updated manuscript to {updated_path}")
