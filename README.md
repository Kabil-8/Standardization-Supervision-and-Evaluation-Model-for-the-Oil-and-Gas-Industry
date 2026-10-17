# Standardization, Supervision, and Evaluation Model for the Oil and Gas Industry

> **TECR-Net: Temporal Cross-Modal Residual Network for Risk-Based Offshore Inspection Prioritization**  
> *Retrospective Rare-Event Modeling on Bureau of Safety and Environmental Enforcement (BSEE) Offshore Infrastructure Data*

---

## 📌 Executive Summary & Project Overview

Offshore oil and gas regulatory supervision traditionally depends on periodic calendar-based inspections and static facility checklists. However, modern offshore operations involve highly heterogeneous equipment, fluctuating production levels, aging infrastructure, and distributed operator histories.

This repository implements **TECR-Net (Temporal Cross-Modal Network)**, a deep learning and risk-prioritization framework developed to predict whether a specific offshore platform will incur a documented **Incident of Non-Compliance (INC)** in the subsequent calendar year ($t+1$) based on three consecutive years ($t-2, t-1, t$) of longitudinal operational exposure, operator oversight records, and platform structural characteristics.

### Key Objectives
1. **Dynamic Risk-Based Prioritization**: Replace static checklists with an empirical probability score ranking platforms most likely to require regulatory intervention.
2. **Multimodal Information Integration**: Jointly model time-series production dynamics, operator compliance histories, static physical characteristics, and categorical metadata.
3. **Strict Temporal Integrity & Leakage Prevention**: Eliminate look-ahead bias and contemporaneous leakage through strict retrospective chronological partitioning and sequence-aligned design.
4. **Calibrated Probabilities & Budget Rankings**: Provide Platt-scaled calibrated probabilities and Precision@$K$ ($K \in \{50, 100, 200\}$) metrics aligned to operational inspection budgets.

---

## 🏛️ System Architecture: TECR-Net

TECR-Net resolves severe class imbalance (~3.5% positive prevalence) and multirate data heterogeneity through a specialized three-pathway architecture:

```
                          ┌────────────────────────────────────────────────────────┐
                          │ 3 Consecutive Years (t-2, t-1, t) x 26 Dynamic Features│
                          └───────────────────────────┬────────────────────────────┘
                                                      │
                                           [Dynamic Feature Gate]
                                            x * σ(W_g * x + b_g)
                                                      │
                                            [Bidirectional GRU]
                                            (Hidden dim: 48-80)
                                                      │
                                             [Temporal Attention]
                                                      │
                                                      ▼
                                         [Temporal Representation]
                                                      │
┌─────────────────────────────┐                       │                       ┌─────────────────────────────┐
│  Static Numerical Features  │                       │                       │    Categorical Metadata     │
│ (Depth, Lat, Long, etc.)    │                       │                       │  (Area, Region, Structure)  │
└──────────────┬──────────────┘                       │                       └──────────────┬──────────────┘
               │                                      │                                      │
         [MLP Encoder]                                │                               [Embedding Layers]
               │                                      │                                      │
               ▼                                      ▼                                      ▼
         [Static Vector]                    [Cross-Modal Attention]                  [Categorical Vector]
               │                         (8-Head Multi-Head Attention)                       │
               │                                      │                                      │
               └───────────────────────────► [Residual Fusion] ◄─────────────────────────────┘
                                                      │
                                            [Classification Head]
                                           (Linear -> ReLU -> Drop)
                                                      │
                                                      ▼
                                           [Raw Ensemble Logits]
                                                      │
                                            [Logistic Calibration]
                                                 (Platt-Scale)
                                                      │
                                                      ▼
                                       [Calibrated Next-Year Risk P(y)]
```

### 1. Dynamic Feature Pathway
- **Inputs**: 26 numerical variables across 3 consecutive time steps:
  - Monthly active production months, producing well counts (mean & max).
  - Hydrocarbon flow rates: BOPD, MCFD, BOEPD, BWPD (mean, max, and YoY deltas).
  - Platform compliance history: Current-year INCs, previous-year INCs, 3-year rolling cumulative INCs, and enforcement action counts.
  - Operator-level oversight: Facilities inspected, inspection records, warning INCs, component shut-in orders, and facility shut-in orders.
- **Dynamic Feature Gate**: Computes a sigmoid-activated gating vector $g_t = \sigma(W_g x_t + b_g)$ to adaptively filter sparse or noisy features before sequential processing.
- **Bidirectional GRU (Bi-GRU)**: Captures both forward operational trends and retrospective dependencies across the 3-year sequence.
- **Temporal Attention Pooling**: Automatically weighs which historical year ($t-2$, $t-1$, or $t$) carries the greatest predictive relevance for impending non-compliance.

### 2. Static & Categorical Pathways
- **Static Numerical Encoder**: Multi-layer perceptron (MLP) mapping time-invariant platform features (water depth, coordinates).
- **Categorical Embeddings**: Independent learned embedding dimensions $d = \min(12, \max(3, \lfloor\sqrt{|\mathcal{V}|+1}\rfloor))$ for area codes, region codes, and structural classifications.

### 3. Cross-Modal Attention & Residual Fusion
- An 8-head cross-attention mechanism attends across dynamic temporal states and static/categorical representations.
- Combines cross-modal attention outputs with residual skip connections to preserve modality-specific inductive biases.

### 4. Calibration & 7-Seed Ensembling
- Models are trained across seven independent random seeds (`[11, 22, 33, 44, 55, 66, 77]`).
- Raw logit outputs are ensembled and passed through a univariate **logistic calibrator (Platt scaling)** fitted strictly on the validation cohort (2020–2021) to produce honest posterior probabilities.

---

## 📊 Dataset & Temporal Partitioning Protocol

The study uses public records from the U.S. **Bureau of Safety and Environmental Enforcement (BSEE)** covering 148,672 platform-years across 2011–2024.

### The 6 Required BSEE Tables
| Table / File Name | Description | Key Extracted Fields |
| :--- | :--- | :--- |
| `mv_prod_by_platform_all.txt` | Platform monthly production records | `COMPLEX_ID_NUM`, `STRUCTURE_NUMBER`, `BOPD`, `MCFPD`, `BOEPD`, `BWPD`, `PRODUCING_WELLS`, `PRODUCTION_DATE` |
| `mv_platstruc_structures.txt` | Structural registry & eligibility | `INSTALL_DATE`, `REMOVAL_DATE`, `AREA_CODE`, `REGION_CODE` |
| `mv_platstruc_platformincs.txt` | Specific platform-level violations | `INC_NUMBER`, `INC_ENFRC_CODE`, `INC_ISSUE_DATE`, `MMS_COMPANY_NUM` |
| `mv_insp_by_oper.txt` | Annual operator inspection audits | `FACILITIES_INSPECTED`, `INSPECTION_RECORDS`, `INSPECTION_TYPES`, `INS_DATE` |
| `mv_incs_by_oper.txt` | Operator enforcement actions | `WARNING_INCS`, `COMP_SHUTIN_INCS`, `FAC_SHUTIN_INCS`, `INC_DATE` |
| `mv_acc_investigations.txt` | Investigated accident events | `ACCIDENT_TYPE`, `DATE_OCCURRED`, `LEASE_NUMBER` |

### Chronological Splitting (Zero Look-Ahead Bias)
To prevent temporal data leakage, records are segmented chronologically:
- **Inner Fit (2014–2017)**: Training model parameters and feature gates.
- **Inner Tuning (2018–2019)**: Architecture tuning, learning rate scheduling, and early stopping.
- **Calibration (2020–2021)**: Post-hoc Platt scaling and F1-optimal threshold selection.
- **Exploratory Cohort (2022–2024)**: Preserved for chronological accounting under platform-clustered bootstrap validation.

---

## 🚀 Running on Kaggle (Step-by-Step Guide)

You can run the full end-to-end pipeline directly on Kaggle with GPU or CPU acceleration.

### Step 1: Prepare the Kaggle Dataset
1. Collect the 6 raw BSEE `.txt` tables listed above.
2. Go to **[Kaggle Datasets](https://www.kaggle.com/datasets)** -> Click **New Dataset**.
3. Upload all 6 files into a dataset named `bsee-raw-data` (or similar).

### Step 2: Create a New Kaggle Notebook
1. Open [Kaggle Notebooks](https://www.kaggle.com/code) -> Click **New Notebook**.
2. In the right-hand **Notebook Settings** panel:
   - **Accelerator**: Select **GPU T4 x2** or **GPU P100** (CPU also works, but GPU is faster).
   - **Internet**: Can be toggled **OFF** (all requirements are standard library + pre-installed Kaggle packages: `torch`, `scikit-learn`, `pandas`, `numpy`).
3. Click **Add Input** (top right) -> Search and attach your `bsee-raw-data` dataset.
4. Upload `Final BSEE.ipynb` or this repository's notebook files.

### Step 3: Execute via Notebook or Script

#### Method A: Interactive Notebook (`Final BSEE.ipynb`)
1. Open `Final BSEE.ipynb` in Kaggle.
2. Verify input path:
   ```python
   # The notebook automatically searches for required files in /kaggle/input
   INPUT_DIR = "/kaggle/input"
   OUTPUT_DIR = "/kaggle/working/tecr_net_corrected"
   ```
3. Click **Run All**.
   - **Stage 1 (Cell 1–15)**: Preprocesses raw tables into a clean annual panel without leakage.
   - **Stage 2 (Cell 16–35)**: Constructs exact 3-year consecutive feature sequences.
   - **Stage 3 (Cell 36–45)**: Trains the 7-seed TECR-Net model with AdamW and temporal attention.
   - **Stage 4 (Cell 46–52)**: Applies Platt calibration and F1-optimal threshold selection on the 2020–2021 cohort.
   - **Stage 5 (Cell 53–57)**: Computes benchmark comparisons (Logistic Regression, HistGradientBoosting, TabNet), platform-clustered bootstrap confidence intervals, and budget rankings (Precision@50/100/200).

#### Method B: Single-Cell Script Execution
You can paste and run the end-to-end script directly inside any Kaggle notebook code cell:
```python
# In a Kaggle notebook cell:
!python -c "
import torch
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
"
```

---

## 📁 Output Artifacts Generated by the Pipeline

Upon execution, `/kaggle/working/tecr_net_corrected/` will contain:

| Artifact | Contents |
| :--- | :--- |
| `annual_risk_set.csv` | Full annual platform risk set from 2011 to 2024 with engineered variables. |
| `sequence_index.csv` | Extracted $(t-2, t-1, t \to t+1)$ temporal sequences mapped to role splits. |
| `predictions.csv` | Platform sequence identifiers, raw ensemble logits, calibrated probabilities, and decision thresholds. |
| `metrics.csv` | Comparative metrics (PR-AUC, ROC-AUC, F1, Precision, Recall, Brier score, MCC) for TECR-Net vs. baselines. |
| `annual_unique_platform_precision_at_k.csv` | Precision@$K$ and Recall@$K$ for operational inspection quotas ($K=50, 100, 200$). |
| `exploratory_platform_cluster_bootstrap_ci.csv` | 95% confidence intervals derived from 1,000 platform-clustered bootstrap replicates. |
| `tecr_net_7seed.pt` | Saved PyTorch model checkpoint weights for all 7 random seeds. |
| `preprocessing_and_calibration.joblib` | Serialized preprocessor quantiles, median imputers, and Platt logistic calibrator. |
| `run_manifest.json` | Provenance log with input file SHA-256 hashes, library versions, hyperparameters, and environment specs. |

---

## 💻 Local Interactive Web Application (Flask UI)

The repository includes a web interface under the [`UI/`](file:///e:/DataCon/Project%204%20Oil%20&%20Gas/UI) folder allowing inspectors and researchers to evaluate platform risks in real time.

```
┌─────────────────────────────────────────────────────────────┐
│             TECR-Net Offshore Risk Prioritization           │
├──────────────────────────────┬──────────────────────────────┤
│ Inputs:                      │ Model Output:                │
│  - 3 Consecutive Years Data  │  - 7-Seed Mean Risk Score    │
│  - Static Platform Features  │  - Between-Seed Std Dev      │
│  - Operator Oversight Levels │  - Regulatory Action Status  │
│                              │  - Modality Attention Weights│
└──────────────────────────────┴──────────────────────────────┘
```

### Running Locally
```bash
# 1. Navigate to the UI folder
cd "UI"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the server
python app.py
```
Open `http://127.0.0.1:8501` in your browser.

---

## 🔬 Experimental Results & Baseline Comparison

Evaluated on the calibrated development and exploratory cohorts:
- **PR-AUC (Average Precision)**: TECR-Net demonstrates superior precision on the sparse positive non-compliance class relative to tabular baselines (HistGradientBoosting, Logistic Regression).
- **ROC-AUC**: Consistently achieves 0.63–0.66 across seeds while preserving calibrated probability ranges.
- **Precision@100 (Inspection Budget Efficiency)**: In operational settings where agencies can inspect 100 facilities per year, TECR-Net enriches the true INC discovery rate by more than $3\times$ compared to uninformed random selection.

---

## ⚖️ Ethical & Regulatory Disclaimers

1. **Administrative Outcome vs. Latent Reality**: TECR-Net forecasts the probability of a *recorded* Incident of Non-Compliance in administrative archives. It does not measure latent equipment deterioration or uninspected violations.
2. **Decision Support, Not Automated Enforcement**: Predictions are designed solely to assist inspection scheduling and allocation of supervisory resources. They must never serve as the sole legal basis for administrative penalties, operational sanctions, or liability attributions.
3. **Exploratory Accounting**: Results on the 2022–2024 cohorts reflect exploratory analysis on archived data; prospective operational field validation is required prior to production deployment.
