# TECR-Net Offshore Platform Recorded-INC Risk Decision-Support System

An interactive research decision-support prototype that produces an apparent probability-mapped score that a U.S. Outer Continental Shelf (OCS) offshore platform will experience a recorded Incident of Non-Compliance (INC) in observation year $t+1$.

---

## ⚠️ Mandatory Research & Regulatory Disclaimer

> **TECR-Net is a retrospective research prototype for prioritizing human review. It predicts recorded INC outcomes, which may reflect inspection intensity and administrative processes. It has not undergone independent prospective validation and must not be used as the sole basis for enforcement, sanctions or reduced oversight.**
>
> Model predictions represent heuristic likelihood scores intended strictly to support analytical triage. The application never describes outputs as confirmed latent non-compliance, regulatory guilt, or autonomous enforcement decisions.

---

## Quickstart & Installation

Follow these instructions to run the application locally:

### 1. Extract the Frozen Model Package into `artifacts/`
Extract all frozen checkpoints and configuration files into the `artifacts/` directory within the application root:
```
artifacts/
├── best_params.json           # Optimal hyperparameters (hidden units, attention heads, etc.)
├── preprocessing.json         # Medians, scales, clipping bounds, vocabularies, cardinalities
├── tecrnet_seed_11.pt         # Seed 11 checkpoint (weights, Platt calibrator, threshold)
├── tecrnet_seed_22.pt         # Seed 22 checkpoint
├── tecrnet_seed_33.pt         # Seed 33 checkpoint
├── tecrnet_seed_44.pt         # Seed 44 checkpoint
├── tecrnet_seed_55.pt         # Seed 55 checkpoint
├── tecrnet_seed_66.pt         # Seed 66 checkpoint
└── tecrnet_seed_77.pt         # Seed 77 checkpoint
```

### 2. Install Dependencies
Ensure Python 3.10+ is installed, then install all required packages:
```bash
pip install -r requirements.txt
```

### 3. Start the Interface
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

### 4. Open the Interface
Open the displayed local URL in your web browser (typically `http://localhost:8501`).

---

## Directory & File Structure

```
.
├── app.py                     # Streamlit application UI and dashboard workflows
├── model_service.py           # TECRNet architecture, 7-seed loading, Platt calibration & health check
├── preprocessing.py           # Frozen transformation, validation, and missing/OOV logging
├── requirements.txt           # Python dependencies
├── sample_input.csv           # Multi-platform sample CSV for batch evaluation & exclusion testing
├── sample_input.json          # Single-platform assessment sample data
├── assets/
│   └── style.css              # Custom styling (Navy, Teal, and Amber theme)
├── artifacts/                 # Primary model package directory
│   ├── best_params.json
│   ├── preprocessing.json
│   └── tecrnet_seed_*.pt
└── README.md                  # System documentation and instructions
```

---

## Feature Specifications

TECR-Net consumes exactly **37 variables** across three distinct modalities:

### 1. Dynamic Features (26 variables, measured across $t-2, t-1, t$)
Exactly $26 \times 3 = 78$ dynamic scalar values enter the sequential recurrent pipeline:
* `active_months`
* `producing_wells_mean`
* `producing_wells_max`
* `bopd_mean`
* `bopd_max`
* `mcfd_mean`
* `mcfd_max`
* `boepd_mean`
* `bwpd_mean`
* `platform_age`
* `operator_facilities_inspected`
* `operator_inspection_records`
* `operator_inspection_types`
* `investigated_accident_count`
* `accident_type_count`
* `previous_year_inc_count`
* `inc_count_previous_3_observations`
* `previous_year_accident_count`
* `production_change_bopd`
* `production_change_mcfd`
* `inc_count`
* `enforcement_code_count`
* `any_inc`
* `operator_warning_incs`
* `operator_component_shutin_incs`
* `operator_facility_shutin_incs`

### 2. Static Numerical Features (3 variables, entered once)
* `water_depth` (feet)
* `latitude` (degrees)
* `longitude` (degrees)

### 3. Categorical Variables (8 variables, entered once)
Mapped via frozen categorical vocabularies into entity embeddings:
* `area_code` (e.g., `AC`, `GC`, `MC`, `VR`, etc.)
* `region_code` (`G` = Gulf of Mexico, `P` = Pacific, `Y` = Alaska)
* `structure_type` (`CAIS`, `FIXED`, `WP`, `TLP`, etc.)
* `major_structure_flag` (`Y`, `N`)
* `district_code` (`1.0` through `5.0`)
* `heliport_flag` (`Y`, `N`)
* `attended_8hr_flag` (`Y`, `N`)
* `manned_24hr_flag` (`Y`, `N`)

---

## Prediction Logic & Research Prioritization Bands

1. **Frozen Preprocessing:** Missing numerical values are imputed using the frozen training medians. Values are clipped to $[lower, upper]$ bounds and standardized: $(x - center) / scale$.
2. **Multi-Seed Ensemble:** Seven distinct TECR-Net seed models evaluate the input. Seed scores are transformed using the archived Platt-like probability mapping (fitted on probability inputs, not conventional logit-input Platt scaling).
3. **Ensemble Ranking Score:** Arithmetic average of the seven calibrated probabilities.
4. **Model Variability:** Disagreement among seeds is reported as the sample standard deviation ($SD_{\text{seeds}}$), quantifying architectural variability across seeds rather than a frequentist confidence interval.
5. **Decision Thresholds & Calibration-Selection Protocol:**
   * **Archived Ensemble F1 Operating Threshold (Active Locked Default):** $\tau = 0.04908877$ ($0.04908876769816646$, displayed rounded $0.04909$), the maximum validation F1 threshold selected on the pooled 7-seed ensemble from manifest `eeec8023`. Restores exact reproducibility with the frozen checkpoint metrics.
   * **Symmetric Benchmark Threshold:** $\tau = 0.07110931$ ($7.11\%$), the sequence-aligned comparator threshold from symmetric baseline evaluation (read-only comparison).
   * **Legacy Seed-Mean Threshold:** $\tau_{\text{legacy}} \approx 0.049818$ ($4.98\%$), the unweighted arithmetic mean of individual seed checkpoints (historical reference only).
   * *Protocol Disclosure:* Both Platt-like calibrators and the ensemble threshold ($\tau = 0.04908877 | 0.04909$) were fitted/selected on the 2020–2021 validation cohort. Validation results represent *development-selected optimization* rather than unbiased post-selection performance. The 2022–2024 later cohort is exploratory because its outcomes were previously inspected; no independent evaluation has yet been completed. Arbitrary threshold switching is disabled in the interface to prevent model–threshold mismatch.
6. **Heuristic Research Prioritization Bands (Interface Categories):**
   * **Low (Heuristic):** $p < 0.5 \times \tau$ ($< 2.45\%$ under default $\tau = 0.04908877$)
   * **Moderate (Heuristic):** $0.5 \times \tau \le p < \tau$ ($2.45\% - 4.91\%$)
   * **Elevated (Heuristic):** $\tau \le p < 2 \times \tau$ ($4.91\% - 9.82\%$)
   * **Higher (Heuristic):** $p \ge 2 \times \tau$ ($\ge 9.82\%$)

---

## Input Modes

### Mode 1: Single-Platform Research Assessment
* Interactive form allowing manual parameter specification or instant loading via "📋 Load Sample Platform Data".
* Enforces three consecutive observation years ($t-2, t-1, t$) for the target outcome ($t+1$).
* Generates interactive 7-seed distribution bar charts, multi-year operational trend line plots, cross-modal attention diagnostic charts, and a downloadable JSON assessment dossier.

### Mode 2: Batch Assessment
* Allows uploading CSV datasets with multiple platforms and multi-year observations.
* Automatically groups records by platform identifier and filters for complete, strictly consecutive 3-year sequences.
* Identifies and logs excluded records with transparent audit explanations (e.g. gaps in years, insufficient observation duration).
* Provides summary metrics, risk distribution histograms, filtered interactive tables, and one-click CSV export of predictions.
