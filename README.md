# Standardization-Supervision-and-Evaluation-Model-for-the-Oil-and-Gas-Industry

## TECR-Net Risk Prioritization UI

This local Flask application loads the seven archived TECR-Net seed checkpoints, applies the saved training-period preprocessing and per-seed logistic probability mappings, and returns the seven-seed mean risk score.

## Run

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Run `python -m pip install -r requirements.txt`.
4. Run `python app.py`.
5. Open `http://127.0.0.1:8501`.

Windows users may double-click `run_windows.bat`.

## Inputs

- Three consecutive annual rows: t-2, t-1, and t.
- Static numerical values: water depth, latitude, and longitude.
- Eight categorical platform-context fields.

The **Load training medians** button provides a valid demonstration input. Replace those values with measurements available at the prediction origin.

## Output

- Seven-seed calibrated probability.
- Development-selected threshold status.
- Between-seed standard deviation.
- Temporal and modality attention summaries.
- Highest global feature-gate weights.

## Important limitation

The interface is a research prototype. The score estimates the risk of a *recorded* next-year INC in the archived development design. It is not prospectively validated and must not be used as the sole basis for enforcement, sanctions, reduced oversight, or conclusions about operator culpability.
