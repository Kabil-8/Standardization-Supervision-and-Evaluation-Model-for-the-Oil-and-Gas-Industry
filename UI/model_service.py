"""
TECR-Net Model Service.
Encapsulates frozen multi-seed deep learning ensemble inference for offshore platform recorded-INC risk.
Handles artifact discovery, device allocation, forward inference, Platt calibration,
and diagnostic attention attribution without modifying model parameters.
"""

from __future__ import annotations
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn

from preprocessing import FrozenPreprocessor, DYNAMIC_COLUMNS, STATIC_COLUMNS, CATEGORICAL_COLUMNS

logger = logging.getLogger("tecrnet.model_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


ARCHIVED_ENSEMBLE_THRESHOLD: float = 0.04908876769816646  # Exact archived validation F1 ensemble operating threshold (0.04908877 ~ 0.04909)
LEGACY_SEED_MEAN_THRESHOLD: float = 0.04981793852808685  # Unweighted mean of individual seed checkpoints (~0.04982)
SYMMETRIC_COMPARATOR_THRESHOLD: float = 0.07110931251823678  # Sequence-aligned symmetric baseline comparator threshold (~0.07111)
DEFAULT_OPERATING_THRESHOLD: float = ARCHIVED_ENSEMBLE_THRESHOLD


class TECRNet(nn.Module):
    """Temporal Cross-Modal Residual Network for Recorded INC Risk Assessment."""

    def __init__(
        self,
        n_dynamic: int,
        n_static: int,
        cardinalities: List[int],
        hidden: int = 80,
        dropout: float = 0.269,
        gru_layers: int = 1,
        attention_heads: int = 8,
    ):
        super().__init__()
        self.feature_logits = nn.Parameter(torch.zeros(n_dynamic))
        self.temporal = nn.GRU(
            n_dynamic,
            hidden,
            gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.nonrecurrent_temporal = nn.Linear(n_dynamic, hidden * 2)
        self.temporal_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.temporal_score = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(n_static, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        dims = [min(16, max(3, int(round(math.sqrt(c))))) for c in cardinalities]
        self.embeddings = nn.ModuleList([
            nn.Embedding(c, d, padding_idx=0) for c, d in zip(cardinalities, dims)
        ])
        self.category_encoder = nn.Sequential(
            nn.Linear(sum(dims), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
        )
        self.modality_attention = nn.MultiheadAttention(
            hidden, attention_heads, dropout=dropout, batch_first=True
        )
        self.query = nn.Parameter(torch.zeros(1, 1, hidden))
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, dynamic: torch.Tensor, static: torch.Tensor, categorical: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gated = dynamic * torch.sigmoid(self.feature_logits)
        seq, _ = self.temporal(gated)
        time_weights = torch.softmax(self.temporal_score(seq).squeeze(-1), dim=1)
        temporal = torch.sum(seq * time_weights.unsqueeze(-1), dim=1)
        temporal = self.temporal_projection(temporal)

        static_token = self.static_encoder(static)
        embedded = torch.cat(
            [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)], dim=1
        )
        category_token = self.category_encoder(embedded)

        modalities = torch.stack([temporal, static_token, category_token], dim=1)
        query = self.query.expand(dynamic.size(0), -1, -1)
        attended, modal_weights = self.modality_attention(
            query, modalities, modalities, need_weights=True, average_attn_weights=True
        )
        fused = torch.cat([attended.squeeze(1), modalities.mean(dim=1)], dim=1)
        logits = self.classifier(fused).squeeze(1)
        return logits, time_weights, modal_weights.squeeze(1)


class TECRNetService:
    """Manages frozen model artifacts, device binding, and prediction pipelines."""

    def __init__(
        self,
        artifact_dir: Optional[Union[str, Path]] = None,
        device_preference: str = "auto",
        threshold: Optional[float] = DEFAULT_OPERATING_THRESHOLD,
    ):
        self.artifact_path = self._locate_artifact_dir(artifact_dir)
        self.device = self._resolve_device(device_preference)
        logger.info(f"Initializing TECRNetService from '{self.artifact_path}' on device '{self.device}'")

        self.preprocessing_config = self._load_json("preprocessing.json")
        self.best_params = self._load_json("best_params.json")
        self.preprocessor = FrozenPreprocessor(self.preprocessing_config)

        self.models: List[Tuple[TECRNet, float, float]] = []
        self.thresholds: List[float] = []
        self.seeds: List[int] = []

        self._load_seed_models()

        if len(self.models) != 7:
            raise RuntimeError(
                f"Expected exactly 7 TECR-Net seed models, but loaded {len(self.models)}. "
                "Ensure checkpoints tecrnet_seed_11.pt through tecrnet_seed_77.pt are present."
            )

        self.manifest_id = "eeec8023"
        self.archived_ensemble_threshold = ARCHIVED_ENSEMBLE_THRESHOLD
        self.legacy_seed_mean_threshold = float(np.mean(self.thresholds))
        self.symmetric_threshold = SYMMETRIC_COMPARATOR_THRESHOLD
        self.threshold = float(threshold) if threshold is not None else DEFAULT_OPERATING_THRESHOLD
        logger.info(
            f"Successfully initialized 7 seed models. Active decision threshold: {self.threshold:.8f} "
            f"(Archived Ensemble F1: {self.archived_ensemble_threshold:.8f}, "
            f"Legacy seed-mean: {self.legacy_seed_mean_threshold:.8f}, "
            f"Symmetric comparator: {self.symmetric_threshold:.8f})"
        )

    def get_threshold_mode(self) -> str:
        """Returns the human-readable standard description for the active threshold."""
        if abs(self.threshold - self.archived_ensemble_threshold) < 1e-5:
            return "Archived Ensemble F1 (0.04908877 | rounded 0.04909)"
        elif abs(self.threshold - self.symmetric_threshold) < 1e-5:
            return "Symmetric Comparator (0.07111)"
        elif abs(self.threshold - self.legacy_seed_mean_threshold) < 1e-5:
            return "Legacy Seed-Mean (0.04982)"
        return f"Custom ({self.threshold:.5f})"

    def set_threshold(self, new_threshold: float) -> None:
        """Sets the decision threshold."""
        self.threshold = float(new_threshold)
        logger.info(f"Active decision threshold updated to: {self.threshold:.8f} ({self.get_threshold_mode()})")

    def _locate_artifact_dir(self, custom_path: Optional[Union[str, Path]]) -> Path:
        """Finds the artifact directory, prioritizing user path, then artifacts/, then model_artifacts/."""
        candidates = []
        if custom_path:
            candidates.append(Path(custom_path))
        base_dir = Path(__file__).resolve().parent
        candidates.extend([
            base_dir / "artifacts",
            base_dir / "model_artifacts",
            Path.cwd() / "artifacts",
            Path.cwd() / "model_artifacts",
        ])

        for path in candidates:
            if path.exists() and (path / "preprocessing.json").exists() and (path / "best_params.json").exists():
                return path

        raise FileNotFoundError(
            f"Could not locate frozen TECR-Net model artifacts. Checked locations:\n"
            + "\n".join(f" - {p}" for p in candidates)
            + "\nPlease extract the model package containing preprocessing.json, best_params.json, "
            "and tecrnet_seed_*.pt files into the artifacts/ directory."
        )

    def _resolve_device(self, preference: str) -> torch.device:
        if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_json(self, filename: str) -> Dict[str, Any]:
        file_path = self.artifact_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required model artifact missing: {file_path}")
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Error parsing artifact JSON '{file_path}': {e}") from e

    def _load_seed_models(self) -> None:
        """Loads and freezes the 7 TECR-Net seed checkpoints."""
        checkpoint_files = sorted(self.artifact_path.glob("tecrnet_seed_*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files matching 'tecrnet_seed_*.pt' found in {self.artifact_path}")

        n_dyn = len(self.preprocessing_config["dynamic_columns"])
        n_stat = len(self.preprocessing_config["static_columns"])
        cardinalities = self.preprocessing_config["cardinalities"]

        net_params = {
            k: self.best_params[k]
            for k in ["hidden", "dropout", "gru_layers", "attention_heads"]
        }

        for path in checkpoint_files:
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                model = TECRNet(n_dyn, n_stat, cardinalities, **net_params)
                model.load_state_dict(ckpt["model_state"], strict=False)
                model.to(self.device)
                model.eval()

                # Calibrator parameters
                coef = float(np.asarray(ckpt["calibrator_coef"]).reshape(-1)[0])
                intercept = float(np.asarray(ckpt["calibrator_intercept"]).reshape(-1)[0])
                threshold = float(ckpt["threshold"])
                seed = int(ckpt.get("seed", 0))

                self.models.append((model, coef, intercept))
                self.thresholds.append(threshold)
                self.seeds.append(seed)
                logger.info(f"Loaded seed model {seed} from {path.name} (threshold={threshold:.5f})")
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint '{path.name}': {e}") from e

    def health_check(self) -> Dict[str, Any]:
        """Verifies artifact presence, feature count, tensor dimensions, and forward inference."""
        checks = {
            "artifact_directory": str(self.artifact_path),
            "manifest_id": self.manifest_id,
            "device": str(self.device),
            "num_seed_models": len(self.models),
            "seed_ids": self.seeds,
            "threshold": self.threshold,
            "threshold_mode": self.get_threshold_mode(),
            "archived_ensemble_threshold": self.archived_ensemble_threshold,
            "legacy_seed_mean_threshold": self.legacy_seed_mean_threshold,
            "symmetric_threshold": self.symmetric_threshold,
            "calibration_protocol_disclosure": (
                "Platt-like scaling and operating threshold (0.04908877 | rounded 0.04909) were fitted/selected on the 2020-2021 "
                "validation cohort. Reported validation metrics are development-selected optimizations. "
                "The 2022–2024 later cohort is exploratory because its outcomes were previously inspected; no independent evaluation has yet been completed."
            ),
            "dynamic_feature_count": len(self.preprocessing_config["dynamic_columns"]),
            "static_feature_count": len(self.preprocessing_config["static_columns"]),
            "categorical_feature_count": len(self.preprocessing_config["categorical_columns"]),
        }

        # Test dummy forward pass
        try:
            dummy_dyn = torch.zeros((1, 3, len(self.preprocessing_config["dynamic_columns"])), device=self.device)
            dummy_stat = torch.zeros((1, len(self.preprocessing_config["static_columns"])), device=self.device)
            dummy_cat = torch.zeros((1, len(self.preprocessing_config["categorical_columns"])), dtype=torch.long, device=self.device)

            with torch.no_grad():
                for idx, (model, coef, intercept) in enumerate(self.models):
                    logits, tw, mw = model(dummy_dyn, dummy_stat, dummy_cat)
                    p = torch.sigmoid(logits).item()
                    cal_p = 1.0 / (1.0 + math.exp(-(coef * p + intercept)))
                    if not (0.0 <= cal_p <= 1.0):
                        raise ValueError(f"Seed {self.seeds[idx]} produced out-of-range probability: {cal_p}")

            checks["forward_inference_check"] = "PASSED"
            checks["status"] = "HEALTHY"
        except Exception as e:
            checks["forward_inference_check"] = f"FAILED: {e}"
            checks["status"] = "UNHEALTHY"
            logger.error(f"Health check failed: {e}", exc_info=True)

        return checks

    @staticmethod
    def get_risk_band(probability: float, threshold: float) -> Tuple[str, str, str]:
        """Classifies probability into research prioritization bands.
        Returns:
            (band_label, description, badge_color)
        """
        if threshold <= 0:
            return "Indeterminate", "Threshold not defined", "#6c757d"

        ratio = probability / threshold
        if probability < 0.5 * threshold:
            return "Low (Heuristic)", "Below 0.5 × Saved Threshold (Heuristic)", "#10B981"  # Emerald Teal
        elif probability < threshold:
            return "Moderate (Heuristic)", "0.5 × Threshold to Below Threshold (Heuristic)", "#0284C7"  # Deep Sky Blue
        elif probability < 2.0 * threshold:
            return "Elevated (Heuristic)", "Threshold to Below 2 × Threshold (Heuristic)", "#D97706"  # Amber Orange
        else:
            return "Higher (Heuristic)", "At least 2 × Saved Threshold (Heuristic)", "#DC2626"  # Crimson / Dark Red

    def predict_single(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Executes full inference for a single platform 3-year sequence."""
        years = payload.get("years", [])
        if len(years) != 3:
            raise ValueError(f"Exactly three consecutive feature years are required, got {len(years)}")

        # Step 1: Preprocessing with median imputation and OOV category logging
        dyn_arr, dyn_imputed = self.preprocessor.transform_dynamic_sequence(years)
        stat_arr, stat_imputed = self.preprocessor.transform_static(payload.get("static", {}))
        cat_arr, cat_unknown = self.preprocessor.transform_categorical(payload.get("categorical", {}))

        # Step 2: Convert to tensors
        dyn_t = torch.tensor(dyn_arr[None, :, :], dtype=torch.float32, device=self.device)
        stat_t = torch.tensor(stat_arr[None, :], dtype=torch.float32, device=self.device)
        cat_t = torch.tensor(cat_arr[None, :], dtype=torch.long, device=self.device)

        # Step 3: Run inference across all 7 seed models
        calibrated_probs: List[float] = []
        raw_neural_probs: List[float] = []
        temporal_attns: List[np.ndarray] = []
        modality_attns: List[np.ndarray] = []

        with torch.no_grad():
            for model, coef, intercept in self.models:
                logits, tw, mw = model(dyn_t, stat_t, cat_t)
                p = torch.sigmoid(logits).item()
                cal_p = 1.0 / (1.0 + math.exp(-(coef * p + intercept)))
                raw_neural_probs.append(p)
                calibrated_probs.append(cal_p)
                temporal_attns.append(tw.cpu().numpy()[0])
                modality_attns.append(mw.cpu().numpy()[0])

        ensemble_prob = float(np.mean(calibrated_probs))
        seed_sd = float(np.std(calibrated_probs, ddof=1)) if len(calibrated_probs) > 1 else 0.0

        band_name, band_desc, band_color = self.get_risk_band(ensemble_prob, self.threshold)
        is_above_threshold = ensemble_prob >= self.threshold

        # Diagnostic weights
        avg_temporal_attn = np.mean(temporal_attns, axis=0)
        avg_modality_attn = np.mean(modality_attns, axis=0)
        gates = np.mean(
            [torch.sigmoid(m.feature_logits).detach().cpu().numpy() for m, _, _ in self.models],
            axis=0,
        )
        ranked_gates = sorted(
            zip(self.preprocessor.dynamic_cols, gates.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "ensemble_probability": ensemble_prob,
            "ensemble_percentage": ensemble_prob * 100,
            "saved_threshold": self.threshold,
            "saved_threshold_percentage": self.threshold * 100,
            "threshold_mode": self.get_threshold_mode(),
            "archived_ensemble_threshold": self.archived_ensemble_threshold,
            "legacy_seed_mean_threshold": self.legacy_seed_mean_threshold,
            "symmetric_threshold": self.symmetric_threshold,
            "is_above_threshold": is_above_threshold,
            "status_label": "Above Saved Threshold" if is_above_threshold else "Below Saved Threshold",
            "risk_ratio_to_threshold": ensemble_prob / self.threshold if self.threshold > 0 else None,
            "prioritization_band": band_name,
            "prioritization_band_desc": band_desc,
            "prioritization_band_color": band_color,
            "seed_mean": ensemble_prob,
            "seed_sd": seed_sd,
            "seed_variability_label": "Seed Standard Deviation (Model Variability, not Confidence Interval)",
            "seed_predictions": [
                {
                    "seed": s,
                    "calibrated_probability": cp,
                    "calibrated_percentage": cp * 100,
                    "raw_probability": rp,
                    "threshold": th,
                }
                for s, cp, rp, th in zip(self.seeds, calibrated_probs, raw_neural_probs, self.thresholds)
            ],
            "temporal_attention": {
                "t-2": float(avg_temporal_attn[0]),
                "t-1": float(avg_temporal_attn[1]),
                "t": float(avg_temporal_attn[2]),
            },
            "modality_attention": {
                "Temporal Dynamics": float(avg_modality_attn[0]),
                "Static Platform/Location": float(avg_modality_attn[1]),
                "Categorical Structure/Flags": float(avg_modality_attn[2]),
            },
            "top_feature_gates": [{"feature": name, "weight": float(wt)} for name, wt in ranked_gates[:10]],
            "imputed_features": dyn_imputed + stat_imputed,
            "unknown_categories": cat_unknown,
            "regulatory_disclaimer": (
                "TECR-Net is a retrospective research prototype for prioritizing human review. "
                "It predicts recorded INC outcomes, which may reflect inspection intensity and administrative processes. "
                "It has not undergone independent prospective validation and must not be used as the sole basis for enforcement, "
                "sanctions or reduced oversight."
            ),
        }

    def predict_batch(self, valid_sequences: List[Dict[str, Any]]) -> pd.DataFrame:
        """Processes multiple valid 3-year sequences and compiles a results DataFrame."""
        rows = []
        for seq in valid_sequences:
            result = self.predict_single(seq)
            platform_id = seq.get("platform_id", "Unknown")
            t_minus_2 = seq.get("t_minus_2_year", "t-2")
            t_year = seq.get("t_year", "t")
            target_year = seq.get("target_prediction_year", "t+1")

            rows.append({
                "Platform Identifier": platform_id,
                "Observation Window": f"{t_minus_2} - {t_year}",
                "Target Year (t+1)": target_year,
                "Ensemble Ranking Score": round(result["ensemble_probability"], 5),
                "Risk (%)": round(result["ensemble_percentage"], 2),
                "Threshold Status": result["status_label"],
                "Heuristic Research Band": result["prioritization_band"],
                "Seed SD (Variability)": round(result["seed_sd"], 5),
                "Risk Ratio to Threshold": round(result["risk_ratio_to_threshold"], 2) if result["risk_ratio_to_threshold"] else np.nan,
                "Imputed Fields Count": len(result["imputed_features"]),
                "Unknown Categories Count": len(result["unknown_categories"]),
            })

        return pd.DataFrame(rows)
