"""
TECR-Net Preprocessing and Validation Pipeline.
Strictly applies frozen training-fitted parameters.
Ensures zero data leakage, validates temporal sequence consecutiveness,
and records all missing, imputed, and out-of-vocabulary values.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("tecrnet.preprocessing")

DYNAMIC_COLUMNS: List[str] = [
    "active_months",
    "producing_wells_mean",
    "producing_wells_max",
    "bopd_mean",
    "bopd_max",
    "mcfd_mean",
    "mcfd_max",
    "boepd_mean",
    "bwpd_mean",
    "platform_age",
    "operator_facilities_inspected",
    "operator_inspection_records",
    "operator_inspection_types",
    "investigated_accident_count",
    "accident_type_count",
    "previous_year_inc_count",
    "inc_count_previous_3_observations",
    "previous_year_accident_count",
    "production_change_bopd",
    "production_change_mcfd",
    "inc_count",
    "enforcement_code_count",
    "any_inc",
    "operator_warning_incs",
    "operator_component_shutin_incs",
    "operator_facility_shutin_incs",
]

STATIC_COLUMNS: List[str] = [
    "water_depth",
    "latitude",
    "longitude",
]

CATEGORICAL_COLUMNS: List[str] = [
    "area_code",
    "region_code",
    "structure_type",
    "major_structure_flag",
    "district_code",
    "heliport_flag",
    "attended_8hr_flag",
    "manned_24hr_flag",
]

# Sensitive or future fields that must strictly NEVER enter feature tensors
FORBIDDEN_COLUMNS: set[str] = {
    "target",
    "label",
    "outcome",
    "inc_next_year",
    "next_year_inc",
    "next_year_any_inc",
    "next_year_inc_count",
    "next_year_enforcement_code_count",
    "next_year_operator_warning_incs",
    "next_year_operator_component_shutin_incs",
    "next_year_operator_facility_shutin_incs",
    "recorded_inc_t1",
}


class FrozenPreprocessor:
    """Encapsulates frozen transformation objects loaded from preprocessing.json."""

    def __init__(self, preprocessing_config: Dict[str, Any]):
        self.config = preprocessing_config
        self.dynamic_cols = self.config.get("dynamic_columns", DYNAMIC_COLUMNS)
        self.static_cols = self.config.get("static_columns", STATIC_COLUMNS)
        self.categorical_cols = self.config.get("categorical_columns", CATEGORICAL_COLUMNS)

        self.dyn_spec = self.config["dynamic_processor"]
        self.stat_spec = self.config["static_processor"]
        self.vocabularies = self.config["category_vocabularies"]
        self.cardinalities = self.config["cardinalities"]

        # Cache pre-converted float32 arrays
        self._dyn_median = np.asarray(self.dyn_spec["median"], dtype=np.float32)
        self._dyn_lower = np.asarray(self.dyn_spec["lower"], dtype=np.float32)
        self._dyn_upper = np.asarray(self.dyn_spec["upper"], dtype=np.float32)
        self._dyn_center = np.asarray(self.dyn_spec["center"], dtype=np.float32)
        self._dyn_scale = np.asarray(self.dyn_spec["scale"], dtype=np.float32)

        self._stat_median = np.asarray(self.stat_spec["median"], dtype=np.float32)
        self._stat_lower = np.asarray(self.stat_spec["lower"], dtype=np.float32)
        self._stat_upper = np.asarray(self.stat_spec["upper"], dtype=np.float32)
        self._stat_center = np.asarray(self.stat_spec["center"], dtype=np.float32)
        self._stat_scale = np.asarray(self.stat_spec["scale"], dtype=np.float32)

    def get_defaults(self) -> Dict[str, Any]:
        """Provides default median values for dynamic and static features, and valid category defaults."""
        dyn = dict(zip(self.dynamic_cols, [float(v) for v in self._dyn_median]))
        sta = dict(zip(self.static_cols, [float(v) for v in self._stat_median]))
        cat = {name: next(iter(vocab.keys())) for name, vocab in self.vocabularies.items()}
        return {
            "years": [dyn.copy(), dyn.copy(), dyn.copy()],
            "static": sta,
            "categorical": cat,
        }

    def transform_dynamic_sequence(
        self, years_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Transforms 3 feature years of 26 dynamic variables.
        Returns:
            scaled_array: (3, 26) float32 numpy array
            imputed_log: List of missing/imputed records
        """
        if len(years_data) != 3:
            raise ValueError(f"Exactly 3 feature years required, got {len(years_data)}")

        raw_rows = []
        imputed_log = []

        year_labels = ["t-2", "t-1", "t"]
        for y_idx, y_data in enumerate(years_data):
            label = year_labels[y_idx] if y_idx < len(year_labels) else f"year_{y_idx}"
            row = []
            for col_idx, col in enumerate(self.dynamic_cols):
                val = y_data.get(col, np.nan)
                try:
                    num_val = float(val) if val is not None and str(val).strip() != "" else np.nan
                except (ValueError, TypeError):
                    num_val = np.nan

                if not np.isfinite(num_val):
                    imputed_log.append({
                        "feature": col,
                        "temporal_step": label,
                        "type": "dynamic",
                        "imputed_value": float(self._dyn_median[col_idx]),
                    })
                    num_val = float(self._dyn_median[col_idx])
                row.append(num_val)
            raw_rows.append(row)

        arr = np.asarray(raw_rows, dtype=np.float32)
        arr = np.clip(arr, self._dyn_lower, self._dyn_upper)
        transformed = (arr - self._dyn_center) / self._dyn_scale
        return transformed.astype(np.float32), imputed_log

    def transform_static(
        self, static_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Transforms the 3 static numerical variables.
        Returns:
            scaled_array: (3,) float32 numpy array
            imputed_log: List of missing/imputed records
        """
        raw_row = []
        imputed_log = []

        for col_idx, col in enumerate(self.static_cols):
            val = static_data.get(col, np.nan)
            try:
                num_val = float(val) if val is not None and str(val).strip() != "" else np.nan
            except (ValueError, TypeError):
                num_val = np.nan

            if not np.isfinite(num_val):
                imputed_log.append({
                    "feature": col,
                    "temporal_step": "static",
                    "type": "static_numerical",
                    "imputed_value": float(self._stat_median[col_idx]),
                })
                num_val = float(self._stat_median[col_idx])
            raw_row.append(num_val)

        arr = np.asarray(raw_row, dtype=np.float32)
        arr = np.clip(arr, self._stat_lower, self._stat_upper)
        transformed = (arr - self._stat_center) / self._stat_scale
        return transformed.astype(np.float32), imputed_log

    def transform_categorical(
        self, categorical_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Maps 8 categorical fields to integer vocabulary codes.
        Unknown or unmapped values receive index 0.
        Returns:
            encoded_codes: (8,) int64 numpy array
            unknown_log: List of unrecognized category values
        """
        codes = []
        unknown_log = []

        for col in self.categorical_cols:
            val = categorical_data.get(col)
            val_str = str(val).strip() if val is not None else "Unknown"
            vocab = self.vocabularies.get(col, {})
            code = int(vocab.get(val_str, 0))

            if code == 0:
                unknown_log.append({
                    "feature": col,
                    "provided_value": val_str,
                    "mapped_code": 0,
                    "known_classes": list(vocab.keys()),
                })
            codes.append(code)

        return np.asarray(codes, dtype=np.int64), unknown_log


def validate_batch_dataframe(
    df: pd.DataFrame,
    preprocessor: FrozenPreprocessor,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validates an uploaded batch DataFrame.
    Groups by platform, checks for strictly consecutive 3-year sequences (t-2, t-1, t),
    and rejects malformed sequences with detailed explanations.

    Returns:
        valid_sequences: List of dict payloads ready for TECR-Net inference
        excluded_records: List of exclusion explanations with platform ID and reason
    """
    valid_sequences: List[Dict[str, Any]] = []
    excluded_records: List[Dict[str, Any]] = []

    if df.empty:
        excluded_records.append({
            "platform_id": "N/A",
            "year": "N/A",
            "reason": "The uploaded CSV file is empty.",
        })
        return valid_sequences, excluded_records

    # Normalize column names to lowercase and strip whitespace
    df_clean = df.copy()
    df_clean.columns = [str(c).strip().lower() for c in df_clean.columns]

    # Identify platform identifier column
    id_candidates = ["platform_id", "complex_id", "platform", "platform_number", "id"]
    id_col = next((c for c in id_candidates if c in df_clean.columns), None)
    if not id_col:
        excluded_records.append({
            "platform_id": "ALL",
            "year": "ALL",
            "reason": f"Required platform identifier column missing. Expected one of: {id_candidates}",
        })
        return valid_sequences, excluded_records

    # Identify year column
    year_candidates = ["year", "feature_year", "obs_year", "t"]
    year_col = next((c for c in year_candidates if c in df_clean.columns), None)
    if not year_col:
        excluded_records.append({
            "platform_id": "ALL",
            "year": "ALL",
            "reason": f"Required year column missing. Expected one of: {year_candidates}",
        })
        return valid_sequences, excluded_records

    # Convert year to numeric integer
    df_clean["parsed_year"] = pd.to_numeric(df_clean[year_col], errors="coerce")
    invalid_year_rows = df_clean[df_clean["parsed_year"].isna()]
    if not invalid_year_rows.empty:
        for _, row in invalid_year_rows.iterrows():
            excluded_records.append({
                "platform_id": str(row[id_col]),
                "year": str(row[year_col]),
                "reason": "Invalid or missing year value (must be an integer).",
            })
        df_clean = df_clean.dropna(subset=["parsed_year"])

    df_clean["parsed_year"] = df_clean["parsed_year"].astype(int)

    # Group by platform ID
    grouped = df_clean.groupby(id_col)

    for platform_id, group in grouped:
        platform_str = str(platform_id).strip()
        sorted_group = group.sort_values(by="parsed_year")
        years = sorted_group["parsed_year"].tolist()

        if len(years) < 3:
            excluded_records.append({
                "platform_id": platform_str,
                "year": f"Available: {years}",
                "reason": f"Insufficient observation years: exactly 3 consecutive years required, found {len(years)}.",
            })
            continue

        # Look for valid consecutive 3-year triplets (y, y+1, y+2)
        # We search from the most recent downwards to find candidate sequences
        triplet_found = False
        # Iterate over all possible contiguous windows of 3
        for i in range(len(sorted_group) - 2):
            w = sorted_group.iloc[i : i + 3]
            w_years = w["parsed_year"].tolist()
            if w_years[1] == w_years[0] + 1 and w_years[2] == w_years[1] + 1:
                triplet_found = True
                t_minus_2_yr = w_years[0]
                t_minus_1_yr = w_years[1]
                t_yr = w_years[2]
                predict_yr = t_yr + 1

                # Construct payload
                years_dynamic = []
                for _, r in w.iterrows():
                    row_dyn = {col: r.get(col, np.nan) for col in preprocessor.dynamic_cols}
                    years_dynamic.append(row_dyn)

                # Static features from the latest observation (t)
                latest_row = w.iloc[2]
                static_dict = {col: latest_row.get(col, np.nan) for col in preprocessor.static_cols}

                # Categorical features from the latest observation (t)
                cat_dict = {col: latest_row.get(col, "Unknown") for col in preprocessor.categorical_cols}

                valid_sequences.append({
                    "platform_id": platform_str,
                    "t_minus_2_year": t_minus_2_yr,
                    "t_minus_1_year": t_minus_1_yr,
                    "t_year": t_yr,
                    "target_prediction_year": predict_yr,
                    "years": years_dynamic,
                    "static": static_dict,
                    "categorical": cat_dict,
                    "display_metadata": {
                        "platform_id": platform_str,
                        "time_span": f"{t_minus_2_yr} -> {t_yr}",
                        "target_year": predict_yr,
                    },
                })

        if not triplet_found:
            excluded_records.append({
                "platform_id": platform_str,
                "year": f"Available: {years}",
                "reason": "No strictly consecutive 3-year sequence (t-2, t-1, t) found among available records.",
            })

    return valid_sequences, excluded_records
