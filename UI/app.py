"""
TECR-Net: Offshore Platform Recorded-INC Risk Model
Interactive Decision-Support Research Dashboard
"""

from __future__ import annotations
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model_service import TECRNetService
from preprocessing import (
    DYNAMIC_COLUMNS,
    STATIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    validate_batch_dataframe,
)

# Configure logging
logger = logging.getLogger("tecrnet.app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Streamlit Page Setup
st.set_page_config(
    page_title="TECR-Net | Offshore Platform Risk Decision Support",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom CSS
def load_custom_css():
    css_path = Path(__file__).resolve().parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

load_custom_css()

# Model Service Resource Caching (Prevents reload and retraining on rerun)
@st.cache_resource(show_spinner="Loading frozen TECR-Net model artifacts...")
def get_model_service() -> TECRNetService:
    logger.info("Instantiating cached TECRNetService...")
    return TECRNetService()

try:
    service = get_model_service()
    if not hasattr(service, "archived_ensemble_threshold") or not hasattr(service, "get_threshold_mode"):
        st.cache_resource.clear()
        service = get_model_service()
    model_loaded = True
    load_error_message = ""
except Exception as e:
    model_loaded = False
    load_error_message = str(e)
    logger.error(f"Failed to load TECRNetService: {e}", exc_info=True)


# Render Mandatory Regulatory Disclaimer Banner
def render_disclaimer():
    st.markdown(
        """
        <div class="disclaimer-banner">
            <div class="disclaimer-title">
                <span>⚠️ RESEARCH DECISION-SUPPORT PROTOTYPE NOTICE</span>
            </div>
            <p class="disclaimer-body">
                <strong>TECR-Net is a retrospective research prototype for prioritizing human review.</strong>
                It predicts recorded INC outcomes, which may reflect inspection intensity and administrative processes.
                It has not undergone independent prospective validation and must not be used as the sole basis for
                enforcement, sanctions or reduced oversight. Interface prioritization bands are research review
                categories, not confirmed latent non-compliance, regulatory guilt, or autonomous enforcement decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_calibration_protocol_notice():
    st.markdown(
        """
        <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-left: 4px solid #2563EB; border-radius: 6px; padding: 0.85rem 1rem; margin-bottom: 1.2rem;">
            <div style="font-weight: 700; color: #1E40AF; font-size: 0.88rem; display: flex; align-items: center; gap: 0.5rem;">
                <span>🔬 CALIBRATION-SELECTION PROTOCOL & PROVENANCE DISCLOSURE</span>
            </div>
            <p style="color: #1E3A8A; font-size: 0.82rem; margin: 0.4rem 0 0 0; line-height: 1.55;">
                <strong>Cohort Separation Disclosure:</strong> The original Platt-like calibrators and the ensemble operating threshold (τ = 0.04908877 | rounded 0.04909)
                were fitted and selected exclusively on the 2020–2021 validation cohort. Because this single cohort supported both calibration
                and threshold selection, validation metrics reflect <em>development-selected optimization</em> rather than unbiased post-selection performance.
                The 2022–2024 later cohort is exploratory because its outcomes were previously inspected; no independent evaluation has yet been completed.
                The interface defaults strictly to <strong>τ = 0.04908877 (0.04909)</strong> for exact reproducibility with the frozen checkpoint package manifest (eeec8023).
                Arbitrary threshold switching is disabled to prevent model–threshold mismatch.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Sidebar Navigation & System Health
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1.2rem 0;">
            <h2 style="color: #0B2545; margin: 0; font-size: 1.35rem; font-weight: 700;">⚓ TECR-Net UI</h2>
            <p style="color: #64748B; margin: 0.2rem 0 0 0; font-size: 0.82rem; font-weight: 500;">Recorded-INC Research Ranking Prototype</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Overview & Dashboard",
            "Single-Platform Research Assessment",
            "Batch Assessment",
            "Model Architecture & Limitations",
        ],
        index=0,
    )

    st.markdown("<hr style='border-color: #E2E8F0; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #475569; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;'>Artifact Health Status</h4>", unsafe_allow_html=True)

    if model_loaded:
        health = service.health_check()
        if health.get("status") == "HEALTHY":
            th_val = getattr(service, "archived_ensemble_threshold", 0.04908876769816646)
            service.set_threshold(th_val)
            manifest_id = getattr(service, "manifest_id", "eeec8023")
            seed_list = health.get("seed_ids", getattr(service, "seeds", []))

            st.markdown(
                f"""
                <div class="health-pill">
                    <span class="health-dot"></span> 7 Seeds Ready ({health.get('device', 'CPU').upper()})
                </div>
                <div style="font-size: 0.78rem; color: #475569; margin-top: 0.6rem; line-height: 1.6;">
                    <strong style="color: #0F172A;">Loaded model:</strong> Archived TECR-Net ensemble<br>
                    <strong style="color: #0F172A;">Model version:</strong> {manifest_id} (Frozen manifest)<br>
                    <strong style="color: #0F172A;">Decision threshold:</strong> {th_val:.8f} (rounded: {th_val:.5f})<br>
                    <strong style="color: #0F172A;">Threshold source:</strong> Frozen ensemble manifest<br>
                    <strong style="color: #0F172A;">Threshold modification:</strong> Disabled<br>
                    <strong style="color: #0F172A;">Input Dimensions:</strong> 26 dyn × 3 yr | 3 stat | 8 cat<br>
                    <strong style="color: #0F172A;">Seeds:</strong> {', '.join(str(s) for s in seed_list)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<hr style='border-color: #E2E8F0; margin: 1.2rem 0 0.8rem 0;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #475569; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;'>Threshold Information</h4>", unsafe_allow_html=True)

            archived_th = getattr(service, "archived_ensemble_threshold", 0.04908876769816646)
            symm_th = getattr(service, "symmetric_threshold", 0.07110931251823678)
            legacy_th = getattr(service, "legacy_seed_mean_threshold", 0.04981793852808685)

            st.markdown(
                f"""
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.75rem 0.85rem;">
                    <div style="font-size: 0.76rem; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 0.4rem;">
                        📊 Reference Thresholds (Read-Only)
                    </div>
                    <div style="font-size: 0.75rem; color: #334155; line-height: 1.6;">
                        • <strong>Active Ensemble:</strong> τ = {archived_th:.8f} ({archived_th * 100:.2f}% | rounded: 0.04909)<br>
                        • <strong>Symmetric Comparator:</strong> τ = {symm_th:.8f} ({symm_th * 100:.2f}%)<br>
                        • <strong>Legacy Seed-Mean:</strong> τ = {legacy_th:.8f} ({legacy_th * 100:.2f}%)<br>
                    </div>
                    <p style="font-size: 0.71rem; color: #64748B; margin: 0.5rem 0 0 0; line-height: 1.45;">
                        <em>Notice: Arbitrary threshold switching is disabled. Only the threshold belonging to the loaded model, calibrator, and manifest is active. Symmetric and legacy thresholds belong to different analytical constructions and cannot control classifications for this loaded ensemble.</em>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error(f"Health Check Failed: {health.get('forward_inference_check', 'Error')}")
    else:
        st.error("Model artifacts not loaded.")


# ==============================================================================
# PAGE 1: OVERVIEW & DASHBOARD
# ==============================================================================
if page == "Overview & Dashboard":
    st.markdown(
        """
        <div class="tecrnet-header">
            <h1>TECR-Net Offshore Decision-Support System</h1>
            <p>Retrospective deep-learning ensemble model producing an apparent probability-mapped score that a U.S. outer continental
            shelf platform will have a recorded Incident of Non-Compliance (INC) in observation year t+1.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_disclaimer()
    render_calibration_protocol_notice()

    if not model_loaded:
        st.error(f"### Model Initialization Error\n{load_error_message}\nPlease ensure the artifacts directory contains all 7 seed models, best_params.json, and preprocessing.json.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Ensemble Architecture", "7 TECR-Net Seeds", "Frozen Preprocessing")
    with col2:
        st.metric("Decision Threshold", f"{service.threshold * 100:.2f}%", f"τ = {service.threshold:.5f} (Frozen Manifest)")
    with col3:
        st.metric("Temporal Framing", "3-Year History", "Input: t-2,t-1,t | Target: t+1")
    with col4:
        st.metric("Feature Modalities", "37 Total Features", "26 Dyn | 3 Stat | 8 Cat")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    c_left, c_right = st.columns([1.1, 0.9])

    with c_left:
        st.markdown(
            """
            <div class="tecrnet-card">
                <h3>System Purpose & Operational Scope</h3>
                <p style="font-size: 0.92rem; color: #475569; line-height: 1.6;">
                    The <strong>TECR-Net</strong> (Temporal Cross-Modal Residual Network) prototype serves as an
                    analytical decision-support tool designed to aid regulatory researchers and inspection planning teams.
                    By processing multi-year operational data, platform structural metadata, and historical enforcement records,
                    the model generates an ensemble ranking score indicating relative priority for subsequent recorded INC review.
                </p>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                    <h4 style="font-size: 0.88rem; color: #0B2545; margin-top: 0; text-transform: uppercase;">Core Analytical Principles</h4>
                    <ul style="font-size: 0.85rem; color: #334155; margin-bottom: 0; padding-left: 1.2rem; line-height: 1.6;">
                        <li><strong>Input History (t-2, t-1, t):</strong> Exactly three consecutive calendar years of historical operating, production, and enforcement observations constitute model input.</li>
                        <li><strong>Outcome Window (t+1):</strong> The target outcome is whether a recorded INC occurs during calendar year t+1. Outcome year t+1 information is strictly excluded from input features.</li>
                        <li><strong>Frozen Inference Pipeline:</strong> Models, weights, clipping ranges, medians, and calibrators are completely frozen from manifest <code>eeec8023</code>. No dynamic re-training occurs within the interface.</li>
                        <li><strong>Target Leakage Prevention:</strong> Identification codes, complex IDs, and forward-looking outcome indicators (t+1) are strictly barred from entering feature tensors.</li>
                        <li><strong>Multi-Seed Model Variability:</strong> Predictions display the seed standard deviation across all seven seeds to communicate structural model variability, not confidence intervals.</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    tau = service.threshold
    tau_pct = tau * 100
    half_tau_pct = 0.5 * tau_pct
    double_tau_pct = 2.0 * tau_pct

    with c_right:
        st.markdown(
            f"""
            <div class="tecrnet-card">
                <h3>Heuristic Research Prioritization Bands</h3>
                <p style="font-size: 0.88rem; color: #64748B; margin-bottom: 0.8rem;">
                    Scores are categorized into relative heuristic research prioritization bands based on the active ensemble threshold (τ = {tau_pct:.2f}% · {thresh_mode_str}):
                </p>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                    <thead>
                        <tr style="border-bottom: 2px solid #E2E8F0; text-align: left; color: #0B2545;">
                            <th style="padding: 0.4rem 0.2rem;">Heuristic Research Band</th>
                            <th style="padding: 0.4rem 0.2rem;">Range Relative to Threshold</th>
                            <th style="padding: 0.4rem 0.2rem;">Current Cutoff</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom: 1px solid #F1F5F9;">
                            <td style="padding: 0.5rem 0.2rem;"><span class="badge badge-low">Low (Heuristic Research Band)</span></td>
                            <td style="padding: 0.5rem 0.2rem; color: #475569;">p &lt; 0.5 × τ</td>
                            <td style="padding: 0.5rem 0.2rem; font-family: monospace;">&lt; {half_tau_pct:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #F1F5F9;">
                            <td style="padding: 0.5rem 0.2rem;"><span class="badge badge-moderate">Moderate (Heuristic Research Band)</span></td>
                            <td style="padding: 0.5rem 0.2rem; color: #475569;">0.5 × τ ≤ p &lt; τ</td>
                            <td style="padding: 0.5rem 0.2rem; font-family: monospace;">{half_tau_pct:.2f}% - {tau_pct:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #F1F5F9;">
                            <td style="padding: 0.5rem 0.2rem;"><span class="badge badge-elevated">Elevated (Heuristic Research Band)</span></td>
                            <td style="padding: 0.5rem 0.2rem; color: #475569;">τ ≤ p &lt; 2 × τ</td>
                            <td style="padding: 0.5rem 0.2rem; font-family: monospace;">{tau_pct:.2f}% - {double_tau_pct:.2f}%</td>
                        </tr>
                        <tr>
                            <td style="padding: 0.5rem 0.2rem;"><span class="badge badge-higher">Higher (Heuristic Research Band)</span></td>
                            <td style="padding: 0.5rem 0.2rem; color: #475569;">p ≥ 2 × τ</td>
                            <td style="padding: 0.5rem 0.2rem; font-family: monospace;">≥ {double_tau_pct:.2f}%</td>
                        </tr>
                    </tbody>
                </table>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.8rem; line-height: 1.4;">
                    <em>Note: These interface categories are heuristic research bands for prioritizing analytical review and do not represent validated regulatory risk classifications.</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 2: SINGLE-PLATFORM ASSESSMENT
# ==============================================================================
elif page == "Single-Platform Research Assessment":
    st.markdown(
        """
        <div class="tecrnet-header">
            <h1>Single-Platform Research Assessment</h1>
            <p>Input consecutive three-year feature records for an individual offshore structure to compute
            apparent probability-mapped ensemble score for year t+1.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_disclaimer()

    if not model_loaded:
        st.error(f"Cannot perform assessment: {load_error_message}")
        st.stop()

    # Pre-populate defaults
    defaults = service.preprocessor.get_defaults()
    schema_vocab = service.preprocessing_config["category_vocabularies"]

    # Session state initialization for form values
    if "single_form_data" not in st.session_state:
        st.session_state.single_form_data = defaults

    # Action Toolbar
    col_btn1, col_btn2, col_info = st.columns([1.2, 1.2, 3.6])
    with col_btn1:
        if st.button("📋 Load Sample Platform Data", use_container_width=True):
            sample_json_path = Path(__file__).resolve().parent / "sample_input.json"
            if sample_json_path.exists():
                st.session_state.single_form_data = json.loads(sample_json_path.read_text(encoding="utf-8"))
                st.session_state["platform_ref_id"] = "OCS-G-04401"
                st.session_state["platform_name"] = "Galveston Block 310-A"
                st.session_state["base_year"] = 2021
                st.toast("Loaded sample platform data from sample_input.json!", icon="✅")
                st.rerun()
            else:
                st.warning("sample_input.json not found in root directory.")
    with col_btn2:
        if st.button("🔄 Reset to Training Medians", use_container_width=True):
            st.session_state.single_form_data = defaults
            st.session_state["platform_ref_id"] = "PLT-CUSTOM-01"
            st.session_state["platform_name"] = ""
            st.session_state["base_year"] = 2021
            st.toast("Reset form to training medians.", icon="ℹ️")
            st.rerun()
    with col_info:
        st.caption("Identifiers entered below are used strictly for report display and never enter neural network inference.")

    st.markdown("<hr style='border-color: #E2E8F0; margin: 1rem 0;'>", unsafe_allow_html=True)

    # Platform Identification & Temporal Consecutiveness Section
    with st.container():
        st.markdown("<div class='form-section-title'>1. Platform Metadata & Observation Years (t-2, t-1, t)</div>", unsafe_allow_html=True)
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            disp_platform_id = st.text_input("Platform Identifier (Display Only)", value=st.session_state.get("platform_ref_id", "OCS-G-04401"))
        with m_col2:
            disp_platform_name = st.text_input("Platform / Complex Name (Display Only)", value=st.session_state.get("platform_name", "Galveston Block 310-A"))
        with m_col3:
            base_year = st.number_input("Observation Start Year (t-2)", min_value=1950, max_value=2050, value=st.session_state.get("base_year", 2021), step=1)
        with m_col4:
            year_t_minus_2 = int(base_year)
            year_t_minus_1 = int(base_year + 1)
            year_t = int(base_year + 2)
            year_target = int(base_year + 3)
            st.markdown(
                f"""
                <div style="background: #F1F5F9; border-radius: 6px; padding: 0.45rem 0.75rem; margin-top: 1.55rem; font-size: 0.83rem;">
                    <strong>Consecutive Sequence:</strong> {year_t_minus_2} → {year_t_minus_1} → {year_t} (Target: <strong>{year_target}</strong>)
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Dynamic Variables Tabs (t-2, t-1, t)
    st.markdown("<div class='form-section-title' style='margin-top: 1.5rem;'>2. Dynamic Operational & Inspection Features (26 Features × 3 Years)</div>", unsafe_allow_html=True)

    cur_years = st.session_state.single_form_data.get("years", defaults["years"])
    while len(cur_years) < 3:
        cur_years.append(defaults["years"][0].copy())

    year_tabs = st.tabs([
        f"📅 Year t-2 ({year_t_minus_2})",
        f"📅 Year t-1 ({year_t_minus_1})",
        f"📅 Year t ({year_t}) [Most Recent]",
    ])

    user_years_data: List[Dict[str, float]] = []

    for idx, (tab, y_label, y_val) in enumerate(zip(year_tabs, ["t-2", "t-1", "t"], [year_t_minus_2, year_t_minus_1, year_t])):
        with tab:
            st.caption(f"Enter dynamic operational, production, and compliance records for calendar year {y_val} ({y_label}). Missing fields will be imputed with frozen training medians.")
            y_dict = cur_years[idx]
            collected_year: Dict[str, float] = {}

            g1, g2, g3 = st.columns(3)

            with g1:
                st.markdown("<strong>Production & Wells</strong>", unsafe_allow_html=True)
                collected_year["active_months"] = st.number_input(f"active_months", 0.0, 12.0, float(y_dict.get("active_months", 12.0)), key=f"am_{idx}")
                collected_year["producing_wells_mean"] = st.number_input(f"producing_wells_mean", 0.0, 100.0, float(y_dict.get("producing_wells_mean", 1.0)), key=f"pwm_{idx}")
                collected_year["producing_wells_max"] = st.number_input(f"producing_wells_max", 0.0, 100.0, float(y_dict.get("producing_wells_max", 1.0)), key=f"pwx_{idx}")
                collected_year["bopd_mean"] = st.number_input(f"bopd_mean (Oil bbl/day)", 0.0, 100000.0, float(y_dict.get("bopd_mean", 31.0)), key=f"bom_{idx}")
                collected_year["bopd_max"] = st.number_input(f"bopd_max", 0.0, 100000.0, float(y_dict.get("bopd_max", 64.0)), key=f"box_{idx}")
                collected_year["mcfd_mean"] = st.number_input(f"mcfd_mean (Gas mcf/day)", 0.0, 200000.0, float(y_dict.get("mcfd_mean", 151.2)), key=f"mcm_{idx}")
                collected_year["mcfd_max"] = st.number_input(f"mcfd_max", 0.0, 200000.0, float(y_dict.get("mcfd_max", 301.5)), key=f"mcx_{idx}")
                collected_year["boepd_mean"] = st.number_input(f"boepd_mean", 0.0, 100000.0, float(y_dict.get("boepd_mean", 116.5)), key=f"bem_{idx}")
                collected_year["bwpd_mean"] = st.number_input(f"bwpd_mean (Water bbl/day)", 0.0, 100000.0, float(y_dict.get("bwpd_mean", 99.0)), key=f"bwm_{idx}")
                collected_year["platform_age"] = st.number_input(f"platform_age (years)", 0.0, 100.0, float(y_dict.get("platform_age", 28.0)), key=f"pa_{idx}")

            with g2:
                st.markdown("<strong>Operator Inspections & Incidents</strong>", unsafe_allow_html=True)
                collected_year["operator_facilities_inspected"] = st.number_input(f"operator_facilities_inspected", 0.0, 2000.0, float(y_dict.get("operator_facilities_inspected", 53.0)), key=f"ofi_{idx}")
                collected_year["operator_inspection_records"] = st.number_input(f"operator_inspection_records", 0.0, 2000.0, float(y_dict.get("operator_inspection_records", 53.0)), key=f"oir_{idx}")
                collected_year["operator_inspection_types"] = st.number_input(f"operator_inspection_types", 0.0, 3000.0, float(y_dict.get("operator_inspection_types", 97.0)), key=f"oit_{idx}")
                collected_year["investigated_accident_count"] = st.number_input(f"investigated_accident_count", 0.0, 50.0, float(y_dict.get("investigated_accident_count", 0.0)), key=f"iac_{idx}")
                collected_year["accident_type_count"] = st.number_input(f"accident_type_count", 0.0, 50.0, float(y_dict.get("accident_type_count", 0.0)), key=f"atc_{idx}")
                collected_year["previous_year_accident_count"] = st.number_input(f"previous_year_accident_count", 0.0, 50.0, float(y_dict.get("previous_year_accident_count", 0.0)), key=f"pyac_{idx}")
                collected_year["production_change_bopd"] = st.number_input(f"production_change_bopd", -10.0, 50.0, float(y_dict.get("production_change_bopd", -0.15)), key=f"pcb_{idx}")
                collected_year["production_change_mcfd"] = st.number_input(f"production_change_mcfd", -10.0, 50.0, float(y_dict.get("production_change_mcfd", -0.16)), key=f"pcm_{idx}")

            with g3:
                st.markdown("<strong>Recorded Non-Compliance (INCs)</strong>", unsafe_allow_html=True)
                collected_year["inc_count"] = st.number_input(f"inc_count (Year {y_label})", 0.0, 50.0, float(y_dict.get("inc_count", 0.0)), key=f"inc_{idx}")
                collected_year["enforcement_code_count"] = st.number_input(f"enforcement_code_count", 0.0, 50.0, float(y_dict.get("enforcement_code_count", 0.0)), key=f"ecc_{idx}")
                collected_year["any_inc"] = st.selectbox(f"any_inc flag", [0.0, 1.0], index=1 if float(y_dict.get("any_inc", 0.0)) == 1.0 else 0, key=f"ai_{idx}")
                collected_year["previous_year_inc_count"] = st.number_input(f"previous_year_inc_count", 0.0, 50.0, float(y_dict.get("previous_year_inc_count", 0.0)), key=f"pyic_{idx}")
                collected_year["inc_count_previous_3_observations"] = st.number_input(f"inc_count_previous_3_obs", 0.0, 100.0, float(y_dict.get("inc_count_previous_3_observations", 0.0)), key=f"icp3_{idx}")
                collected_year["operator_warning_incs"] = st.number_input(f"operator_warning_incs", 0.0, 500.0, float(y_dict.get("operator_warning_incs", 2.0)), key=f"owi_{idx}")
                collected_year["operator_component_shutin_incs"] = st.number_input(f"operator_component_shutin_incs", 0.0, 500.0, float(y_dict.get("operator_component_shutin_incs", 1.0)), key=f"ocsi_{idx}")
                collected_year["operator_facility_shutin_incs"] = st.number_input(f"operator_facility_shutin_incs", 0.0, 100.0, float(y_dict.get("operator_facility_shutin_incs", 0.0)), key=f"ofsi_{idx}")

            user_years_data.append(collected_year)

    # Static Numerical & Categorical Attributes
    st.markdown("<div class='form-section-title' style='margin-top: 1.5rem;'>3. Static Numerical & Structural Categorical Attributes</div>", unsafe_allow_html=True)

    col_stat, col_cat = st.columns([1, 1.3])

    cur_static = st.session_state.single_form_data.get("static", defaults["static"])
    cur_cat = st.session_state.single_form_data.get("categorical", defaults["categorical"])

    with col_stat:
        st.markdown("<div class='tecrnet-card' style='height: 100%;'><h4>Static Numerical Attributes</h4>", unsafe_allow_html=True)
        stat_water_depth = st.number_input("water_depth (ft)", 0.0, 10000.0, float(cur_static.get("water_depth", 62.0)))
        stat_latitude = st.number_input("latitude (NAD27 / WGS84)", 15.0, 45.0, float(cur_static.get("latitude", 28.955716)), format="%.6f")
        stat_longitude = st.number_input("longitude", -120.0, -70.0, float(cur_static.get("longitude", -91.01020194)), format="%.6f")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_cat:
        st.markdown("<div class='tecrnet-card' style='height: 100%;'><h4>Categorical Variables (Vocabulary Encoded)</h4>", unsafe_allow_html=True)
        cat_c1, cat_c2 = st.columns(2)

        with cat_c1:
            area_options = list(schema_vocab["area_code"].keys())
            cur_area = str(cur_cat.get("area_code", "AC"))
            cat_area = st.selectbox("area_code", area_options, index=area_options.index(cur_area) if cur_area in area_options else 0)

            region_options = list(schema_vocab["region_code"].keys())
            cur_reg = str(cur_cat.get("region_code", "G"))
            cat_region = st.selectbox("region_code", region_options, index=region_options.index(cur_reg) if cur_reg in region_options else 0)

            struct_options = list(schema_vocab["structure_type"].keys())
            cur_struct = str(cur_cat.get("structure_type", "CAIS"))
            cat_struct = st.selectbox("structure_type", struct_options, index=struct_options.index(cur_struct) if cur_struct in struct_options else 0)

            dist_options = list(schema_vocab["district_code"].keys())
            cur_dist = str(cur_cat.get("district_code", "1.0"))
            cat_district = st.selectbox("district_code", dist_options, index=dist_options.index(cur_dist) if cur_dist in dist_options else 0)

        with cat_c2:
            maj_options = list(schema_vocab["major_structure_flag"].keys())
            cur_maj = str(cur_cat.get("major_structure_flag", "N"))
            cat_major = st.selectbox("major_structure_flag", maj_options, index=maj_options.index(cur_maj) if cur_maj in maj_options else 0)

            heli_options = list(schema_vocab["heliport_flag"].keys())
            cur_heli = str(cur_cat.get("heliport_flag", "N"))
            cat_heliport = st.selectbox("heliport_flag", heli_options, index=heli_options.index(cur_heli) if cur_heli in heli_options else 0)

            att_options = list(schema_vocab["attended_8hr_flag"].keys())
            cur_att = str(cur_cat.get("attended_8hr_flag", "N"))
            cat_attended = st.selectbox("attended_8hr_flag", att_options, index=att_options.index(cur_att) if cur_att in att_options else 0)

            man_options = list(schema_vocab["manned_24hr_flag"].keys())
            cur_man = str(cur_cat.get("manned_24hr_flag", "N"))
            cat_manned = st.selectbox("manned_24hr_flag", man_options, index=man_options.index(cur_man) if cur_man in man_options else 0)

        st.markdown("</div>", unsafe_allow_html=True)

    # Submission Action
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    if st.button("🚀 Run TECR-Net Assessment", type="primary", use_container_width=True):
        payload = {
            "years": user_years_data,
            "static": {
                "water_depth": stat_water_depth,
                "latitude": stat_latitude,
                "longitude": stat_longitude,
            },
            "categorical": {
                "area_code": cat_area,
                "region_code": cat_region,
                "structure_type": cat_struct,
                "major_structure_flag": cat_major,
                "district_code": cat_district,
                "heliport_flag": cat_heliport,
                "attended_8hr_flag": cat_attended,
                "manned_24hr_flag": cat_manned,
            },
        }

        with st.spinner("Executing 7-seed multihead cross-modal neural inference..."):
            try:
                res = service.predict_single(payload)
                st.session_state["last_single_result"] = res
                st.session_state["last_single_meta"] = {
                    "platform_id": disp_platform_id,
                    "platform_name": disp_platform_name,
                    "window": f"{year_t_minus_2} - {year_t}",
                    "target_year": year_target,
                }
            except Exception as e:
                st.error(f"Inference Failure: {e}")
                logger.error(f"Prediction failed: {e}", exc_info=True)

    # Render Results Section if available
    if "last_single_result" in st.session_state:
        res = st.session_state["last_single_result"]
        meta = st.session_state["last_single_meta"]

        st.markdown("<hr style='border-color: #007A78; border-width: 2px; margin: 2rem 0;'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 1rem;">
                <div>
                    <h2 style="color: #0B2545; margin: 0; font-size: 1.55rem; font-weight: 700;">Research Assessment Output: {meta['platform_id']}</h2>
                    <p style="color: #64748B; margin: 0.2rem 0 0 0; font-size: 0.9rem;">
                        {meta['platform_name']} | Sequence {meta['window']} | Recorded-INC outcome year: <strong>{meta['target_year']}</strong>
                    </p>
                </div>
                <div>
                    <span class="badge { 'badge-low' if 'Low' in res['prioritization_band'] else ('badge-moderate' if 'Moderate' in res['prioritization_band'] else ('badge-elevated' if 'Elevated' in res['prioritization_band'] else 'badge-higher')) }" style="font-size: 0.95rem; padding: 0.45rem 1rem;">
                        Heuristic Research Band: {res['prioritization_band']}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Primary KPI Metric Cards
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.metric(
                "Ensemble Ranking Score",
                f"{res['ensemble_percentage']:.2f}%",
                f"{res['ensemble_probability']:.4f}",
            )
        with kpi2:
            st.metric(
                "Decision Threshold",
                f"{res['saved_threshold_percentage']:.2f}%",
                f"τ = {res['saved_threshold']:.5f} ({res.get('threshold_mode', 'Active')})",
            )
        with kpi3:
            st.metric(
                "Threshold Comparison",
                res["status_label"],
                f"{res['risk_ratio_to_threshold']:.2f}× threshold" if res["risk_ratio_to_threshold"] else "N/A",
            )
        with kpi4:
            st.metric(
                "Seed Variability (SD)",
                f"± {res['seed_sd'] * 100:.3f}%",
                "Model variability",
                help="Standard deviation across 7 seed checkpoints. Represents architectural variability, NOT a statistical confidence interval.",
            )
        with kpi5:
            st.metric(
                "Heuristic Research Band",
                res["prioritization_band"],
                res["prioritization_band_desc"],
                help="Heuristic research prioritization band based on ratio to saved decision threshold. Not an enforcement decision or legal non-compliance determination.",
            )

        # 7-Seed Probability Chart & 3-Year Trend Chart
        c_seed, c_trend = st.columns([1, 1.2])

        with c_seed:
            st.markdown("<div class='tecrnet-card'><h3>Seven-Seed Ensemble Distribution</h3>", unsafe_allow_html=True)
            seeds_df = pd.DataFrame(res["seed_predictions"])
            seeds_df["Seed Label"] = [f"Seed {s}" for s in seeds_df["seed"]]

            fig_seed = go.Figure()
            fig_seed.add_trace(go.Bar(
                x=seeds_df["Seed Label"],
                y=seeds_df["calibrated_percentage"],
                marker_color="#0E7490",
                text=[f"{v:.2f}%" for v in seeds_df["calibrated_percentage"]],
                textposition="auto",
                name="Calibrated Seed Risk",
            ))

            fig_seed.add_hline(
                y=res["ensemble_percentage"],
                line_dash="solid",
                line_color="#0B2545",
                line_width=2,
                annotation_text=f"Ensemble Mean ({res['ensemble_percentage']:.2f}%)",
                annotation_position="top left",
            )
            fig_seed.add_hline(
                y=res["saved_threshold_percentage"],
                line_dash="dash",
                line_color="#D97724",
                line_width=2,
                annotation_text=f"Decision Threshold ({res['saved_threshold_percentage']:.2f}%)",
                annotation_position="bottom right",
            )

            fig_seed.update_layout(
                yaxis_title="Apparent probability-mapped score (%)",
                template="plotly_white",
                margin=dict(l=20, r=20, t=30, b=20),
                height=320,
            )
            st.plotly_chart(fig_seed, use_container_width=True)
            st.caption("Seed scores transformed using the archived Platt-like probability mapping (fitted on probability inputs). Dotted line indicates frozen ensemble threshold.")
            st.markdown("</div>", unsafe_allow_html=True)

        with c_trend:
            st.markdown("<div class='tecrnet-card'><h3>Three-Year Operational & Compliance Trend</h3>", unsafe_allow_html=True)
            # Compile trend data from user input
            trend_records = []
            year_labels_plot = [f"t-2 ({meta['window'].split(' - ')[0]})", f"t-1", f"t ({meta['window'].split(' - ')[1]})"]
            for y_label, y_data in zip(year_labels_plot, user_years_data):
                trend_records.append({
                    "Year": y_label,
                    "BOPD (Mean)": y_data.get("bopd_mean", 0),
                    "MCFD (Mean)": y_data.get("mcfd_mean", 0),
                    "Past INCs": y_data.get("inc_count", 0),
                    "Facilities Inspected": y_data.get("operator_facilities_inspected", 0),
                })
            trend_df = pd.DataFrame(trend_records)

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=trend_df["Year"], y=trend_df["Past INCs"],
                name="INC Count", mode="lines+markers", line=dict(color="#B91C1C", width=3), yaxis="y1",
            ))
            fig_trend.add_trace(go.Scatter(
                x=trend_df["Year"], y=trend_df["BOPD (Mean)"],
                name="BOPD Mean", mode="lines+markers", line=dict(color="#0284C7", width=2), yaxis="y2",
            ))
            fig_trend.add_trace(go.Scatter(
                x=trend_df["Year"], y=trend_df["MCFD (Mean)"],
                name="MCFD Mean", mode="lines+markers", line=dict(color="#059669", width=2, dash="dot"), yaxis="y2",
            ))

            fig_trend.update_layout(
                yaxis=dict(title="INC Count", side="left", showgrid=False),
                yaxis2=dict(title="Production Volume", side="right", overlaying="y", showgrid=True),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_white",
                margin=dict(l=20, r=20, t=30, b=20),
                height=320,
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            st.caption("Temporal trajectory across evaluation years preceding predicted year.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Cross-Modal & Temporal Attention Visualizations
        st.markdown(
            """
            <div class='tecrnet-card'>
                <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; margin-bottom: 0.3rem;">
                    <h3 style="margin: 0;">Model Diagnostic Associations & Attention Weights</h3>
                    <span style="font-size: 0.78rem; font-weight: 600; color: #475569; background: #F1F5F9; padding: 0.2rem 0.6rem; border-radius: 4px; border: 1px solid #E2E8F0;">
                        Model Associations · Not Causal Explanations
                    </span>
                </div>
                <p style="font-size: 0.85rem; color: #475569; margin: 0 0 1rem 0; line-height: 1.5;">
                    <em>Methodological Note: Attention distributions and dynamic feature gates reflect internal computational associations learned within the neural network architecture. They describe relative mathematical weighting across modalities and sequence steps—they do <strong>not</strong> constitute causal explanations, physical mechanisms, root-cause analyses, or real-world importance factors.</em>
                </p>
            """,
            unsafe_allow_html=True,
        )
        att_c1, att_c2, att_c3 = st.columns(3)

        with att_c1:
            st.markdown("<strong>Temporal Attention (Model Association)</strong>", unsafe_allow_html=True)
            temp_df = pd.DataFrame(list(res["temporal_attention"].items()), columns=["Step", "Weight"])
            fig_ta = px.pie(temp_df, names="Step", values="Weight", color_discrete_sequence=["#0B2545", "#0E7490", "#D97724"], hole=0.45)
            fig_ta.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=220, showlegend=True)
            st.plotly_chart(fig_ta, use_container_width=True)
            st.caption("Recurrent step association weights (internal sequence pooling; not causal drivers).")

        with att_c2:
            st.markdown("<strong>Cross-Modal Attention (Model Association)</strong>", unsafe_allow_html=True)
            mod_df = pd.DataFrame(list(res["modality_attention"].items()), columns=["Modality", "Attention Weight"])
            fig_ma = px.bar(mod_df, x="Attention Weight", y="Modality", orientation="h", color="Modality", color_discrete_sequence=["#007A78", "#1E3A5F", "#E76F51"])
            fig_ma.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=220, showlegend=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_ma, use_container_width=True)
            st.caption("Modality token associations (internal architectural weighting; not causal importance).")

        with att_c3:
            st.markdown("<strong>Top Dynamic Feature Gates (Model Association)</strong>", unsafe_allow_html=True)
            gate_df = pd.DataFrame(res["top_feature_gates"][:5])
            fig_gate = px.bar(gate_df, x="weight", y="feature", orientation="h", color_discrete_sequence=["#0B2545"])
            fig_gate.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=220, yaxis=dict(autorange="reversed"), xaxis_title="Learned Gate Weight")
            st.plotly_chart(fig_gate, use_container_width=True)
            st.caption("Learned input gating activations (internal dynamic feature scaling; not causal explanation).")

        st.markdown("</div>", unsafe_allow_html=True)

        # Preprocessing & Missing Data Audit Log
        st.markdown("<div class='tecrnet-card'><h3>Preprocessing & Imputation Audit Trail</h3>", unsafe_allow_html=True)
        imputed_list = res.get("imputed_features", [])
        unknown_cat_list = res.get("unknown_categories", [])

        if not imputed_list and not unknown_cat_list:
            st.success("✅ Input completeness check passed: All 26×3 dynamic variables, 3 static values, and 8 categorical attributes were valid and within vocabularies. Zero median imputations required.")
        else:
            if imputed_list:
                st.warning(f"⚠️ {len(imputed_list)} Missing Field(s) Imputed: Frozen training medians were used for the following inputs:")
                st.dataframe(pd.DataFrame(imputed_list), use_container_width=True)
            if unknown_cat_list:
                st.info(f"ℹ️ {len(unknown_cat_list)} Unseen Categorical Value(s): Mapped to code 0 (unknown/padding token):")
                st.dataframe(pd.DataFrame(unknown_cat_list), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Export Prediction Record
        rec_json = json.dumps({
            "metadata": meta,
            "prediction_results": {
                "ensemble_probability": res["ensemble_probability"],
                "decision_threshold": res["saved_threshold"],
                "above_threshold": res["is_above_threshold"],
                "prioritization_band": res["prioritization_band"],
                "seed_variability_sd": res["seed_sd"],
                "seed_predictions": res["seed_predictions"],
            },
            "temporal_attention": res["temporal_attention"],
            "modality_attention": res["modality_attention"],
            "imputed_features": res["imputed_features"],
            "disclaimer": res["regulatory_disclaimer"],
        }, indent=2)

        st.download_button(
            label="💾 Download Prediction Record (JSON)",
            data=rec_json,
            file_name=f"tecrnet_prediction_{meta['platform_id']}_{meta['target_year']}.json",
            mime="application/json",
        )


# ==============================================================================
# PAGE 3: BATCH ASSESSMENT
# ==============================================================================
elif page == "Batch Assessment":
    st.markdown(
        """
        <div class="tecrnet-header">
            <h1>Batch Platform Risk Assessment</h1>
            <p>Upload multi-platform CSV records. The engine validates column schemas, extracts strictly consecutive
            three-year sequences (t-2, t-1, t), reports excluded rows, and performs ensemble risk scoring.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_disclaimer()

    if not model_loaded:
        st.error(f"Cannot perform batch assessment: {load_error_message}")
        st.stop()

    b_top1, b_top2 = st.columns([1.5, 3.5])
    with b_top1:
        # Provide sample CSV download button
        sample_csv_path = Path(__file__).resolve().parent / "sample_input.csv"
        if sample_csv_path.exists():
            st.download_button(
                label="📥 Download Sample Batch CSV",
                data=sample_csv_path.read_text(encoding="utf-8"),
                file_name="sample_input.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with b_top2:
        st.caption("Upload a CSV containing `platform_id`, `year`, 26 dynamic features, 3 static features, and 8 categorical attributes.")

    uploaded_file = st.file_uploader("Upload Offshore Platforms CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! Loaded {len(raw_df)} rows and {len(raw_df.columns)} columns.")

            # Validate batch dataframe and construct exact 3-year sequences
            with st.spinner("Validating schema and identifying consecutive 3-year sequences..."):
                valid_seqs, excluded_records = validate_batch_dataframe(raw_df, service.preprocessor)

            # Display Validation Summary Cards
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            vb1, vb2, vb3 = st.columns(3)
            with vb1:
                st.metric("Total Rows Uploaded", len(raw_df))
            with vb2:
                st.metric("Valid 3-Year Sequences", len(valid_seqs), "Ready for inference")
            with vb3:
                st.metric("Excluded Records / Platforms", len(excluded_records), "Filtered with audit reason")

            # Exclusion Report Expander
            if excluded_records:
                with st.expander(f"⚠️ View {len(excluded_records)} Excluded Record(s) and Reasons", expanded=(len(valid_seqs) == 0)):
                    st.markdown("<p style='font-size: 0.88rem; color: #64748B;'>The following records could not form valid, strictly consecutive 3-year observation triplets (t-2, t-1, t):</p>", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(excluded_records), use_container_width=True)

            if not valid_seqs:
                st.warning("No valid consecutive 3-year sequences were found. Please inspect the exclusion report above.")
            else:
                st.markdown("<hr style='border-color: #E2E8F0; margin: 1.5rem 0;'>", unsafe_allow_html=True)
                if st.button(f"⚡ Execute Batch Inference on {len(valid_seqs)} Valid Sequences", type="primary", use_container_width=True):
                    with st.spinner(f"Running 7-seed ensemble inference across {len(valid_seqs)} sequences..."):
                        batch_results_df = service.predict_batch(valid_seqs)
                        st.session_state["batch_results"] = batch_results_df

                if "batch_results" in st.session_state:
                    res_df = st.session_state["batch_results"]

                    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
                    st.markdown("<h3>Batch Research Assessment Output</h3>", unsafe_allow_html=True)

                    # Summary Metrics of Batch
                    num_above = (res_df["Threshold Status"] == "Above Saved Threshold").sum()
                    pct_above = (num_above / len(res_df)) * 100 if len(res_df) > 0 else 0
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Sequences Assessed", len(res_df))
                    with m2:
                        st.metric(f"Above Threshold (τ = {service.threshold * 100:.2f}%)", f"{num_above} ({pct_above:.1f}%)")
                    with m3:
                        st.metric("Mean Predicted Risk", f"{res_df['Risk (%)'].mean():.2f}%")
                    with m4:
                        st.metric("Max Predicted Risk", f"{res_df['Risk (%)'].max():.2f}%")

                    # Filter by heuristic research band
                    band_col = "Heuristic Research Band" if "Heuristic Research Band" in res_df.columns else "Research Prioritization Band"
                    bands_available = ["All"] + sorted(res_df[band_col].unique().tolist())
                    selected_band = st.selectbox("Filter by Heuristic Research Band", bands_available)

                    filtered_df = res_df if selected_band == "All" else res_df[res_df[band_col] == selected_band]

                    st.dataframe(filtered_df, use_container_width=True)

                    # Batch Distribution Chart
                    fig_dist = px.histogram(
                        res_df,
                        x="Risk (%)",
                        color=band_col,
                        color_discrete_map={
                            "Low": "#10B981",
                            "Moderate": "#0284C7",
                            "Elevated": "#D97706",
                            "Higher": "#DC2626",
                            "Low (Heuristic)": "#10B981",
                            "Moderate (Heuristic)": "#0284C7",
                            "Elevated (Heuristic)": "#D97706",
                            "Higher (Heuristic)": "#DC2626",
                        },
                        nbins=20,
                        title="Distribution of Batch Predicted INC Risk (%)",
                    )
                    fig_dist.add_vline(
                        x=service.threshold * 100,
                        line_dash="dash",
                        line_color="#0B2545",
                        annotation_text=f"Decision Threshold ({service.threshold * 100:.2f}%)",
                    )
                    fig_dist.update_layout(template="plotly_white", height=320)
                    st.plotly_chart(fig_dist, use_container_width=True)

                    # Download Results CSV
                    csv_buffer = io.StringIO()
                    res_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Batch Results (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name="tecrnet_batch_predictions.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"Error processing CSV: {e}")
            logger.error(f"Batch processing failure: {e}", exc_info=True)


# ==============================================================================
# PAGE 4: MODEL INFORMATION & LIMITATIONS
# ==============================================================================
elif page == "Model Architecture & Limitations":
    st.markdown(
        """
        <div class="tecrnet-header">
            <h1>Model Architecture, Calibration & Limitations</h1>
            <p>Comprehensive technical specification of the Temporal Cross-Modal Residual Network (TECR-Net),
            ensemble configuration, and critical operational constraints.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_disclaimer()

    tab_arch, tab_cal, tab_limits = st.tabs([
        "🏗️ Network Architecture",
        "⚖️ Calibration & Thresholding",
        "⚠️ Methodological Limitations",
    ])

    with tab_arch:
        st.markdown(
            r"""
            <div class="tecrnet-card">
                <h3>TECR-Net Neural Architecture</h3>
                <p style="font-size: 0.92rem; color: #475569; line-height: 1.6;">
                    TECR-Net is engineered to synthesize multi-rate temporal dynamics, static geomechanical/location data,
                    and discrete facility categorical flags into a unified representation for non-compliance risk estimation.
                </p>
                <div style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
                    <ol>
                        <li><strong>Three-Year Temporal Input:</strong> Models platform behavior over consecutive feature years $t-2, t-1, t$ using 26 dynamic operational and compliance features.</li>
                        <li><strong>Feature Gating Mechanism (Learned Association):</strong> Learned per-feature gate $\sigma(\mathbf{w}_{\text{gate}})$ modulates the dynamic input tensor within the neural network:
                            <br><code>gated = dynamic * sigmoid(feature_logits)</code>
                            <br><em>(Represents internal dynamic feature weighting within the model—not real-world causal attribution).</em>
                        </li>
                        <li><strong>Bidirectional GRU:</strong> Encodes dynamic sequences with a bidirectional Gated Recurrent Unit (hidden size = 80 per direction) yielding temporal representations:
                            <br><code>seq, _ = GRU(gated)</code>
                        </li>
                        <li><strong>Temporal Attention Pooling (Learned Association):</strong> Projects recurrent states through a non-linear scoring network:
                            <br><code>score = Linear(Tanh(Linear(seq)))</code>, normalized via Softmax across the 3 temporal steps to compute a weighted temporal summary.
                            <br><em>(Represents sequence step weighting within the GRU—not a causal explanation of temporal drivers).</em>
                        </li>
                        <li><strong>Static Numerical Encoder:</strong> Multilayer perceptron with GELU activations and LayerNorm projecting 3 static continuous variables (water depth, latitude, longitude) into the 80-dimensional hidden space.</li>
                        <li><strong>Categorical Entity Embeddings:</strong> Discrete structural and administrative flags are embedded with dimension $d = \min(16, \max(3, \text{round}(\sqrt{c})))$, concatenated, and projected through a dense LayerNorm block.</li>
                        <li><strong>Cross-Modal Multi-Head Attention (Learned Association):</strong> Queries the stacked modality tokens <code>[Temporal, Static, Categorical]</code> using a learned query vector and 8 attention heads.
                            <br><em>(Learned internal architectural token weighting—not real-world factor importance).</em>
                        </li>
                        <li><strong>Residual Multi-Modal Fusion:</strong> Concatenates the attended multimodal summary with the unweighted mean modality representation, feeding into a 2-layer MLP classification head.</li>
                    </ol>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_cal:
        st.markdown(
            f"""
            <div class="tecrnet-card">
                <h3>Ensemble Formulation & Probability Calibration Protocol</h3>
                <p style="font-size: 0.92rem; color: #475569; line-height: 1.6;">
                    To maximize generalization across offshore operational regimes, predictions are generated by an
                    ensemble of <strong>seven independently trained TECR-Net seed models</strong> (Seeds 11, 22, 33, 44, 55, 66, 77).
                </p>
                <div style="background-color: #FEF3C7; border: 1px solid #FCD34D; border-left: 4px solid #D97706; padding: 0.85rem 1rem; border-radius: 6px; margin: 1rem 0;">
                    <div style="font-weight: 700; color: #92400E; font-size: 0.88rem;">
                        ⚠️ Calibration-Selection Protocol & Cohort Separation Disclosure
                    </div>
                    <p style="color: #78350F; font-size: 0.84rem; margin: 0.4rem 0 0 0; line-height: 1.55;">
                        The original Platt-like calibrators and the ensemble operating threshold (τ = 0.04908877 | rounded 0.04909) were fitted and selected
                        using the <strong>2020–2021 validation cohort</strong>. Because the same cohort was used for fitting calibration
                        slopes, tuning stopping points, and selecting the optimal F1 threshold, reported validation results
                        represent <strong>development-selected optimization</strong> rather than unbiased out-of-sample performance.
                        The <strong>2022–2024 later cohort is exploratory because its outcomes were previously inspected; no independent evaluation has yet been completed.</strong>
                    </p>
                </div>
                <div style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
                    <ul>
                        <li><strong>Seed scores transformed using the archived Platt-like probability mapping:</strong> Raw neural network probability outputs are transformed using pre-fit logistic calibrators (fitted on probability inputs, not conventional logit-input Platt scaling):
                            <br><code>P_calibrated = 1 / (1 + exp(-(coef * P_raw + intercept)))</code>
                            <br>Calibrators are completely frozen from checkpoint metadata.
                        </li>
                        <li><strong>Ensemble Pooling:</strong> The final predicted probability is the exact arithmetic mean of the 7 calibrated seed outputs.</li>
                        <li><strong>Reconciliation of Decision Thresholds:</strong>
                            <ul>
                                <li><strong>Archived Ensemble F1 (τ = {service.archived_ensemble_threshold:.8f} | {service.archived_ensemble_threshold * 100:.2f}% | rounded: 0.04909) [Active Locked Threshold]:</strong> The maximum validation F1 threshold selected on the pooled 7-seed ensemble from manifest <code>eeec8023</code>. This active threshold is locked to restore exact reproducibility with the frozen checkpoint metrics.</li>
                                <li><strong>Symmetric Comparator (τ = {service.symmetric_threshold:.8f} | {service.symmetric_threshold * 100:.2f}%):</strong> The sequence-aligned benchmark threshold derived under symmetric baseline evaluation protocols (for read-only comparison).</li>
                                <li><strong>Legacy Seed-Mean (τ = {service.legacy_seed_mean_threshold:.8f} | {service.legacy_seed_mean_threshold * 100:.2f}%):</strong> The unweighted arithmetic average of individual seed checkpoint thresholds (for historical comparison).</li>
                            </ul>
                        </li>
                        <li><strong>Model Variability (Seed SD):</strong> Disagreements among seeds are reported as the sample standard deviation. It reflects epistemic model variability across training seeds and must <em>not</em> be confused with frequentist confidence intervals.</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_limits:
        st.markdown(
            """
            <div class="tecrnet-card">
                <h3>Known Limitations & Responsible Use</h3>
                <div style="background-color: #FEF2F2; border-left: 4px solid #B91C1C; padding: 1rem; border-radius: 6px; margin-bottom: 1.2rem;">
                    <h4 style="color: #991B1B; margin-top: 0; font-size: 0.92rem; text-transform: uppercase;">Essential Operational Constraint</h4>
                    <p style="color: #7F1D1D; font-size: 0.88rem; margin: 0; line-height: 1.5;">
                        TECR-Net is strictly a retrospective decision-support prototype. It has not undergone prospective
                        validation in active operational environments. It must never be used to make autonomous regulatory determinations,
                        issue notices of incident, assess financial civil penalties, or justify reduced safety oversight.
                    </p>
                </div>
                <div style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
                    <h4>Key Methodological Factors</h4>
                    <ul>
                        <li><strong>Interpretability as Model Associations:</strong> Attention weights, temporal pooling distributions, and dynamic feature gates describe internal mathematical associations within the neural network. They must <em>never</em> be interpreted as causal explanations, root causes, or real-world importance factors.</li>
                        <li><strong>Recorded INC Bias vs. Latent Non-Compliance:</strong> The target variable reflects <em>recorded</em> administrative incidents of non-compliance. These records naturally correlate with inspection frequency, inspector specialization, and operator administrative reporting practices. Absence of a recorded INC does not guarantee absence of safety or environmental deficiencies.</li>
                        <li><strong>Historical Feature Availability & Survival Bias:</strong> Models require three continuous years of historical production and inspection records. Platforms decommissioned or shut-in during the temporal observation window are systematically excluded, introducing survivorship selection effects.</li>
                        <li><strong>Regional & Structural Differences:</strong> Most training data originates from the Gulf of Mexico (Region G). Applications to Pacific (P) or Alaska (Y) regions operate under lower historical sample representation.</li>
                        <li><strong>Heuristic Research Bands:</strong> Prioritization bands (Low, Moderate, Elevated, Higher) are heuristic research review tiers intended solely to assist human triage. They do not constitute statutory risk levels.</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Global Footer
st.markdown(
    """
    <div style="text-align: center; color: #94A3B8; font-size: 0.78rem; margin-top: 3.5rem; padding-top: 1rem; border-top: 1px solid #E2E8F0;">
        TECR-Net Decision-Support Research System | Bureau of Safety & Environmental Enforcement (BSEE) Research Sandbox | Python 3.10+ & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
