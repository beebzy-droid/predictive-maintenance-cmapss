"""
Streamlit dashboard — deployed on Hugging Face Spaces.

This is the slim, XGBoost-only version of the local app/dashboard.py.
The full local version (with LSTM live inference + FastAPI backend) is in the
project's GitHub repo at https://github.com/beebzy-droid/predictive-maintenance-cmapss

Differences from local version:
- XGBoost only (no TensorFlow / LSTM in this deployment)
- In-process inference (no FastAPI HTTP layer)
- Comparison tab shows pre-computed LSTM results from Week 5, not live predictions
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from inference import InferenceEngine

# ---- Configuration ----

PROJECT_DIR = Path(__file__).resolve().parent
DEMO_DATA_PATH = PROJECT_DIR / "example_data" / "demo_engines.json"

# Sensor name lookup
SENSOR_NAMES = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]


# ---- Page config ----

st.set_page_config(
    page_title="C-MAPSS RUL Prediction Demo",
    page_icon="🛠️",
    layout="wide",
)


# ---- Cached resources ----


@st.cache_resource
def get_inference_engine():
    """Singleton inference engine — loaded once per session."""
    return InferenceEngine()


@st.cache_data
def load_demo_engines():
    """Load the canned demo engines once."""
    with open(DEMO_DATA_PATH) as f:
        return json.load(f)


def regime_color(regime):
    return {"Critical": "#e74c3c", "Watch": "#f39c12", "Healthy": "#27ae60"}.get(
        regime, "#888888"
    )


# ---- Header ----

st.title("🛠️ C-MAPSS FD001 — RUL Prediction Demo")

st.info(
    "**This is a streamlined live demo running only the XGBoost model.** "
    "The full version with both XGBoost and LSTM live inference + FastAPI backend "
    "is in the [GitHub repo](https://github.com/beebzy-droid/predictive-maintenance-cmapss). "
    "Both versions use identical training pipelines — only the deployment surface differs."
)

st.markdown(
    "**Predicting Remaining Useful Life (RUL) for turbofan engines using sensor data.** "
    "This dashboard demonstrates a model trained on NASA's C-MAPSS dataset (a proxy for "
    "rotating industrial equipment). Pick an engine from the dropdown to see the model's "
    "predictions compared to the ground truth."
)


# ---- Load resources ----

engine = get_inference_engine()
examples = load_demo_engines()


# ---- Sidebar: engine selection ----

st.sidebar.header("Engine Selection")

engine_labels = {
    f"Engine #{e['engine_id']} — {e['regime']} (actual RUL = {e['actual_rul']:.0f})": e
    for e in examples
}

selected_label = st.sidebar.selectbox(
    "Choose a demo engine",
    options=list(engine_labels.keys()),
    index=0,
)
selected_engine = engine_labels[selected_label]

st.sidebar.markdown(f"**Test set index:** {selected_engine['test_index']}")
st.sidebar.markdown(f"**Actual RUL:** {selected_engine['actual_rul']:.0f} cycles")
st.sidebar.markdown(
    f"**Regime:** <span style='color:{regime_color(selected_engine['regime'])}; "
    f"font-weight:bold'>{selected_engine['regime']}</span>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About the regimes:**\n"
    "- **Critical** (actual RUL < 30): engine is near failure\n"
    "- **Watch** (30 ≤ RUL < 80): engine is degrading\n"
    "- **Healthy** (RUL ≥ 80): engine is fine"
)


# ---- Main tab structure ----

tab_main, tab_comparison, tab_about = st.tabs(
    [
        "🔍 Live Prediction",
        "📊 Classical vs Deep",
        "ℹ️ About",
    ]
)


# ===== TAB 1: Live prediction =====
with tab_main:
    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.subheader("Prediction")

        # Cache the prediction by engine_id so Streamlit re-runs don't re-trigger inference
        @st.cache_data
        def _cached_predict(engine_id: int, window_tuple: tuple) -> dict:
            # window_tuple is a hashable representation of the window for caching
            window = np.array(window_tuple)
            return engine.predict_xgb(window, is_normalized=True)

        # Convert window to a hashable tuple for the cache key
        window_tuple = tuple(tuple(row) for row in selected_engine["window"])
        xgb_result = _cached_predict(selected_engine["engine_id"], window_tuple)

        st.markdown("#### 🌲 XGBoost")
        col_a, col_b = st.columns(2)
        col_a.metric("Predicted RUL", f"{xgb_result['predicted_rul']:.1f}")
        col_b.metric("Actual RUL", f"{selected_engine['actual_rul']:.0f}")

        # Use plain markdown with colored text instead of st.error/st.success
        # to avoid height-changing alert boxes that cause layout shifts
        if xgb_result["alarm"]:
            st.markdown(
                f"<div style='padding: 0.6rem; border-radius: 0.4rem; "
                f"background-color: rgba(220, 53, 69, 0.2); "
                f"border: 1px solid rgba(220, 53, 69, 0.5); color: #ff6b7a;'>"
                f"🚨 <b>ALARM</b> — Predicted RUL ≤ {xgb_result['alarm_threshold']}. "
                f"Maintenance recommended."
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='padding: 0.6rem; border-radius: 0.4rem; "
                f"background-color: rgba(40, 167, 69, 0.2); "
                f"border: 1px solid rgba(40, 167, 69, 0.5); color: #6bcb77;'>"
                f"✅ No alarm. Engine in regime: <b>{xgb_result['regime']}</b>."
                f"</div>",
                unsafe_allow_html=True,
            )

        # Show error vs actual
        st.markdown("")  # small spacer
        error = xgb_result["predicted_rul"] - selected_engine["actual_rul"]
        bias_label = (
            "Conservative (early)"
            if error < -0.5
            else ("Late (dangerous)" if error > 0.5 else "Spot on")
        )
        st.markdown(f"**Error:** {error:+.1f} cycles ({bias_label})")

    with col_left:
        st.subheader("Sensor traces (30 cycles before current observation)")

        st.markdown(
            "These are the z-scored sensor readings the model sees. The four key sensors "
            "shown below are those identified in the EDA as most informative."
        )

        window_array = np.array(selected_engine["window"])
        timesteps = np.arange(30)

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        key_sensors_with_idx = [
            ("sensor_4", SENSOR_NAMES.index("sensor_4"), "Rising"),
            ("sensor_11", SENSOR_NAMES.index("sensor_11"), "Rising"),
            ("sensor_12", SENSOR_NAMES.index("sensor_12"), "Falling"),
            ("sensor_20", SENSOR_NAMES.index("sensor_20"), "Falling"),
        ]
        for ax, (name, idx, direction) in zip(axes.flat, key_sensors_with_idx):
            ax.plot(
                timesteps,
                window_array[:, idx],
                marker="o",
                markersize=4,
                linewidth=1.5,
                color="steelblue",
            )
            ax.axhline(y=0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.set_title(f"{name} ({direction} as engine degrades)", fontsize=10)
            ax.set_xlabel("Cycle within window")
            ax.set_ylabel("Z-scored value")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("All 14 sensors over the 30-cycle window")
        df_window = pd.DataFrame(window_array, columns=SENSOR_NAMES)
        df_window.index.name = "Cycle"
        st.dataframe(
            df_window.style.background_gradient(cmap="RdBu_r", axis=None), height=400
        )


# ===== TAB 2: Classical vs Deep =====
with tab_comparison:
    st.header("Classical vs Deep — Why XGBoost was chosen")

    st.markdown(
        "This project tested both classical (XGBoost, Random Forest) and deep (LSTM) approaches "
        "for predicting Remaining Useful Life on FD001 turbofan data. With matched preprocessing — "
        "no smoothing tricks, identical engineered features for classical, raw windowed input for "
        "deep — **classical methods outperformed deep learning on this dataset.**"
    )

    st.markdown("---")

    st.subheader("Test set results (FD001, 100 engines)")

    metrics_df = pd.DataFrame(
        {
            "Model": [
                "Honest-tuned XGBoost",
                "LSTM-64×2 (best)",
                "Vanilla LSTM (untuned)",
            ],
            "Test RMSE": [11.62, 12.66, 13.69],
            "Test Score": [207, 291, 425],
            "F1 at T=35": [0.98, "~0.95", "~0.85"],
            "Reproducibility": [
                "Engine-level GroupKFold CV",
                "Architecture sweep (fixed seed)",
                "Single training run",
            ],
        }
    )
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    st.markdown(
        "**Headline:** XGBoost beats the best LSTM by 1.04 RMSE points and 84 Score points. "
        "Score doubled on the LSTM because it makes more *late* predictions — the asymmetrically-penalized error type."
    )

    st.warning(
        "⚠️ **Note:** This live demo runs XGBoost only. The LSTM comparison numbers above are "
        "from the Week 5 evaluation in the GitHub repo (see `notebooks/04_lstm.ipynb`). "
        "To see live LSTM predictions side-by-side with XGBoost, clone the repo and run the local version."
    )

    st.markdown("---")

    st.subheader("Three reasons the LSTM lost")
    st.markdown("""
        **1. Strong feature engineering already extracts most of the signal.**
        The 70 tabular features used by XGBoost (mean, std, slope, min, max per sensor over the window)
        encode the same sensor-degradation rates that an LSTM would have to discover from raw data.
        SHAP analysis (Week 4) confirmed XGBoost relies primarily on slope features — exactly the
        physical signal the EDA highlighted.

        **2. The piecewise-linear RUL target favors trees structurally.**
        XGBoost can encode "if all sensors look healthy, predict 125" as a literal leaf rule.
        LSTMs must learn the flat top of the target through a smooth nonlinearity, which is harder
        and requires more representational capacity than the dataset rewards.

        **3. Without aggressive preprocessing innovations, LSTM has no advantage.**
        Task 2 of Week 5 showed that moving-median smoothing on z-scored data weakens slope-RUL
        correlations rather than helping. The published deep-LSTM benchmarks (Asif et al., RMSE 7.78)
        used heavy preprocessing on *raw* sensor values that this project didn't replicate.
        """)


# ===== TAB 3: About =====
with tab_about:
    st.header("About this project")

    st.markdown("""
        **Problem:** Predict Remaining Useful Life (RUL) of turbofan engines from sensor time-series data.
        Models that do this well enable **condition-based maintenance** — replacing parts before they fail
        without wasting good components.

        **Dataset:** NASA C-MAPSS FD001 (Saxena & Goebel 2008). 100 train engines run to failure;
        100 test engines truncated at random points; sensor readings from 21 channels per cycle.
        FD001 is the simplest subset — single operating condition, single fault mode.

        **Approach:** This project builds the full pipeline (EDA, feature engineering, modeling,
        evaluation, deployment) end-to-end. The classical-vs-deep comparison is presented honestly:
        classical methods (XGBoost) outperformed deep learning (LSTM) on this dataset under matched
        preprocessing.

        ### Engineering practices applied
        - **Engine-level data splits** (not row-level random splits) to prevent leakage from
          consecutive sensor windows of the same engine
        - **Engine-level GroupKFold cross-validation** during hyperparameter tuning (caught a leakage
          bug that initially showed CV-RMSE 3.59 vs. honest 13.60)
        - **Winner's curse detection** in LSTM Optuna tuning — chose the reproducible result over
          the apparent peak performance
        - **NASA C-MAPSS Score metric** alongside RMSE — captures the operational asymmetry between
          early and late predictions
        - **SHAP interpretation** confirming model-EDA-features agreement across three independent
          analyses
        - **Calibration analysis by regime** — model is dramatically more accurate in the Critical
          regime (RMSE 5.08) than the overall RMSE suggests (11.62)

        ### Stack
        - **Modeling:** scikit-learn, XGBoost, TensorFlow/Keras (local only)
        - **Interpretation:** SHAP
        - **Tuning:** Optuna with GroupKFold
        - **API:** FastAPI + Uvicorn (local only)
        - **Dashboard:** Streamlit + matplotlib
        - **Testing:** pytest (46 tests, all passing)

        ### Links
        - **Full repo with both models live:** [github.com/beebzy-droid/predictive-maintenance-cmapss](https://github.com/beebzy-droid/predictive-maintenance-cmapss)
        - **Production readiness analysis:** [`docs/production_readiness.md`](https://github.com/beebzy-droid/predictive-maintenance-cmapss/blob/main/docs/production_readiness.md)
        - **Engineering decisions:** [`docs/decisions.md`](https://github.com/beebzy-droid/predictive-maintenance-cmapss/blob/main/docs/decisions.md)
        """)


# ---- Footer ----

st.markdown("---")
st.markdown(
    "**Project repo:** [beebzy-droid/predictive-maintenance-cmapss](https://github.com/beebzy-droid/predictive-maintenance-cmapss) · "
    "**Stack:** Python · XGBoost · Streamlit · pandas · matplotlib · scikit-learn"
)
