"""
Streamlit dashboard for the predictive maintenance demo.

Run locally:
    streamlit run app/dashboard.py

The FastAPI server must be running on port 8000 (see app/api.py).
"""

import json
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Configuration
API_BASE = "http://localhost:8000"

# Sensor names matching src.features.KEEP_SENSORS
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

# Sensors flagged in Week 2 EDA as most informative
KEY_RISING_SENSORS = ["sensor_4", "sensor_11"]
KEY_FALLING_SENSORS = ["sensor_12", "sensor_20"]

# ---- Page config ----
st.set_page_config(
    page_title="C-MAPSS RUL Prediction Demo",
    page_icon="🛠️",
    layout="wide",
)


# ---- Helper functions ----


@st.cache_data(ttl=300)
def fetch_examples():
    """Get the canned demo engines from the API."""
    try:
        response = requests.get(f"{API_BASE}/examples", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Could not reach the API at {API_BASE}: {e}")
        st.error(
            "Make sure the FastAPI server is running: `uvicorn app.api:app --port 8000`"
        )
        return None


def predict(window, model_name, engine_id):
    """Call the prediction API for one model."""
    try:
        response = requests.post(
            f"{API_BASE}/predict/{model_name}",
            json={
                "window": window,
                "is_normalized": True,
                "engine_id": engine_id,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Prediction call failed: {e}")
        return None


def regime_color(regime):
    return {"Critical": "#e74c3c", "Watch": "#f39c12", "Healthy": "#27ae60"}.get(
        regime, "#888888"
    )


# ---- Header ----

st.title("🛠️ C-MAPSS FD001 — RUL Prediction Demo")

st.markdown(
    "**Predicting Remaining Useful Life (RUL) for turbofan engines using sensor data.** "
    "This dashboard demonstrates a model trained on NASA's C-MAPSS dataset, which simulates "
    "turbofan degradation. Pick an engine from the dropdown to see the model's predictions "
    "compared to the ground truth."
)

st.markdown(
    "Two models are served side-by-side: **honest-tuned XGBoost** (the recommended model, "
    "Test RMSE 11.62, F1=0.98 as a maintenance alarm at threshold T=35) and **LSTM-64×2** "
    "(included for the classical-vs-deep comparison story — see project README)."
)

# ---- Load demo engines ----

examples = fetch_examples()
if examples is None:
    st.stop()

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

# Show metadata
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
    "- **Healthy** (RUL ≥ 80): engine is fine\n\n"
    "See [`docs/production_readiness.md`](https://github.com/beebzy-droid/predictive-maintenance-cmapss/blob/main/docs/production_readiness.md) "
    "for the full operational analysis."
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
        st.subheader("Predictions")

        # XGBoost prediction
        with st.spinner("Running XGBoost..."):
            xgb_result = predict(
                selected_engine["window"], "xgb", selected_engine["engine_id"]
            )
        if xgb_result is not None:
            st.markdown("#### 🌲 XGBoost (recommended)")
            col_a, col_b = st.columns(2)
            col_a.metric("Predicted RUL", f"{xgb_result['predicted_rul']:.1f}")
            col_b.metric("Actual RUL", f"{selected_engine['actual_rul']:.0f}")
            if xgb_result["alarm"]:
                st.error(
                    f"🚨 **ALARM** — Predicted RUL ≤ {xgb_result['alarm_threshold']}. Maintenance recommended."
                )
            else:
                st.success(
                    f"✅ No alarm. Engine in regime: **{xgb_result['regime']}**."
                )

        st.markdown("---")

        # LSTM prediction
        with st.spinner("Running LSTM..."):
            lstm_result = predict(
                selected_engine["window"], "lstm", selected_engine["engine_id"]
            )
        if lstm_result is not None:
            st.markdown("#### 🧠 LSTM (comparison)")
            col_a, col_b = st.columns(2)
            col_a.metric("Predicted RUL", f"{lstm_result['predicted_rul']:.1f}")
            col_b.metric("Actual RUL", f"{selected_engine['actual_rul']:.0f}")
            if lstm_result["alarm"]:
                st.error(
                    f"🚨 **ALARM** — Predicted RUL ≤ {lstm_result['alarm_threshold']}."
                )
            else:
                st.success(
                    f"✅ No alarm. Engine in regime: **{lstm_result['regime']}**."
                )

    with col_left:
        st.subheader("Sensor traces (30 cycles before current observation)")

        st.markdown(
            "These are the z-scored sensor readings the model sees as input. "
            "The four key sensors highlighted are those flagged as most informative "
            "in the EDA — see `notebooks/01_eda.ipynb`."
        )

        window_array = np.array(selected_engine["window"])  # (30, 14)
        timesteps = np.arange(30)

        # 2x2 grid of the four key sensors
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

        # All 14 sensors heatmap
        st.subheader("All 14 sensors over the 30-cycle window")
        df_window = pd.DataFrame(window_array, columns=SENSOR_NAMES)
        df_window.index.name = "Cycle"
        st.dataframe(
            df_window.style.background_gradient(cmap="RdBu_r", axis=None), height=400
        )


# ===== TAB 2: Classical vs Deep comparison =====
with tab_comparison:
    st.header("Classical vs Deep — Why XGBoost was chosen")

    st.markdown(
        "This project tested both classical (XGBoost, Random Forest) and deep (LSTM) approaches "
        "for predicting Remaining Useful Life on FD001 turbofan data. With matched preprocessing — "
        "no smoothing tricks, identical engineered features for classical, raw windowed input for "
        "deep — **classical methods outperformed deep learning on this dataset.**"
    )

    st.markdown("---")

    # Test-set comparison metrics
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
        "Score doubled on the LSTM because it makes more *late* predictions (the asymmetrically-penalized error type)."
    )

    st.markdown("---")

    # Live comparison on the currently selected engine
    st.subheader(f"Live comparison on Engine #{selected_engine['engine_id']}")
    st.markdown(
        f"Selected engine has **actual RUL = {selected_engine['actual_rul']:.0f}** "
        f"({selected_engine['regime']} regime). Predictions from both models:"
    )

    if xgb_result is not None and lstm_result is not None:
        xgb_error = xgb_result["predicted_rul"] - selected_engine["actual_rul"]
        lstm_error = lstm_result["predicted_rul"] - selected_engine["actual_rul"]

        col_x, col_y = st.columns(2)
        col_x.metric(
            "XGBoost error (pred − actual)",
            f"{xgb_error:+.1f} cycles",
            delta=(
                ("Conservative" if xgb_error < 0 else "Late")
                if abs(xgb_error) > 0.5
                else "Right on"
            ),
        )
        col_y.metric(
            "LSTM error (pred − actual)",
            f"{lstm_error:+.1f} cycles",
            delta=(
                ("Conservative" if lstm_error < 0 else "Late")
                if abs(lstm_error) > 0.5
                else "Right on"
            ),
        )

    st.markdown("---")

    # The findings that justify the choice
    st.subheader("Three reasons the LSTM lost (Week 5 analysis)")

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
        The project's Task 2 experiment showed that moving-median smoothing on z-scored data
        weakens slope-RUL correlations rather than helping. The published deep-LSTM benchmarks
        (Asif et al., RMSE 7.78) used heavy preprocessing on *raw* sensor values that we didn't replicate.
        """)

    st.info(
        "💡 The LSTM is kept in the demo for transparency — the project's claim that "
        "**classical methods are sufficient for FD001** would be unverifiable without showing "
        "the deep model's actual behavior. Try picking different engines and watch the LSTM "
        "consistently hedge below the cap (around 110) when the engine is healthy, while "
        "XGBoost commits to 125. That's the under-capacity behavior diagnosed in Week 5."
    )


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
        - **Modeling:** scikit-learn, XGBoost, TensorFlow/Keras
        - **Interpretation:** SHAP
        - **Tuning:** Optuna with GroupKFold
        - **API:** FastAPI + Uvicorn
        - **Dashboard:** Streamlit + matplotlib
        - **Testing:** pytest (46 tests, all passing)

        ### Links
        - **Repo:** [github.com/beebzy-droid/predictive-maintenance-cmapss](https://github.com/beebzy-droid/predictive-maintenance-cmapss)
        - **Production readiness analysis:** [`docs/production_readiness.md`](https://github.com/beebzy-droid/predictive-maintenance-cmapss/blob/main/docs/production_readiness.md)
        - **Engineering decisions:** [`docs/decisions.md`](https://github.com/beebzy-droid/predictive-maintenance-cmapss/blob/main/docs/decisions.md)
        """)


# ---- Footer ----

st.markdown("---")
st.markdown(
    "**Project repo:** [beebzy-droid/predictive-maintenance-cmapss](https://github.com/beebzy-droid/predictive-maintenance-cmapss) · "
    "**Stack:** Python · XGBoost · TensorFlow · FastAPI · Streamlit · pandas · matplotlib"
)
