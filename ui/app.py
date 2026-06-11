"""Streamlit UI — upload, train, leaderboard, explain, predict, export."""

import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="AutoML", page_icon="🤖", layout="wide")
st.title("AutoML System")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key, default in [
    ("job_id", None),
    ("feature_names", []),
    ("task_type", ""),
    ("training_done", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    top_n_models     = st.slider("Top N models to tune", 1, 8, 3)
    n_optuna_trials  = st.slider("Optuna trials per model", 10, 100, 50)
    ensemble_strategy = st.selectbox("Ensemble strategy", ["weighted", "simple", "stacking"])
    enable_fe        = st.checkbox("Enable feature engineering", value=True)
    time_limit       = st.number_input("Time limit (s)", min_value=60, max_value=3600, value=300)


# ---------------------------------------------------------------------------
# Section 1 — Upload
# ---------------------------------------------------------------------------
st.header("1 · Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    preview_df = pd.read_csv(uploaded_file)
    uploaded_file.seek(0)   # reset after preview read

    st.write(f"**{preview_df.shape[0]:,} rows × {preview_df.shape[1]} columns**")
    st.dataframe(preview_df.head(5), use_container_width=True)

    target_col = st.selectbox("Select target column", preview_df.columns.tolist())

    if st.button("Start Training", type="primary"):
        with st.spinner("Submitting job…"):
            response = requests.post(
                f"{API_BASE}/train",
                files={"file": (uploaded_file.name, uploaded_file, "text/csv")},
                data={
                    "target": target_col,
                    "top_n_models": top_n_models,
                    "n_optuna_trials": n_optuna_trials,
                    "ensemble_strategy": ensemble_strategy,
                    "enable_feature_engineering": str(enable_fe).lower(),
                    "time_limit": time_limit,
                },
            )
        if response.status_code == 200:
            st.session_state.job_id = response.json()["job_id"]
            st.session_state.training_done = False
            st.success(f"Job submitted: `{st.session_state.job_id}`")
        else:
            st.error(f"Failed to start training: {response.text}")


# ---------------------------------------------------------------------------
# Section 2 — Training progress
# ---------------------------------------------------------------------------
if st.session_state.job_id and not st.session_state.training_done:
    st.header("2 · Training Progress")

    status_resp = requests.get(f"{API_BASE}/status/{st.session_state.job_id}")
    if status_resp.status_code == 200:
        status_data = status_resp.json()
        job_status   = status_data["status"]
        progress     = status_data["progress"]
        current_step = status_data["current_step"]

        st.progress(progress / 100, text=f"{progress}% — {current_step}")

        if job_status == "done":
            st.success("Training complete!")
            st.session_state.training_done = True
        elif job_status == "failed":
            st.error(f"Training failed: {status_data.get('error', 'unknown error')}")
        else:
            time.sleep(3)
            st.rerun()
    else:
        st.warning("Could not reach the API. Is the server running?")


# ---------------------------------------------------------------------------
# Section 3 — Leaderboard
# ---------------------------------------------------------------------------
if st.session_state.training_done:
    st.header("3 · Model Leaderboard")

    lb_resp = requests.get(f"{API_BASE}/leaderboard/{st.session_state.job_id}")
    if lb_resp.status_code == 200:
        records = lb_resp.json().get("leaderboard", [])
        if records:
            lb_df = pd.DataFrame(records)

            # Colour-code CV Score column
            st.dataframe(
                lb_df.style.background_gradient(subset=["CV Score"], cmap="Greens"),
                use_container_width=True,
            )

            fig = px.bar(
                lb_df,
                x="Model",
                y="CV Score",
                error_y="Std",
                title="Model Comparison",
                color="CV Score",
                color_continuous_scale="Teal",
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 4 — Explainability
# ---------------------------------------------------------------------------
if st.session_state.training_done:
    st.header("4 · Feature Importance (SHAP)")

    exp_resp = requests.get(f"{API_BASE}/explain/{st.session_state.job_id}")
    if exp_resp.status_code == 200:
        exp_data = exp_resp.json()
        importance = exp_data.get("feature_importance", {})
        plot_paths = exp_data.get("plot_paths", {})

        if importance:
            imp_df = pd.DataFrame(
                list(importance.items())[:20], columns=["Feature", "Mean |SHAP|"]
            )
            fig = px.bar(
                imp_df.sort_values("Mean |SHAP|"),
                x="Mean |SHAP|",
                y="Feature",
                orientation="h",
                title="Top 20 Features by SHAP Importance",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(imp_df, use_container_width=True)

        if "beeswarm" in plot_paths:
            st.image(plot_paths["beeswarm"], caption="SHAP Beeswarm")

        if "waterfall" in plot_paths:
            st.image(plot_paths["waterfall"], caption="SHAP Waterfall (row 0)")
    else:
        st.info("Explanation not available yet.")


# ---------------------------------------------------------------------------
# Section 5 — Single-row prediction
# ---------------------------------------------------------------------------
if st.session_state.training_done:
    st.header("5 · Predict")

    lb_resp = requests.get(f"{API_BASE}/leaderboard/{st.session_state.job_id}")
    if lb_resp.status_code == 200:
        records = lb_resp.json().get("leaderboard", [])
        if records:
            # Build input form from leaderboard feature names (approximate)
            st.write("Enter feature values:")
            feature_inputs: dict = {}

            # We don't expose feature names from the API yet — use a free-text JSON entry
            raw_json = st.text_area(
                "Feature values (JSON)",
                value='{"Age": 30, "Fare": 10.5, "Pclass": 3}',
                height=120,
            )

            if st.button("Predict"):
                import json
                try:
                    features = json.loads(raw_json)
                except json.JSONDecodeError:
                    st.error("Invalid JSON — please check your input.")
                    features = None

                if features:
                    pred_resp = requests.post(
                        f"{API_BASE}/predict",
                        json={"job_id": st.session_state.job_id, "features": features},
                    )
                    if pred_resp.status_code == 200:
                        result = pred_resp.json()
                        st.metric("Prediction", result["prediction"])
                        if "probabilities" in result:
                            proba = result["probabilities"]
                            st.write("**Class probabilities:**")
                            for i, p in enumerate(proba):
                                st.progress(p, text=f"Class {i}: {p:.1%}")
                    else:
                        st.error(f"Prediction failed: {pred_resp.text}")


# ---------------------------------------------------------------------------
# Section 6 — Export
# ---------------------------------------------------------------------------
if st.session_state.training_done:
    st.header("6 · Export Model")

    col1, col2 = st.columns(2)

    with col1:
        export_resp = requests.get(f"{API_BASE}/export/{st.session_state.job_id}")
        if export_resp.status_code == 200:
            st.download_button(
                label="Download best model (.pkl)",
                data=export_resp.content,
                file_name="best_model.pkl",
                mime="application/octet-stream",
            )

    with col2:
        st.info(
            "To run batch predictions, use the `/predict/batch` endpoint directly "
            "with a CSV upload."
        )
