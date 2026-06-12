"""Streamlit UI — upload, train, leaderboard, explain, predict, export."""

import io
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Cached helpers — Streamlit reruns the whole script on every interaction
# (and every 3s while polling), so anything expensive must be cached.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_preview(csv_bytes: bytes) -> pd.DataFrame:
    """Parse the uploaded CSV once per file instead of on every rerun."""
    return pd.read_csv(io.BytesIO(csv_bytes))


@st.cache_data(show_spinner=False)
def fetch_leaderboard(job_id: str) -> dict:
    """Leaderboard is immutable once training is done — fetch once per job."""
    resp = requests.get(f"{API_BASE}/leaderboard/{job_id}", timeout=15)
    return resp.json() if resp.status_code == 200 else {}


@st.cache_data(show_spinner=False)
def fetch_explanation(job_id: str) -> dict:
    resp = requests.get(f"{API_BASE}/explain/{job_id}", timeout=15)
    return resp.json() if resp.status_code == 200 else {}


@st.cache_data(show_spinner=False)
def fetch_model_bytes(job_id: str) -> bytes | None:
    """The pickled model can be tens of MB — download once, not per rerun."""
    resp = requests.get(f"{API_BASE}/export/{job_id}", timeout=60)
    return resp.content if resp.status_code == 200 else None

# Human-readable labels for the evaluation metrics used by the backend
METRIC_LABELS = {
    "roc_auc": "ROC AUC",
    "f1_macro": "F1 Macro",
    "rmse": "RMSE",
}
# Metrics where a lower score means a better model
LOWER_IS_BETTER = {"rmse"}

st.set_page_config(page_title="AutoML", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
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
    n_optuna_trials  = st.slider("Optuna trials per model", 1, 50, 10)
    ensemble_strategy = st.selectbox("Ensemble strategy", ["weighted", "simple", "stacking"])
    enable_fe        = st.checkbox("Enable feature engineering", value=True)
    time_limit       = st.number_input("Time limit (s)", min_value=60, max_value=3600, value=300)


# ---------------------------------------------------------------------------
# Overview — what this project is and how to demo it
# ---------------------------------------------------------------------------
st.markdown(
    "Upload any tabular CSV, pick a target column, and get a tuned, explained, "
    "ready-to-download ML model — no code required."
)

with st.expander("ℹ️ About this project — what it does, how it's built, how to demo",expanded=True):
    st.markdown(
        """
#### What it does
An end-to-end **AutoML system for tabular data**. From a single CSV upload it automatically:

1. **Profiles the data** — classifies columns (numerical / categorical / datetime / text), drops ID columns, flags leakage suspects
2. **Detects the task** — binary, multiclass, or regression, with the right metric (ROC AUC / F1 Macro / RMSE)
3. **Preprocesses** — median imputation, robust scaling, one-hot & ordinal encoding, datetime expansion
4. **Engineers features** *(optional)* — polynomial interactions on the most predictive columns
5. **Trains a model zoo** — Logistic/Ridge, RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost, MLP, KNN, SVM — each with 5-fold cross-validation
6. **Tunes the top models** with **Optuna** hyperparameter search
7. **Ensembles** the tuned models (weighted average, simple average, or stacking)
8. **Explains predictions** with **SHAP** — global feature importance, beeswarm, waterfall
9. **Tracks every run** in **MLflow**

#### How it's built
- **FastAPI** backend — training runs as an async background job; the UI polls a status
  endpoint for live progress (the progress bar advances after every Optuna trial)
- **Streamlit** frontend (this app) — talks to the API over REST, holds no ML logic itself
- **scikit-learn / XGBoost / LightGBM / CatBoost / Optuna / SHAP / MLflow** under the hood
- Single Docker container deployment — both services in one image

#### How to demo (2 minutes)
1. Grab a dataset — e.g. [Titanic](https://www.kaggle.com/c/titanic/data) (binary classification)
   or [Housing](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) (regression)
2. **Upload** the CSV below and select the target column (`Survived` / `price`)
3. Hit **Start Training** and watch the live progress (~1–3 min with default settings)
4. Explore the **leaderboard**, **SHAP feature importance**, and run a **single-row prediction**
5. **Download** the trained ensemble as a pickle file

#### Limits (free-tier hosting)
Max **20 MB / 50k rows** per upload · one training job at a time · models live in memory
(train and predict in the same session)
        """
    )


# ---------------------------------------------------------------------------
# Section 1 — Upload
# ---------------------------------------------------------------------------
st.header("1 · Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    csv_bytes = uploaded_file.getvalue()
    preview_df = load_preview(csv_bytes)

    st.write(f"**{preview_df.shape[0]:,} rows × {preview_df.shape[1]} columns**")
    st.dataframe(preview_df.head(5), width="stretch")

    target_col = st.selectbox("Select target column", preview_df.columns.tolist())

    if st.button("Start Training", type="primary"):
        with st.spinner("Submitting job…"):
            response = requests.post(
                f"{API_BASE}/train",
                files={"file": (uploaded_file.name, csv_bytes, "text/csv")},
                timeout=120,
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
            st.success("Training started!")
        else:
            st.error(f"Failed to start training: {response.text}")


# ---------------------------------------------------------------------------
# Section 2 — Training progress
# ---------------------------------------------------------------------------
if st.session_state.job_id and not st.session_state.training_done:
    st.header("2 · Training Progress")

    try:
        status_resp = requests.get(
            f"{API_BASE}/status/{st.session_state.job_id}", timeout=10
        )
    except requests.RequestException:
        status_resp = None

    if status_resp is not None and status_resp.status_code == 200:
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

    lb_data = fetch_leaderboard(st.session_state.job_id)
    if lb_data:
        records = lb_data.get("leaderboard", [])
        if records:
            lb_df = pd.DataFrame(records)

            task_type = lb_data.get("task_type", "")
            metric = lb_data.get("metric", "")
            metric_label = METRIC_LABELS.get(metric, metric or "CV Score")
            lower_is_better = metric in LOWER_IS_BETTER

            if metric:
                direction = "lower is better" if lower_is_better else "higher is better"
                st.markdown(
                    f"Task: **{task_type.replace('_', ' ')}** &nbsp;·&nbsp; "
                    f"Metric: **{metric_label}** ({direction})"
                )

            # Colour-code CV Score column — darkest green = best model
            cmap = "Greens_r" if lower_is_better else "Greens"
            st.dataframe(
                lb_df.style.background_gradient(subset=["CV Score"], cmap=cmap),
                width="stretch",
            )

            fig = px.bar(
                lb_df,
                x="Model",
                y="CV Score",
                error_y="Std",
                title=f"Model Comparison — {metric_label}",
                labels={"CV Score": f"CV Score ({metric_label})"},
                color="CV Score",
                color_continuous_scale="Teal_r" if lower_is_better else "Teal",
            )
            st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Section 4 — Explainability
# ---------------------------------------------------------------------------
if st.session_state.training_done:
    st.header("4 · Feature Importance (SHAP)")

    exp_data = fetch_explanation(st.session_state.job_id)
    if exp_data:
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
            st.plotly_chart(fig, width="stretch")
            st.dataframe(imp_df, width="stretch")

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

    st.write("Enter feature values:")

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
                timeout=60,
            )
            if pred_resp.status_code == 200:
                result = pred_resp.json()
                st.metric("Prediction", result["prediction"])
                if "probabilities" in result:
                    st.write("**Class probabilities:**")
                    for label, p in result["probabilities"].items():
                        st.progress(float(p), text=f"{label}: {float(p):.1%}")
            else:
                st.error(f"Prediction failed: {pred_resp.text}")


# ---------------------------------------------------------------------------
# Section 6 — Export
# ---------------------------------------------------------------------------
if st.session_state.training_done:
    st.header("6 · Export Model")

    col1, col2 = st.columns(2)

    with col1:
        model_bytes = fetch_model_bytes(st.session_state.job_id)
        if model_bytes:
            st.download_button(
                label="Download trained pipeline (.pkl)",
                data=model_bytes,
                file_name="automl_pipeline.pkl",
                mime="application/octet-stream",
            )
        st.caption(
            "Full pipeline: preprocessing + feature engineering + ensemble — "
            "`.predict()` accepts raw, unprocessed data."
        )

    with col2:
        st.markdown("**Using the downloaded pipeline:**")
        st.code(
            """# Requires this project's source code:
# git clone https://github.com/AzhadArshad/automl
# pip install -r requirements-deploy.txt   (run from the repo root)

import pickle
import pandas as pd

with open("automl_pipeline.pkl", "rb") as f:
    aml = pickle.load(f)

new_data = pd.read_csv("new_data.csv")   # same columns as training
print(aml.predict(new_data))""",
            language="python",
        )
        st.caption(
            "⚠️ The pickle references this project's classes, so it only loads "
            "with the repo code on your Python path."
        )
