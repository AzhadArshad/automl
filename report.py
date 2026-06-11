"""HTML report generator — self-contained, embeds all charts as base64."""

import base64
import logging
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.io as pio

logger = logging.getLogger(__name__)


def _img_to_base64(path: str) -> Optional[str]:
    """Read an image file and return a base64-encoded data URI string.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        A data URI string ("data:image/png;base64,…") or None if the file
        is missing or unreadable.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:
        logger.warning("Could not read image %s: %s", path, exc)
        return None


def _leaderboard_html(leaderboard: pd.DataFrame) -> str:
    """Render the leaderboard DataFrame as a styled HTML table.

    Higher CV Score rows get a progressively greener background.

    Args:
        leaderboard: DataFrame with Model, CV Score, Std, Fit Time (s) columns.

    Returns:
        HTML string for the table.
    """
    max_score = leaderboard["CV Score"].max()
    min_score = leaderboard["CV Score"].min()
    score_range = max(max_score - min_score, 1e-6)

    rows = []
    for _, row in leaderboard.iterrows():
        intensity = int(((row["CV Score"] - min_score) / score_range) * 120)
        bg = f"rgba(0, {180 + intensity}, 100, 0.15)"
        rows.append(
            f"<tr style='background:{bg}'>"
            f"<td>{row['Model']}</td>"
            f"<td><b>{row['CV Score']:.4f}</b></td>"
            f"<td>{row['Std']:.4f}</td>"
            f"<td>{row['Fit Time (s)']:.1f}s</td>"
            f"</tr>"
        )

    return f"""
    <table>
      <thead>
        <tr>
          <th>Model</th><th>CV Score</th><th>Std</th><th>Fit Time</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def _null_summary_html(null_summary: dict[str, float]) -> str:
    """Render the null summary as a compact HTML table.

    Only shows columns that have at least one null.

    Args:
        null_summary: Dict of column name → null percentage (0.0–1.0).

    Returns:
        HTML string or a short message if no nulls are present.
    """
    nulls = {k: v for k, v in null_summary.items() if v > 0}
    if not nulls:
        return "<p>No missing values detected.</p>"

    rows = "".join(
        f"<tr><td>{col}</td><td>{pct:.1%}</td></tr>"
        for col, pct in sorted(nulls.items(), key=lambda kv: -kv[1])
    )
    return f"""
    <table>
      <thead><tr><th>Column</th><th>Null %</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """


def generate_report(
    profile: Any,
    task_type: Any,
    metric: str,
    leaderboard: pd.DataFrame,
    tuning_results: dict,
    explainer_result: Any,
    fit_duration: float,
    output_path: str = "report.html",
) -> None:
    """Build and save a self-contained HTML AutoML report.

    All charts are embedded as base64 images or inline Plotly HTML so the
    file is portable with no external dependencies.

    Args:
        profile: DataProfile from core.ingestion.load_data().
        task_type: TaskType enum value.
        metric: Metric string used for evaluation.
        leaderboard: Model leaderboard DataFrame.
        tuning_results: Dict of {model_name: TuningResult} from Optuna.
        explainer_result: ExplainerResult from shap_explainer.explain().
        fit_duration: Total wall-clock time in seconds.
        output_path: Where to write the HTML file.
    """
    # ── Plotly bar chart: model comparison ──────────────────────────────
    fig_lb = px.bar(
        leaderboard,
        x="Model",
        y="CV Score",
        error_y="Std",
        title="Model CV Score Comparison",
        color="CV Score",
        color_continuous_scale="Teal",
        template="plotly_white",
    )
    fig_lb.update_layout(height=380)
    leaderboard_chart_html = pio.to_html(fig_lb, full_html=False, include_plotlyjs="cdn")

    # ── Plotly bar chart: SHAP importance ───────────────────────────────
    shap_chart_html = ""
    if explainer_result and explainer_result.feature_importance:
        top_20 = list(explainer_result.feature_importance.items())[:20]
        imp_df = pd.DataFrame(top_20, columns=["Feature", "Mean |SHAP|"])
        fig_shap = px.bar(
            imp_df.sort_values("Mean |SHAP|"),
            x="Mean |SHAP|",
            y="Feature",
            orientation="h",
            title="Top 20 Features — Mean |SHAP Value|",
            template="plotly_white",
        )
        fig_shap.update_layout(height=480)
        shap_chart_html = pio.to_html(fig_shap, full_html=False, include_plotlyjs=False)

    # ── Embed SHAP beeswarm image ────────────────────────────────────────
    beeswarm_src = None
    waterfall_src = None
    if explainer_result and explainer_result.plot_paths:
        beeswarm_src = _img_to_base64(explainer_result.plot_paths.get("beeswarm", ""))
        waterfall_src = _img_to_base64(explainer_result.plot_paths.get("waterfall", ""))

    # ── Best model params table ──────────────────────────────────────────
    best_model_name = leaderboard["Model"].iloc[0]
    best_params_html = "<p>Default parameters (no tuning result).</p>"
    if best_model_name in tuning_results:
        params = tuning_results[best_model_name].best_params
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in params.items())
        best_params_html = f"""
        <table>
          <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """

    # ── HTML assembly ─────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>AutoML Report</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f8f9fa; color: #212529; margin: 0; padding: 24px;
    }}
    h1 {{ color: #0d6efd; margin-bottom: 4px; }}
    h2 {{ color: #343a40; border-bottom: 2px solid #dee2e6;
         padding-bottom: 6px; margin-top: 40px; }}
    .meta {{ color: #6c757d; font-size: 0.9rem; margin-bottom: 32px; }}
    .card {{
      background: white; border-radius: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,.07);
      padding: 24px; margin-bottom: 24px;
    }}
    .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
    .stat-box {{
      background: #e9f0ff; border-radius: 8px; padding: 16px; text-align: center;
    }}
    .stat-box .value {{ font-size: 2rem; font-weight: 700; color: #0d6efd; }}
    .stat-box .label {{ font-size: 0.85rem; color: #495057; margin-top: 4px; }}
    table {{
      width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 12px;
    }}
    th {{ background: #e9ecef; text-align: left; padding: 8px 12px; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #e9ecef; }}
    img.shap-img {{ max-width: 100%; border-radius: 8px; margin-top: 12px; }}
    .badge {{
      display: inline-block; padding: 2px 10px; border-radius: 20px;
      font-size: 0.8rem; font-weight: 600; background: #0d6efd; color: white;
    }}
    footer {{ text-align: center; color: #adb5bd; font-size: 0.8rem; margin-top: 48px; }}
  </style>
</head>
<body>

<h1>AutoML Report</h1>
<p class="meta">
  Task: <span class="badge">{task_type.value}</span>&nbsp;
  Metric: <span class="badge">{metric}</span>&nbsp;
  Training time: <b>{fit_duration:.1f}s</b>
</p>

<!-- Dataset summary -->
<h2>Dataset Summary</h2>
<div class="card">
  <div class="grid-3">
    <div class="stat-box">
      <div class="value">{profile.n_rows:,}</div>
      <div class="label">Rows</div>
    </div>
    <div class="stat-box">
      <div class="value">{profile.n_cols}</div>
      <div class="label">Columns</div>
    </div>
    <div class="stat-box">
      <div class="value">{len(profile.numerical_cols)}</div>
      <div class="label">Numerical features</div>
    </div>
    <div class="stat-box">
      <div class="value">{len(profile.categorical_cols)}</div>
      <div class="label">Categorical features</div>
    </div>
    <div class="stat-box">
      <div class="value">{len(profile.datetime_cols)}</div>
      <div class="label">Datetime features</div>
    </div>
    <div class="stat-box">
      <div class="value">{len(profile.id_cols)}</div>
      <div class="label">ID cols (dropped)</div>
    </div>
  </div>
</div>

<!-- Missing values -->
<h2>Missing Values</h2>
<div class="card">
  {_null_summary_html(profile.null_summary)}
</div>

<!-- Leaderboard -->
<h2>Model Leaderboard</h2>
<div class="card">
  {_leaderboard_html(leaderboard)}
</div>
<div class="card">
  {leaderboard_chart_html}
</div>

<!-- Best model -->
<h2>Best Model — {best_model_name}</h2>
<div class="card">
  <p>CV Score: <b>{leaderboard['CV Score'].iloc[0]:.4f}</b></p>
  <h3>Tuned Hyperparameters</h3>
  {best_params_html}
</div>

<!-- SHAP importance chart -->
<h2>Feature Importance (SHAP)</h2>
<div class="card">
  {shap_chart_html}
</div>

{'<div class="card"><h3>Beeswarm Plot</h3><img class="shap-img" src="' + beeswarm_src + '"/></div>' if beeswarm_src else ''}
{'<div class="card"><h3>Waterfall Plot (row 0)</h3><img class="shap-img" src="' + waterfall_src + '"/></div>' if waterfall_src else ''}

<footer>Generated by AutoML &nbsp;|&nbsp; {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</footer>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    logger.info("HTML report written to: %s", output_path)
