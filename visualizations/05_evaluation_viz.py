import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Actual results from your output ──────────────────────────
model_labels = ["Baseline", "Low Uncertainty", "Boundary Uncertainty", "High Uncertainty"]
auroc  = [0.675636, 0.589901, 0.713709, 0.637872]
aupr   = [0.731583, 0.656298, 0.742623, 0.668620]
mcc    = [0.161234, 0.131737, 0.322468, 0.067180]
f1     = [0.754728, 0.751508, 0.779844, 0.747484]
tp     = [2514, 2492, 2453, 2525]
fp     = [1623, 1615, 1313, 1706]
tn     = [96,   104,  406,   13]
fn     = [11,   33,   72,    0]
colors = ["#006494", "#437a22", "#01696f", "#a12c7b"]


# ------------------------------------------------------------------
# Chart 1: Which model wins? — side-by-side bar comparison
# ------------------------------------------------------------------
# AUROC, MCC, and F1 are the main measures of how good a model is.
# Higher is always better. The ⭐ marks the best model per metric.
# ------------------------------------------------------------------

fig1 = make_subplots(rows=1, cols=3,
    subplot_titles=["AUROC  (higher = better)",
                    "MCC  (higher = better)",
                    "F1 Score  (higher = better)"])

for col_idx, (metric_name, vals) in enumerate(
        zip(["AUROC", "MCC", "F1"], [auroc, mcc, f1]), start=1):
    best = int(np.argmax(vals))
    bar_colors = [colors[i] if i == best else "#b0ccd4" for i in range(4)]
    fig1.add_trace(go.Bar(
        x=model_labels, y=vals,
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in vals],
        textposition="outside", textfont=dict(size=11),
        showlegend=False, width=0.55
    ), row=1, col=col_idx)
    fig1.add_annotation(
        x=model_labels[best], y=max(vals) + 0.07,
        text="⭐ Best", showarrow=False,
        font=dict(size=11, color=colors[best]), xanchor="center",
        row=1, col=col_idx
    )

fig1.update_yaxes(gridcolor="#dcd9d5", range=[0, 0.95], title_text="Score")
fig1.update_xaxes(tickangle=-20, tickfont=dict(size=10))
fig1.update_layout(
    title=dict(
        text="Step 9 — Which Model Performs Best?<br>"
             "<sup>Boundary Uncertainty (trained on uncertain negatives) wins on all 3 key metrics.</sup>",
        font=dict(size=18), x=0.5),
    plot_bgcolor="#f7f6f2", paper_bgcolor="#f9f8f5",
    height=460, margin=dict(t=120, b=100),
    font=dict(family="Arial", color="#28251d"),
)
fig1.write_image("step9_model_comparison.png", scale=2)
print("✅ step9_model_comparison.png saved")


# ------------------------------------------------------------------
# Chart 2: Confusion matrices — 2×2 grid of heatmaps
# ------------------------------------------------------------------
# For each model, this shows a 2×2 grid:
#   TP = correctly found PTM   | FP = false alarm
#   FN = missed a PTM          | TN = correctly found non-PTM
# Darker green = more predictions of that type.
# ------------------------------------------------------------------

fig2 = make_subplots(rows=2, cols=2,
    subplot_titles=model_labels,
    horizontal_spacing=0.12, vertical_spacing=0.18)

for i, (tp_v, fp_v, tn_v, fn_v) in enumerate(zip(tp, fp, tn, fn)):
    row, col = divmod(i, 2)
    row += 1; col += 1
    z    = [[tn_v, fp_v], [fn_v, tp_v]]
    text = [[f"TN\n{tn_v:,}", f"FP\n{fp_v:,}"],
            [f"FN\n{fn_v:,}", f"TP\n{tp_v:,}"]]
    fig2.add_trace(go.Heatmap(
        z=z, text=text, texttemplate="%{text}",
        colorscale=[[0,"#f9f8f5"],[0.5,"#cedcd8"],[1,"#01696f"]],
        showscale=False, textfont=dict(size=13),
        zmin=0, zmax=max(tp_v, fp_v, tn_v, fn_v), xgap=3, ygap=3
    ), row=row, col=col)
    fig2.update_xaxes(tickvals=[0,1],
        ticktext=['Predicted<br>Negative', 'Predicted<br>Positive'],
        tickfont=dict(size=10), row=row, col=col)
    fig2.update_yaxes(tickvals=[0,1],
        ticktext=["Actual<br>NEGATIVE","Actual<br>POSITIVE"],
        tickfont=dict(size=10), row=row, col=col)

fig2.update_layout(
    title=dict(
        text="Step 9 — Confusion Matrix: What Did Each Model Predict?<br>"
             "<sup>TP = found PTM correctly | TN = found non-PTM correctly | FP = false alarm | FN = missed PTM</sup>",
        font=dict(size=17), x=0.5),
    plot_bgcolor="#f7f6f2", paper_bgcolor="#f9f8f5",
    height=560, margin=dict(t=120, b=60),
    font=dict(family="Arial", color="#28251d")
)
fig2.write_image("step9_confusion_matrices.png", scale=2)
print("✅ step9_confusion_matrices.png saved")


# ------------------------------------------------------------------
# Chart 3: Stacked bar — prediction breakdown per model
# ------------------------------------------------------------------
# Each bar = one model. Segments show how many TP, FP, TN, FN it made.
# More green area = more correct predictions.
# ------------------------------------------------------------------

fig3 = go.Figure()
cat_names  = ["True Positive (TP) — Correct PTM found",
              "False Positive (FP) — False alarm",
              "True Negative (TN) — Correct non-PTM",
              "False Negative (FN) — Missed PTM"]
cat_colors = ["#01696f", "#a13544", "#437a22", "#da7101"]

for cat, vals, col in zip(cat_names, [tp, fp, tn, fn], cat_colors):
    fig3.add_trace(go.Bar(
        name=cat, x=model_labels, y=vals,
        marker_color=col,
        text=[f"{v:,}" for v in vals],
        textposition="inside", textfont=dict(size=10, color="white"),
    ))

fig3.update_layout(
    title=dict(
        text="Step 9 — Prediction Breakdown per Model<br>"
             "<sup>More teal (TP) and green (TN) is better. More red (FP) and orange (FN) means more errors.</sup>",
        font=dict(size=17), x=0.5),
    barmode="stack",
    xaxis=dict(title="Model", tickfont=dict(size=12)),
    yaxis=dict(title="Number of Predictions", gridcolor="#dcd9d5"),
    plot_bgcolor="#f7f6f2", paper_bgcolor="#f9f8f5",
    legend=dict(font=dict(size=11), x=1.01, y=0.98,
        bgcolor="rgba(249,248,245,0.9)", bordercolor="#dcd9d5", borderwidth=1),
    height=480, margin=dict(t=120, b=70, r=280),
    font=dict(family="Arial", color="#28251d")
)
fig3.write_image("step9_prediction_breakdown.png", scale=2)
print("✅ step9_prediction_breakdown.png saved")


# ------------------------------------------------------------------
# Chart 4: Radar chart — model "shape"
# ------------------------------------------------------------------
# Bigger covered area = better overall performance.
# Boundary Uncertainty should cover the largest area.
# MCC is rescaled from [-1,1] to [0,1] just for display.
# ------------------------------------------------------------------

mcc_norm = [(v + 1) / 2 for v in mcc]
radar_data = [auroc, aupr, mcc_norm, f1]
cats = ["AUROC", "AUPR", "MCC (rescaled)", "F1 Score"]

fig4 = go.Figure()
for i, (name, col) in enumerate(zip(model_labels, colors)):
    vals = [radar_data[j][i] for j in range(4)] + [radar_data[0][i]]
    fig4.add_trace(go.Scatterpolar(
        r=vals, theta=cats + [cats[0]],
        fill="toself", name=name,
        line=dict(color=col, width=2.5),
        fillcolor=col, opacity=0.15
    ))

fig4.update_layout(
    title=dict(
        text="Step 9 — Model Shape: Strengths at a Glance<br>"
             "<sup>Bigger area = better across all metrics. Boundary Uncertainty covers the most area.</sup>",
        font=dict(size=17), x=0.5),
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10),
                        gridcolor="#dcd9d5", tickvals=[0.25, 0.5, 0.75, 1.0]),
        angularaxis=dict(tickfont=dict(size=12))
    ),
    showlegend=True,
    legend=dict(font=dict(size=12), x=1.05, y=0.95),
    paper_bgcolor="#f9f8f5",
    height=480, margin=dict(t=120, b=60, r=180),
    font=dict(family="Arial", color="#28251d")
)
fig4.write_image("step9_radar_chart.png", scale=2)
print("✅ step9_radar_chart.png saved")

print("\nAll 4 charts generated successfully!")