import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ------------------------------------------------------------------
# Chart 1: What is a Random Forest?
# ------------------------------------------------------------------
# A Random Forest is 100 small "decision trees", each making a
# prediction independently. The final answer is the MAJORITY vote.
# Example: 3 trees say PTM, 2 say NOT PTM → final answer = PTM.
# ------------------------------------------------------------------

fig1 = go.Figure()

# 5 simple trees as triangles
xs = [1, 2.5, 4, 5.5, 7]
votes = ["PTM", "NOT PTM", "PTM", "PTM", "NOT PTM"]
colors = ["#01696f", "#a13544", "#01696f", "#01696f", "#a13544"]

for i, (x, vote, color) in enumerate(zip(xs, votes, colors), start=1):
    fig1.add_shape(
        type="path",
        path=f"M {x} 0.9 L {x-0.4} 0.35 L {x+0.4} 0.35 Z",
        fillcolor=color,
        line=dict(color="white", width=1),
        layer="below"
    )
    fig1.add_shape(
        type="rect",
        x0=x-0.05, x1=x+0.05, y0=0.18, y1=0.35,
        fillcolor="#8B5E3C", line_width=0, layer="below"
    )
    fig1.add_annotation(
        x=x, y=0.05,
        text=f"<b>T{i}</b><br>{vote}",
        showarrow=False,
        font=dict(size=10, color=color),
        xanchor="center",
        align="center"
    )

# Majority vote box
fig1.add_shape(
    type="rect",
    x0=3.0, x1=5.0, y0=-0.32, y1=-0.08,
    fillcolor="#01696f", line=dict(color="white", width=2)
)
fig1.add_annotation(
    x=4.0, y=-0.20,
    text="<b>Final answer: PTM</b><br>3 out of 5 trees agree",
    showarrow=False,
    font=dict(size=11, color="white"),
    xanchor="center",
    align="center"
)

# fig1.add_annotation(
#     x=4.0, y=0.0,
#     text="Majority vote",
#     showarrow=False,
#     font=dict(size=11, color="#7a7974"),
#     xanchor="center"
# )

fig1.update_layout(
    title=dict(
        text="What is a Random Forest?<br><sup>Many small trees vote. The most common answer wins.</sup>",
        font=dict(size=17),
        x=0.5
    ),
    xaxis=dict(visible=False, range=[0.2, 7.8]),
    yaxis=dict(visible=False, range=[-0.5, 1.1]),
    plot_bgcolor="#f9f8f5",
    paper_bgcolor="#f9f8f5",
    height=320,
    margin=dict(t=90, b=30, l=20, r=20),
    font=dict(family="Arial", color="#28251d")
)

fig1.write_image("step8_what_is_random_forest.png", scale=2)
print("✅ step8_what_is_random_forest.png saved")


# ------------------------------------------------------------------
# Chart 2: Training data size per model
# ------------------------------------------------------------------
# All 4 models see the same positive PTM sites (27,090).
# The ONLY thing that changes is which negative (non-PTM) sites
# each model learns from. Boundary gets the most negatives (10,857).
# ------------------------------------------------------------------

models_short = ["Baseline", "Low\nUncertainty", "Boundary\nUncertainty", "High\nUncertainty"]
positives    = [27090, 27090, 27090, 27090]
negatives    = [4414,  4393,  10857, 4377]

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    name="PTM Sites (Positive)", x=models_short, y=positives,
    marker_color="#01696f",
    text=[f"{v:,}" for v in positives],
    textposition="outside", textfont=dict(size=12)
))
fig2.add_trace(go.Bar(
    name="Non-PTM Sites (Negative)", x=models_short, y=negatives,
    marker_color="#da7101",
    text=[f"{v:,}" for v in negatives],
    textposition="outside", textfont=dict(size=12)
))
for i, (p, n) in enumerate(zip(positives, negatives)):
    fig2.add_annotation(x=models_short[i], y=-3800,
        text=f"Ratio 1:{n/p:.1f}", showarrow=False,
        font=dict(size=10, color="#7a7974"), xanchor="center", yref="y")

fig2.update_layout(
    title=dict(
        text="Step 8 — Training Data: Positives vs Negatives Per Model<br>"
             "<sup>All models share the same 27,090 PTM sites. "
             "Boundary Uncertainty has the most negatives (10,857).</sup>",
        font=dict(size=17), x=0.5),
    barmode="group",
    xaxis=dict(title="Model", tickfont=dict(size=12)),
    yaxis=dict(title="Number of Training Sequences", gridcolor="#dcd9d5", range=[-5500, 34000]),
    plot_bgcolor="#f7f6f2", paper_bgcolor="#f9f8f5",
    legend=dict(font=dict(size=12), x=0.5, xanchor="center", y=-0.18, orientation="h"),
    height=490, margin=dict(t=120, b=100),
    font=dict(family="Arial", color="#28251d")
)
fig2.write_image("step8_training_data_per_model.png", scale=2)
print("✅ step8_training_data_per_model.png saved")


# ------------------------------------------------------------------
# Chart 3: Class balance comparison
# ------------------------------------------------------------------
# Shows how many PTM vs non-PTM sequences each model sees.
# A better balance (closer ratio) often leads to a better model.
# ------------------------------------------------------------------

fig3 = go.Figure()
fig3.add_trace(go.Bar(
    name="PTM Sites (Positive)", x=models_short, y=positives,
    marker_color="#01696f",
    text=[f"{v:,}" for v in positives],
    textposition="outside", textfont=dict(size=12)
))
fig3.add_trace(go.Bar(
    name="Non-PTM Sites (Negative)", x=models_short, y=negatives,
    marker_color="#da7101",
    text=[f"{v:,}" for v in negatives],
    textposition="outside", textfont=dict(size=12)
))

fig3.update_layout(
    title=dict(
        text="Step 8 — Class Balance: How Balanced Is Each Training Set?<br>"
             "<sup>Boundary Uncertainty is the most balanced (27,090 vs 10,857). "
             "Better balance = better model learning.</sup>",
        font=dict(size=17), x=0.5),
    barmode="group",
    xaxis=dict(title="Model", tickfont=dict(size=12)),
    yaxis=dict(title="Count", gridcolor="#dcd9d5"),
    plot_bgcolor="#f7f6f2", paper_bgcolor="#f9f8f5",
    legend=dict(font=dict(size=12), x=0.5, xanchor="center", y=-0.15, orientation="h"),
    height=460, margin=dict(t=120, b=90),
    font=dict(family="Arial", color="#28251d")
)
fig3.write_image("step8_class_balance.png", scale=2)
print("✅ step8_class_balance.png saved")


# ------------------------------------------------------------------
# Chart 4: Training pipeline — step-by-step vertical flow
# ------------------------------------------------------------------
# Shows the 5 steps that happen when training each model.
# Step 3 is the key step that differs across models.
# ------------------------------------------------------------------

steps_v = [
    ("1", "Load  train_scored_with_bins.csv", "#006494"),
    ("2", "Load protein embeddings (.npy)", "#006494"),
    ("3", "Select Positives + Negatives\n(by uncertainty bin — changes per model)", "#da7101"),
    ("4", "Train Random Forest  (100 trees)", "#01696f"),
    ("5", "Save model as .pkl file", "#01696f"),
]

fig4 = go.Figure()
for i, (num, label, col) in enumerate(steps_v):
    y = (len(steps_v) - 1 - i) * 1.4
    fig4.add_shape(type="rect", x0=0.5, x1=3.5, y0=y, y1=y+0.9,
        fillcolor=col, line=dict(color="white", width=2), layer="below")
    fig4.add_shape(type="circle", x0=0.5, x1=1.05, y0=y+0.15, y1=y+0.75,
        fillcolor="white", line=dict(color=col, width=2), layer="above")
    fig4.add_annotation(x=0.775, y=y+0.45, text=f"<b>{num}</b>",
        showarrow=False, font=dict(size=13, color=col), xanchor="center")
    fig4.add_annotation(x=2.2, y=y+0.45, text=f"<b>{label}</b>",
        showarrow=False, font=dict(size=11, color="white"),
        xanchor="center", yanchor="middle", align="center")
    if i < len(steps_v) - 1:
        fig4.add_annotation(x=2.0, y=y-0.2, text="↓",
            showarrow=False, font=dict(size=18, color="#7a7974"), xanchor="center")

fig4.add_annotation(x=2.0, y=-0.6,
    text="↑  This 5-step flow runs 4 times — once for each model  ↑",
    showarrow=False, font=dict(size=11, color="#7a7974"), xanchor="center")

fig4.update_layout(
    title=dict(
        text="Step 8 — Training Pipeline: 5 Steps Per Model<br>"
             "<sup>Only Step 3 changes across the 4 models — "
             "the negative sequences selected differ by uncertainty bin.</sup>",
        font=dict(size=17), x=0.5),
    xaxis=dict(visible=False, range=[0.2, 3.8]),
    yaxis=dict(visible=False, range=[-1.0, 7.5]),
    plot_bgcolor="#f9f8f5", paper_bgcolor="#f9f8f5",
    height=540, margin=dict(t=110, b=50, l=20, r=20),
    font=dict(family="Arial", color="#28251d")
)
fig4.write_image("step8_training_pipeline.png", scale=2)
print("✅ step8_training_pipeline.png saved")

print("\nAll 4 charts generated successfully!")

