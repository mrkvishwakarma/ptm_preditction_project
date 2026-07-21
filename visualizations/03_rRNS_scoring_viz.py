import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

np.random.seed(42)


# ------------------------------------------------------------------
# Step 6a: What is the Latent Space?
# ------------------------------------------------------------------
# We take all negative site embeddings (real + shuffled) and place
# them into a shared "map". Similar sequences end up close together.
# For each real negative site (★), we find the 10 closest points
# and count how many are shuffled — that gives the rRNS score.
# ------------------------------------------------------------------

n = 500
real_x  = np.random.normal( 1.5, 1.0, n)
real_y  = np.random.normal( 1.5, 1.0, n)
shuff_x = np.random.normal(-1.0, 1.2, n)
shuff_y = np.random.normal(-0.5, 1.2, n)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=real_x,  y=real_y,  mode="markers", name="Real Sequences",
    marker=dict(color="#01696f", size=5, opacity=0.55)))
fig1.add_trace(go.Scatter(x=shuff_x, y=shuff_y, mode="markers", name="Shuffled Sequences",
    marker=dict(color="#da7101", size=5, opacity=0.55)))

qx, qy = 1.0, 0.8
nb_x = np.concatenate([np.random.normal(qx, 0.4, 7), np.random.normal(qx-1.5, 0.5, 3)])
nb_y = np.concatenate([np.random.normal(qy, 0.4, 7), np.random.normal(qy-0.8, 0.5, 3)])

fig1.add_trace(go.Scatter(x=[qx], y=[qy], mode="markers",
    name="Query: one negative site",
    marker=dict(color="#a12c7b", size=14, symbol="star")))
for nx, ny in zip(nb_x, nb_y):
    fig1.add_shape(type="line", x0=qx, y0=qy, x1=nx, y1=ny,
        line=dict(color="#7a7974", width=1, dash="dot"))
fig1.add_trace(go.Scatter(x=nb_x, y=nb_y, mode="markers",
    name="Its 10 nearest neighbors",
    marker=dict(color="#fdab43", size=9, symbol="circle-open", line=dict(width=2))))

fig1.update_layout(
    title=dict(
        text="Step 6 — The Latent Space: Real vs Shuffled Sequences<br>"
             "<sup>Each dot = one protein. The star (★) is a query negative site. "
             "Lines connect it to its 10 nearest neighbors.</sup>",
        font=dict(size=18), x=0.5),
    xaxis=dict(title="Dimension 1 (biological similarity)", showgrid=True, gridcolor="#dcd9d5"),
    yaxis=dict(title="Dimension 2 (biological similarity)", showgrid=True, gridcolor="#dcd9d5"),
    plot_bgcolor="#f7f6f2", paper_bgcolor="#f9f8f5",
    legend=dict(font=dict(size=12), x=0.01, y=0.99,
        bgcolor="rgba(249,248,245,0.9)", bordercolor="#dcd9d5", borderwidth=1),
    height=500, margin=dict(t=130, b=60, l=70, r=30),
    font=dict(family="Arial", color="#28251d")
)
fig1.write_image("step6_latent_space.png", scale=2)
print("✅ step6_latent_space.png saved")


# ------------------------------------------------------------------
# Step 6b: What is the rRNS Score?
# ------------------------------------------------------------------
# For each negative site, look at its 10 nearest neighbors.
# rRNS = (number of shuffled neighbors) / 10
#
#   Low  score (≈0.0) → mostly REAL neighbors → trustworthy negative
#   High score (≈1.0) → mostly SHUFFLED neighbors → suspicious negative
# ------------------------------------------------------------------

cases = [
    ("Low rRNS = 0.1\n→ Confident Negative",  9, 1),
    ("Boundary rRNS = 0.4\n→ Uncertain",        6, 4),
    ("High rRNS = 0.9\n→ Suspicious Negative",  1, 9),
]

fig2 = go.Figure()
for i, (label, n_real, n_shuff) in enumerate(cases):
    y_pos = i * 1.5
    for j in range(n_real):
        fig2.add_shape(type="rect", x0=j, x1=j+0.85, y0=y_pos, y1=y_pos+0.7,
            fillcolor="#01696f", line=dict(color="white", width=1.5), layer="below")
    for j in range(n_real, 10):
        fig2.add_shape(type="rect", x0=j, x1=j+0.85, y0=y_pos, y1=y_pos+0.7,
            fillcolor="#fdab43", line=dict(color="white", width=1.5), layer="below")
    fig2.add_annotation(x=-0.3, y=y_pos+0.35, text=f"<b>{label}</b>",
        showarrow=False, font=dict(size=12, color="#28251d"),
        xanchor="right", yanchor="middle", align="right")

fig2.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
    marker=dict(color="#01696f", size=14, symbol="square"),
    name="Real neighbor (biological)"))
fig2.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
    marker=dict(color="#fdab43", size=14, symbol="square"),
    name="Shuffled neighbor (random)"))

fig2.update_layout(
    title=dict(
        text="Step 6b — How is the rRNS Score Calculated?<br>"
             "<sup>Each block = one of the 10 nearest neighbors. "
             "rRNS = (# orange blocks) / 10</sup>",
        font=dict(size=17), x=0.5),
    xaxis=dict(title="Neighbor #1 → #10", range=[-3.5, 10.5],
               tickvals=list(range(10)), ticktext=[str(i+1) for i in range(10)]),
    yaxis=dict(visible=False, range=[-0.5, 4.0]),
    plot_bgcolor="#f9f8f5", paper_bgcolor="#f9f8f5",
    height=380, margin=dict(t=120, b=70, l=230, r=30),
    legend=dict(font=dict(size=12), x=0.5, xanchor="center", y=-0.18, orientation="h"),
    font=dict(family="Arial", color="#28251d")
)
fig2.write_image("step6b_rrns_explained.png", scale=2)
print("✅ step6b_rrns_explained.png saved")


# ------------------------------------------------------------------
# Step 7a: Bin Distribution
# ------------------------------------------------------------------
# After calculating rRNS for all 21,599 negative sites, each one
# is placed into a "bin" based on its score.
# Positive sites (Label=1) keep their own bin automatically.
# ------------------------------------------------------------------

bins   = ["Positive", "Boundary", "Low", "High", "Unassigned"]
counts = [27090,       10857,      4393,  4377,   1972]
colors_bins = ["#01696f", "#da7101", "#437a22", "#a12c7b", "#7a7974"]
descriptions = [
    "Known PTM sites\n(always kept)",
    "Mixed neighbors\n(uncertain)",
    "Reliable negatives\n(safe to use)",
    "Suspicious negatives\n(likely not real)",
    "In-between scores\n(not assigned)"
]

fig3 = go.Figure()
fig3.add_trace(go.Bar(
    x=bins, y=counts, marker_color=colors_bins,
    text=[f"{c:,}" for c in counts],
    textposition="outside", textfont=dict(size=13), width=0.55
))

fig3.update_layout(
    title=dict(
        text="Step 7 — Uncertainty Bins: How Are Negative Sites Categorised?<br>"
             "<sup>Based on their rRNS score, each negative is placed into a bucket — "
             "this helps decide which negatives are safe to train on.</sup>",
        font=dict(size=17), x=0.5),
    xaxis=dict(title="Uncertainty Bin", tickfont=dict(size=13)),
    yaxis=dict(title="Number of Sequences", gridcolor="#dcd9d5", range=[0, 32000]),
    plot_bgcolor="#f7f6f2", paper_bgcolor="#f9f8f5",
    height=480, margin=dict(t=120, b=100),
    font=dict(family="Arial", color="#28251d"), showlegend=False
)
for b, desc in zip(bins, descriptions):
    fig3.add_annotation(x=b, y=-2500, text=desc, showarrow=False,
        font=dict(size=10, color="#7a7974"), align="center", yref="y")
fig3.write_image("step7_bin_distribution.png", scale=2)
print("✅ step7_bin_distribution.png saved")


# ------------------------------------------------------------------
# Step 7b: rRNS Score Thresholds
# ------------------------------------------------------------------
# The score axis (0 to 1) is divided into regions by fixed thresholds.
# The histogram shows where most negative sites landed.
# ------------------------------------------------------------------

np.random.seed(5)
scores = np.concatenate([
    np.random.beta(1.2, 8, 4393),
    np.random.uniform(0.10, 0.30, 1972),
    np.random.beta(4, 4, 10857) * 0.3 + 0.30,
    np.random.beta(8, 1.5, 4377) * 0.4 + 0.60
])

fig4 = go.Figure()
region_defs = [
    (0.0,  0.10, "rgba(67,122,34,0.18)"),
    (0.10, 0.30, "rgba(122,121,116,0.12)"),
    (0.30, 0.60, "rgba(218,113,1,0.18)"),
    (0.60, 1.00, "rgba(161,44,123,0.18)"),
]
for x0, x1, col in region_defs:
    fig4.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=4.5,
        fillcolor=col, line_width=0, layer="below")

fig4.add_trace(go.Histogram(x=scores, nbinsx=60,
    marker_color="#006494", opacity=0.75,
    histnorm="probability density"))

for thresh in [0.10, 0.30, 0.60]:
    fig4.add_vline(x=thresh, line=dict(color="#28251d", width=1.5, dash="dash"))

for xpos, lbl in [(0.05, "LOW\n(Safe)"), (0.20, "UNASSIGNED"),
                   (0.45, "BOUNDARY\n(Uncertain)"), (0.80, "HIGH\n(Suspicious)")]:
    fig4.add_annotation(x=xpos, y=4.2, text=f"<b>{lbl}</b>", showarrow=False,
        font=dict(size=11, color="#28251d"), xanchor="center", align="center")

fig4.update_layout(
    title=dict(
        text="Step 7b — rRNS Score Thresholds<br>"
             "<sup>Score ≤ 0.10 → Low (confident) | 0.30–0.60 → Boundary | ≥ 0.60 → High (suspicious)</sup>",
        font=dict(size=17), x=0.5),
    xaxis=dict(title="rRNS Score  (0 = all real neighbors  →  1 = all shuffled neighbors)",
               tickfont=dict(size=12), range=[0, 1]),
    yaxis=dict(title="Density (how many sites scored here)", gridcolor="#dcd9d5", range=[0, 4.8]),
    plot_bgcolor="#f7f6f2", paper_bgcolor="#f9f8f5",
    height=460, margin=dict(t=120, b=70, l=70, r=30),
    font=dict(family="Arial", color="#28251d"), showlegend=False
)
fig4.write_image("step7b_rrns_thresholds.png", scale=2)
print("✅ step7b_rrns_thresholds.png saved")

print("\nAll 4 charts generated successfully!")

