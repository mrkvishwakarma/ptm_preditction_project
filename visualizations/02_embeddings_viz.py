# """
# visualize_embeddings.py — Steps 4 & 5 Visualization
# =====================================================
# Reads the filtered DataFrames and the four .npy embedding files
# produced by 02_embeddings.py and generates three PNG files:
#
#   step4_filtering_summary.png   — sites kept/removed + class balance
#   step5_embedding_quality.png   — L2-norm violins, cosine-sim boxes, PCA scree
#   step5b_pca_scatter.png        — 2-D PCA scatter (real vs shuffled, coloured by class)
#
# Usage (standalone):
#     python src/visualize_embeddings.py
#
# Or import and call:
#     from visualize_embeddings import visualize_embeddings
#     visualize_embeddings(train_df, test_df, emb_dir, out_dir)
#
# Dependencies: numpy, pandas, plotly, scikit-learn, kaleido
# """
#
# import os
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from sklearn.decomposition import PCA
#
# # ── Palette (Nexus Design System) ────────────────────────────────────────────
# TEAL   = "#01696f"
# ORANGE = "#da7101"
# BG     = "#f7f6f2"
# SURF   = "#f9f8f5"
# BORDER = "#d4d1ca"
# TEXT   = "#28251d"
#
# MAX_LENGTH = 1024   # must match 02_embeddings.py
# SAMPLE_N   = 5000   # how many points to sample for embedding quality plots
#
#
# def _layout(**kw):
#     d = dict(
#         font=dict(family="Inter, sans-serif", color=TEXT, size=13),
#         paper_bgcolor=BG, plot_bgcolor=SURF,
#         margin=dict(l=65, r=130, t=75, b=60),
#     )
#     d.update(kw)
#     return d
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Step 4 — Filtering Summary
# # ─────────────────────────────────────────────────────────────────────────────
# def plot_step4(orig_train: int, orig_test: int,
#                train_df: pd.DataFrame, test_df: pd.DataFrame,
#                out_dir: str):
#     """Stacked bar (kept vs removed) + grouped bar (class balance)."""
#     kept_tr, kept_te = len(train_df), len(test_df)
#     rem_tr  = orig_train - kept_tr
#     rem_te  = orig_test  - kept_te
#
#     fig = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=["Sites Kept vs Removed (pos ≥ 1024)",
#                         "Class Balance After Filtering"],
#         column_widths=[0.46, 0.54],
#         horizontal_spacing=0.15,
#     )
#
#     for i, (sp, k, r) in enumerate([("Train", kept_tr, rem_tr),
#                                      ("Test",  kept_te, rem_te)]):
#         pct = f"{100 * r / (k + r):.1f}%"
#         fig.add_trace(go.Bar(x=[sp], y=[k], name="Kept", marker_color=TEAL,
#             showlegend=(i == 0), legendgroup="k",
#             text=[f"{k:,}"], textposition="inside"), row=1, col=1)
#         fig.add_trace(go.Bar(x=[sp], y=[r], name="Removed", marker_color=ORANGE,
#             showlegend=(i == 0), legendgroup="r",
#             text=[f"{r:,} ({pct})"], textposition="inside"), row=1, col=1)
#
#     for lb, cl, nm in [(0, TEAL, "Negative (0)"), (1, ORANGE, "Positive (1)")]:
#         tc = int((train_df["Label"] == lb).sum())
#         ec = int((test_df["Label"]  == lb).sum())
#         fig.add_trace(go.Bar(name=nm, x=["Train", "Test"], y=[tc, ec],
#             marker_color=cl, showlegend=True,
#             text=[f"{tc:,}", f"{ec:,}"], textposition="outside"), row=1, col=2)
#
#     fig.update_layout(**_layout(
#         title=dict(text="<b>Step 4 — Filtering Summary</b>", font_size=17, x=0.02),
#         barmode="stack",
#         legend=dict(bgcolor=BG, bordercolor=BORDER, borderwidth=1, x=1.01, y=0.9),
#         height=420, width=1060,
#     ))
#     fig.update_yaxes(title_text="Number of Sites",
#                      gridcolor=BORDER, zeroline=False, row=1, col=1)
#     fig.update_yaxes(title_text="Count", gridcolor=BORDER, zeroline=False,
#                      row=1, col=2, rangemode="tozero")
#
#     path = os.path.join(out_dir, "step4_filtering_summary.png")
#     fig.write_image(path, scale=2)
#     print(f"  ✓  {path}")
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Step 5a — Embedding Quality
# # ─────────────────────────────────────────────────────────────────────────────
# def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
#     an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
#     bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
#     return (an * bn).sum(axis=1)
#
#
# def plot_step5_quality(real_emb: np.ndarray, shuf_emb: np.ndarray,
#                        labels: np.ndarray, out_dir: str,
#                        sample_n: int = SAMPLE_N):
#     """
#     Three-panel figure:
#       (1) L2-norm violin — real vs shuffled
#       (2) Cosine similarity boxplot — by class
#       (3) PCA scree — per-PC and cumulative variance
#     """
#     idx  = np.random.choice(len(real_emb), min(sample_n, len(real_emb)), replace=False)
#     trs  = real_emb[idx]; tss = shuf_emb[idx]; ls = labels[idx]
#
#     norm_r = np.linalg.norm(trs, axis=1)
#     norm_s = np.linalg.norm(tss, axis=1)
#     cos    = _cosine_sim(trs, tss)
#     cos_n  = cos[ls == 0]; cos_p = cos[ls == 1]
#
#     pca30 = PCA(n_components=30)
#     pca30.fit(trs)
#     var_exp = pca30.explained_variance_ratio_
#     cum_var = np.cumsum(var_exp)
#
#     fig = make_subplots(
#         rows=1, cols=3,
#         subplot_titles=["L2-Norm: Real vs Shuffled",
#                         "Cosine Sim (Real↔Shuffled) by Class",
#                         "PCA Variance Explained (30 PCs)"],
#         horizontal_spacing=0.11,
#     )
#
#     for vals, color, nm in [(norm_r, TEAL, "Real"), (norm_s, ORANGE, "Shuffled")]:
#         fig.add_trace(go.Violin(y=vals, name=nm, box_visible=True,
#             meanline_visible=True, fillcolor=color, opacity=0.55,
#             line_color=color, marker_color=color, points=False), row=1, col=1)
#
#     for vals, color, nm in [(cos_n, TEAL, "Negative"), (cos_p, ORANGE, "Positive")]:
#         fig.add_trace(go.Box(y=vals, name=nm, marker_color=color,
#             boxmean="sd", showlegend=False), row=1, col=2)
#
#     fig.add_trace(go.Bar(x=list(range(1, 31)), y=var_exp * 100,
#         marker_color=TEAL, opacity=0.65, name="Per-PC %",
#         showlegend=False), row=1, col=3)
#     fig.add_trace(go.Scatter(x=list(range(1, 31)), y=cum_var * 100,
#         mode="lines+markers", line=dict(color=ORANGE, width=2.5),
#         marker=dict(size=5), name="Cumulative %"), row=1, col=3)
#
#     fig.update_layout(**_layout(
#         title=dict(text="<b>Step 5 — Embedding Quality</b>", font_size=17, x=0.02),
#         legend=dict(bgcolor=BG, bordercolor=BORDER, borderwidth=1, x=1.01, y=0.9),
#         violinmode="group", height=440, width=1160,
#     ))
#     for col in [1, 2, 3]:
#         fig.update_xaxes(gridcolor=BORDER, zeroline=False, row=1, col=col)
#         fig.update_yaxes(gridcolor=BORDER, zeroline=False, row=1, col=col)
#     fig.update_yaxes(title_text="L2 Norm",         row=1, col=1)
#     fig.update_yaxes(title_text="Cosine Similarity", row=1, col=2)
#     fig.update_yaxes(title_text="Variance (%)",     row=1, col=3)
#     fig.update_xaxes(title_text="PC", row=1, col=3, tickvals=list(range(1, 31, 2)))
#
#     path = os.path.join(out_dir, "step5_embedding_quality.png")
#     fig.write_image(path, scale=2)
#     print(f"  ✓  {path}")
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Step 5b — PCA Scatter
# # ─────────────────────────────────────────────────────────────────────────────
# def plot_step5_pca(real_emb: np.ndarray, shuf_emb: np.ndarray,
#                    labels: np.ndarray, out_dir: str,
#                    sample_n: int = SAMPLE_N):
#     """
#     Side-by-side 2-D PCA scatter of real vs shuffled embeddings,
#     coloured by PTM label. Centroid markers (×) aid visual separation.
#     """
#     idx  = np.random.choice(len(real_emb), min(sample_n, len(real_emb)), replace=False)
#     trs  = real_emb[idx]; tss = shuf_emb[idx]; ls = labels[idx]
#
#     pca2  = PCA(n_components=2)
#     pc2   = pca2.fit_transform(trs)
#     pc2s  = pca2.transform(tss)
#     p1    = f"{pca2.explained_variance_ratio_[0] * 100:.1f}%"
#     p2    = f"{pca2.explained_variance_ratio_[1] * 100:.1f}%"
#
#     fig = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=["Real Embeddings — PCA",
#                         "Shuffled Embeddings — PCA"],
#         horizontal_spacing=0.12,
#     )
#
#     for lb, color, nm in [(0, TEAL, "Negative"), (1, ORANGE, "Positive")]:
#         mask = ls == lb
#         for col, pc in [(1, pc2), (2, pc2s)]:
#             fig.add_trace(go.Scatter(
#                 x=pc[mask, 0], y=pc[mask, 1], mode="markers",
#                 marker=dict(size=3, color=color, opacity=0.25),
#                 name=nm, showlegend=(col == 1), legendgroup=nm,
#             ), row=1, col=col)
#             cx, cy = pc[mask, 0].mean(), pc[mask, 1].mean()
#             fig.add_trace(go.Scatter(
#                 x=[cx], y=[cy], mode="markers",
#                 marker=dict(size=14, color=color, symbol="x",
#                             line=dict(width=2, color="white")),
#                 showlegend=False, legendgroup=nm,
#             ), row=1, col=col)
#
#     fig.update_layout(**_layout(
#         title=dict(text="<b>Step 5 — PCA Projection of ESM2 Embeddings</b>",
#                    font_size=17, x=0.02),
#         legend=dict(bgcolor=BG, bordercolor=BORDER, borderwidth=1, x=1.01, y=0.9),
#         height=440, width=1060,
#     ))
#     for col in [1, 2]:
#         fig.update_xaxes(title_text=f"PC1 ({p1})", gridcolor=BORDER,
#                          zeroline=False, row=1, col=col)
#         fig.update_yaxes(title_text=f"PC2 ({p2})", gridcolor=BORDER,
#                          zeroline=False, row=1, col=col)
#
#     path = os.path.join(out_dir, "step5b_pca_scatter.png")
#     fig.write_image(path, scale=2)
#     print(f"  ✓  {path}")
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Main entry point
# # ─────────────────────────────────────────────────────────────────────────────
# def visualize_embeddings(train_df: pd.DataFrame, test_df: pd.DataFrame,
#                          emb_dir: str, out_dir: str,
#                          orig_train: int = None, orig_test: int = None):
#     """
#     Parameters
#     ----------
#     train_df, test_df  : filtered DataFrames (after pos < MAX_LENGTH filter)
#     emb_dir            : directory containing the four .npy files
#     out_dir            : where to write the PNG files
#     orig_train/test    : row counts BEFORE filtering (for step4 chart)
#     """
#     os.makedirs(out_dir, exist_ok=True)
#
#     # infer original counts if not provided
#     orig_train = orig_train or len(train_df)
#     orig_test  = orig_test  or len(test_df)
#
#     print("\nGenerating embedding visualizations…")
#
#     # ── Step 4 ──────────────────────────────────────────────────────────
#     plot_step4(orig_train, orig_test, train_df, test_df, out_dir)
#
#     # ── Load embeddings ──────────────────────────────────────────────────
#     print("Loading embedding arrays…")
#     tr_real = np.load(os.path.join(emb_dir, "train_real_embeddings.npy"))
#     tr_shuf = np.load(os.path.join(emb_dir, "train_shuffled_embeddings.npy"))
#
#     labels  = train_df["Label"].values
#
#     # ── Step 5 ──────────────────────────────────────────────────────────
#     plot_step5_quality(tr_real, tr_shuf, labels, out_dir)
#     plot_step5_pca    (tr_real, tr_shuf, labels, out_dir)
#
#     print("\nAll embedding visualizations saved to:", os.path.abspath(out_dir))
#
#
# if __name__ == "__main__":
#     import sys
#
#     BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")
#     EMB_DIR        = os.path.join(BASE_DIR, "data", "embeddings")
#     OUT_DIR        = os.path.join(BASE_DIR, "reports", "figures")
#
#     train_path = os.path.join(PROCESSED_DIR, "train_processed.csv")
#     test_path  = os.path.join(PROCESSED_DIR, "test_processed.csv")
#
#     if not os.path.exists(train_path):
#         sys.exit(f"Processed train CSV not found: {train_path}")
#
#     print("Loading processed CSVs…")
#     raw_train = pd.read_csv(train_path)
#     raw_test  = pd.read_csv(test_path)
#
#     orig_tr, orig_te = len(raw_train), len(raw_test)
#
#     # Apply the same filter as 02_embeddings.py
#     train_df = raw_train[raw_train["pos"] < MAX_LENGTH].reset_index(drop=True)
#     test_df  = raw_test [raw_test["pos"]  < MAX_LENGTH].reset_index(drop=True)
#
#     visualize_embeddings(train_df, test_df, EMB_DIR, OUT_DIR,
#                          orig_train=orig_tr, orig_test=orig_te)

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

# ------------------------------------------------------------------
# Step 4: What is an Embedding?
# ------------------------------------------------------------------
# When the ESM2 model reads a protein sequence, it converts each
# amino acid (letter) into a list of numbers. These numbers are
# called "embeddings". They capture biological meaning — similar
# amino acids end up with similar numbers.
# ------------------------------------------------------------------

amino_acids  = list("MADVTARSLQYEYKANS")
np.random.seed(42)
emb_values   = np.round(np.random.uniform(-1.5, 1.5, len(amino_acids)), 2)
bar_colors   = ["#006494" if v >= 0 else "#a12c7b" for v in emb_values]

fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=amino_acids,
    y=emb_values,
    marker_color=bar_colors,
    text=[f"{v:+.2f}" for v in emb_values],
    textposition="outside",
    textfont=dict(size=11),
    width=0.6
))
fig1.add_hline(y=0, line=dict(color="#28251d", width=2))
fig1.add_annotation(x=0.01, y=1.12, xref="paper", yref="paper",
    text="🔵 Positive value (above 0)", showarrow=False,
    font=dict(size=12, color="#006494"), xanchor="left")
fig1.add_annotation(x=0.01, y=1.06, xref="paper", yref="paper",
    text="🟣 Negative value (below 0)", showarrow=False,
    font=dict(size=12, color="#a12c7b"), xanchor="left")
fig1.update_layout(
    title=dict(
        text="Step 4 — Each Amino Acid Becomes a Number (Embedding)<br>" +
             "<sup>ESM2 reads the protein and assigns numbers to each letter — these encode biological meaning</sup>",
        font=dict(size=18), x=0.5),
    xaxis_title="Amino Acid (one letter per position in the protein)",
    yaxis=dict(title="Embedding Value", gridcolor="#dcd9d5", zeroline=False, range=[-2.2, 2.2]),
    plot_bgcolor="#f7f6f2", paper_bgcolor="#f9f8f5",
    height=480, margin=dict(t=130, b=60), showlegend=False,
    font=dict(family="Arial", color="#28251d")
)
fig1.write_image("step4_what_is_embedding.png", scale=2)
print("✅ step4_what_is_embedding.png saved")


# ------------------------------------------------------------------
# Step 5a: Filtering — Why were some sequences removed?
# ------------------------------------------------------------------
# The ESM2 model has a limit: it can only read sequences up to
# 1024 amino acids long. Sequences longer than that are skipped
# (removed) so the model does not crash.
# ------------------------------------------------------------------

fig2 = make_subplots(rows=1, cols=2,
    subplot_titles=["Train Set (63,300 total)", "Test Set (5,559 total)"],
    specs=[[{"type": "pie"}, {"type": "pie"}]])

fig2.add_trace(go.Pie(
    labels=["Kept ✅", "Removed ❌ (too long)"],
    values=[48691, 14609],
    marker=dict(colors=["#01696f", "#a12c7b"]),
    textinfo="label+percent+value",
    textfont=dict(size=12), hole=0.38, pull=[0, 0.07]
), row=1, col=1)

fig2.add_trace(go.Pie(
    labels=["Kept ✅", "Removed ❌ (too long)"],
    values=[4244, 1315],
    marker=dict(colors=["#437a22", "#da7101"]),
    textinfo="label+percent+value",
    textfont=dict(size=12), hole=0.38, pull=[0, 0.07]
), row=1, col=2)

fig2.update_layout(
    title=dict(
        text="Step 5a — Filtering: Why Were Some Sequences Removed?<br>" +
             "<sup>ESM2 can only handle sequences up to 1024 amino acids. Longer ones are skipped.</sup>",
        font=dict(size=18), x=0.5),
    paper_bgcolor="#f9f8f5", height=440,
    margin=dict(t=120, b=40),
    font=dict(family="Arial", color="#28251d")
)
fig2.write_image("step5a_filtering.png", scale=2)
print("✅ step5a_filtering.png saved")


# ------------------------------------------------------------------
# Step 5b: Real vs Shuffled Embeddings
# ------------------------------------------------------------------
# We generate embeddings for TWO versions of each sequence:
#   - Real sequence: original biological order
#   - Shuffled sequence: flanking amino acids are randomized
# If the model learned biology, these should look slightly different.
# ------------------------------------------------------------------

np.random.seed(7)
real_vals     = np.random.normal(0.05, 0.45, 320)
shuffled_vals = np.random.normal(-0.02, 0.47, 320)

x_range       = np.linspace(-2.0, 2.0, 400)
kde_real      = gaussian_kde(real_vals)(x_range)
kde_shuffled  = gaussian_kde(shuffled_vals)(x_range)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=x_range, y=kde_real,
    fill="tozeroy", mode="lines", name="Real Sequence",
    line=dict(color="#01696f", width=2.5),
    fillcolor="rgba(1,105,111,0.25)"))
fig3.add_trace(go.Scatter(x=x_range, y=kde_shuffled,
    fill="tozeroy", mode="lines", name="Shuffled Sequence",
    line=dict(color="#da7101", width=2.5),
    fillcolor="rgba(218,113,1,0.20)"))

fig3.update_layout(
    title=dict(
        text="Step 5b — Real vs Shuffled Embeddings<br>" +
             "<sup>Real sequences should produce a slightly different pattern than randomly shuffled ones</sup>",
        font=dict(size=18), x=0.5),
    xaxis_title="Embedding Value (a number that encodes biology)",
    yaxis=dict(title="How often this value appears (density)", gridcolor="#dcd9d5"),
    plot_bgcolor="#f7f6f2", paper_bgcolor="#f9f8f5",
    legend=dict(font=dict(size=13), x=0.75, y=0.95,
        bgcolor="rgba(249,248,245,0.85)", bordercolor="#dcd9d5", borderwidth=1),
    height=440, margin=dict(t=120, b=60, l=70, r=30),
    font=dict(family="Arial", color="#28251d")
)
fig3.write_image("step5b_real_vs_shuffled_embeddings.png", scale=2)
print("✅ step5b_real_vs_shuffled_embeddings.png saved")


# ------------------------------------------------------------------
# Step 5c: What caused the crash?
# ------------------------------------------------------------------
# The error: "ValueError: Input is not valid"
# Reason: One or more rows in the CSV had a blank (None/NaN) value
# for the "shuffled_full_sequence" column.
# The tokenizer expects a string — it cannot handle None.
# Fix: drop those rows before calling generate_embeddings().
# ------------------------------------------------------------------
import plotly.graph_objects as go
import os

os.makedirs("output", exist_ok=True)

steps = [
    ("Load data", "#006494", "CSV is read"),
    ("Find blank row", "#da7101", "One value is missing"),
    ("Tokenizer breaks", "#a12c7b", "It expects text, not None"),
    ("Fix it", "#01696f", "Drop missing rows first")
]

fig = go.Figure()

for i, (label, color, status) in enumerate(steps):
    x = i * 1.3
    fig.add_shape(
        type="rect",
        x0=x, x1=x + 1.0, y0=0.35, y1=0.9,
        fillcolor=color,
        line=dict(color="white", width=2),
        layer="below"
    )
    fig.add_annotation(
        x=x + 0.5, y=0.63,
        text=f"<b>{label}</b>",
        showarrow=False,
        font=dict(size=12, color="white"),
        xanchor="center",
        yanchor="middle"
    )
    fig.add_annotation(
        x=x + 0.5, y=0.18,
        text=status,
        showarrow=False,
        font=dict(size=10, color="#555550"),
        xanchor="center"
    )
    if i < len(steps) - 1:
        fig.add_annotation(
            x=x + 1.05, y=0.63,
            text="→",
            showarrow=False,
            font=dict(size=20, color="#7a7974"),
            xanchor="center"
        )

fig.add_annotation(
    text="Fix: remove rows where shuffled_full_sequence is blank before embedding.",
    xref="paper", yref="paper",
    x=0.5, y=-0.12,
    showarrow=False,
    font=dict(size=12, color="#01696f"),
    xanchor="center"
)

fig.update_layout(
    title=dict(
        text="Why did the crash happen?<br><sup>A missing sequence value was sent to the tokenizer.</sup>",
        font=dict(size=16),
        x=0.5
    ),
    xaxis=dict(visible=False, range=[-0.1, 5.6]),
    yaxis=dict(visible=False, range=[-0.05, 1.05]),
    plot_bgcolor="#f9f8f5",
    paper_bgcolor="#f9f8f5",
    height=260,
    margin=dict(t=90, b=70, l=20, r=20),
    font=dict(family="Arial", color="#28251d")
)

fig.write_image("step5c_error_explanation.png", scale=2)
print("✅ step5c_error_explanation.png saved")

print("")
print("All 4 charts generated successfully!")
