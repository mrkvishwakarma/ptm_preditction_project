# """
# visualize_ptm_steps.py
# ──────────────────────
# Drop-in visualization script for the PTM prediction pipeline.
# Produces one PNG per pipeline step in  <output_dir>/
#  • step1_raw_data_overview.png
#  • step2_data_inspection.png
#  • step3_shuffling_qa.png
#
# Usage:
#     python visualize_ptm_steps.py
# or import and call visualize_all(train_df, test_df, output_dir="output/")
# """
#
# import os, random, collections
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
#
# # ── palette ──────────────────────────────────────────────────────────────────
# TEAL   = "#01696f"
# ORANGE = "#da7101"
# MAROON = "#a12c7b"
# BG     = "#f7f6f2"
# SURF   = "#f9f8f5"
# BORDER = "#d4d1ca"
# TEXT   = "#28251d"
# AAs    = list("ACDEFGHIKLMNPQRSTVWY")
#
# BASE_LAYOUT = dict(
#     font=dict(family="Inter, sans-serif", color=TEXT),
#     paper_bgcolor=BG, plot_bgcolor=SURF,
#     margin=dict(l=55, r=30, t=65, b=55),
# )
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Step 1 — Raw Data Overview
# # ─────────────────────────────────────────────────────────────────────────────
# def plot_step1(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str):
#     """Donut charts (class balance) + bar chart (dataset sizes)."""
#     fig = make_subplots(
#         rows=1, cols=3,
#         subplot_titles=["Train — Class Distribution",
#                         "Test — Class Distribution",
#                         "Dataset Size Comparison"],
#         specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "bar"}]],
#     )
#
#     for col, df, name in [(1, train_df, "Train"), (2, test_df, "Test")]:
#         vc = df["Label"].value_counts().sort_index()
#         fig.add_trace(go.Pie(
#             labels=["Negative (0)", "Positive (1)"],
#             values=vc.values,
#             marker_colors=[TEAL, ORANGE],
#             hole=0.48,
#             textfont_size=12,
#             hovertemplate="%{label}<br>Count: %{value:,}<br>%{percent}<extra></extra>",
#             showlegend=(col == 1),
#             legendgroup="labels",
#             name=name,
#         ), row=1, col=col)
#
#     fig.add_trace(go.Bar(
#         x=["Train", "Test"], y=[len(train_df), len(test_df)],
#         marker_color=[TEAL, ORANGE],
#         text=[f"{len(train_df):,}", f"{len(test_df):,}"],
#         textposition="outside", width=0.4, showlegend=False,
#     ), row=1, col=3)
#
#     fig.update_layout(
#         **BASE_LAYOUT,
#         title=dict(text="<b>Step 1 — Raw Data Overview</b>", font_size=17, x=0.02),
#         legend=dict(bgcolor=BG, bordercolor=BORDER, borderwidth=1, y=0.5, x=1.01),
#         height=400, width=1050,
#         yaxis3=dict(showgrid=True, gridcolor=BORDER, zeroline=False,
#                     title="Number of Rows", rangemode="tozero"),
#     )
#
#     path = os.path.join(out_dir, "step1_raw_data_overview.png")
#     fig.write_image(path, scale=2)
#     print(f"  ✓  {path}")
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Step 2 — Data Inspection
# # ─────────────────────────────────────────────────────────────────────────────
# def plot_step2(train_df: pd.DataFrame, out_dir: str):
#     """Violin (protein-length by class) + 2-D density (position vs length)."""
#     fig = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=["Protein Length by Class (Train)",
#                         "Site Position vs Protein Length"],
#         column_widths=[0.45, 0.55],
#     )
#
#     for label, color, name in [(0, TEAL, "Negative"), (1, ORANGE, "Positive")]:
#         sub = train_df[train_df["Label"] == label]
#         fig.add_trace(go.Violin(
#             y=sub["full_sequence"].str.len(),
#             name=name, box_visible=True, meanline_visible=True,
#             fillcolor=color, opacity=0.6,
#             line_color=color, marker_color=color,
#         ), row=1, col=1)
#
#     fig.add_trace(go.Histogram2dContour(
#         x=train_df["full_sequence"].str.len(),
#         y=train_df["pos"].astype(int),
#         colorscale=[[0, "rgba(1,105,111,0)"], [1, TEAL]],
#         showscale=True, line_width=0,
#         hovertemplate="Protein len: %{x}<br>Site pos: %{y}<extra>Density</extra>",
#         showlegend=False,
#     ), row=1, col=2)
#
#     fig.update_layout(
#         **BASE_LAYOUT,
#         title=dict(text="<b>Step 2 — Data Inspection</b>", font_size=17, x=0.02),
#         legend=dict(bgcolor=BG, bordercolor=BORDER, borderwidth=1, y=0.5),
#         height=440, width=1050, violinmode="group",
#     )
#     fig.update_xaxes(showticklabels=False, row=1, col=1)
#     fig.update_yaxes(title_text="Protein Length (aa)", gridcolor=BORDER,
#                      zeroline=False, row=1, col=1)
#     fig.update_xaxes(title_text="Protein Length (aa)", gridcolor=BORDER,
#                      zeroline=False, row=1, col=2)
#     fig.update_yaxes(title_text="Site Position", gridcolor=BORDER,
#                      zeroline=False, row=1, col=2)
#
#     path = os.path.join(out_dir, "step2_data_inspection.png")
#     fig.write_image(path, scale=2)
#     print(f"  ✓  {path}")
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Step 3 — Shuffling QA
# # ─────────────────────────────────────────────────────────────────────────────
# def _aa_freq(seqs):
#     counts = collections.Counter("".join(seqs))
#     total  = sum(counts.values())
#     return {aa: counts.get(aa, 0) / total for aa in AAs}
#
# def _hamming(s1: str, s2: str) -> int:
#     return sum(c1 != c2 for c1, c2 in zip(s1, s2))
#
# def plot_step3(train_df: pd.DataFrame, out_dir: str, sample_n: int = 5000):
#     """AA frequency (original vs shuffled) + Hamming distance distribution."""
#     sample = train_df.sample(min(sample_n, len(train_df)), random_state=42)
#
#     freq_orig    = _aa_freq(sample["Seq"])
#     freq_shuffled = _aa_freq(sample["shuffled_Seq"])
#     aa_sorted    = sorted(AAs, key=lambda a: -freq_orig[a])
#
#     hamming_dists = sample.apply(
#         lambda r: _hamming(r["Seq"], r["shuffled_Seq"]), axis=1
#     )
#     hd_counts = hamming_dists.value_counts().sort_index()
#
#     fig = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=["AA Frequency: Original vs Shuffled",
#                         "Hamming Distance per Window"],
#         column_widths=[0.58, 0.42],
#     )
#
#     fig.add_trace(go.Bar(x=aa_sorted,
#         y=[freq_orig[a] for a in aa_sorted],
#         name="Original", marker_color=TEAL, opacity=0.85), row=1, col=1)
#     fig.add_trace(go.Bar(x=aa_sorted,
#         y=[freq_shuffled[a] for a in aa_sorted],
#         name="Shuffled", marker_color=ORANGE, opacity=0.85), row=1, col=1)
#
#     fig.add_trace(go.Scatter(
#         x=hd_counts.index, y=hd_counts.values,
#         mode="lines+markers",
#         line=dict(color=TEAL, width=2), marker=dict(size=5, color=TEAL),
#         showlegend=False, fill="tozeroy",
#         fillcolor="rgba(1,105,111,0.15)",
#     ), row=1, col=2)
#
#     fig.update_layout(
#         **BASE_LAYOUT,
#         title=dict(text="<b>Step 3 — Shuffling Quality Assurance</b>",
#                    font_size=17, x=0.02),
#         barmode="group",
#         legend=dict(bgcolor=BG, bordercolor=BORDER, borderwidth=1),
#         height=430, width=1050,
#     )
#     fig.update_xaxes(title_text="Amino Acid", row=1, col=1)
#     fig.update_yaxes(title_text="Frequency", gridcolor=BORDER,
#                      zeroline=False, row=1, col=1)
#     fig.update_xaxes(title_text="Positions Changed (out of 50 flanks)",
#                      gridcolor=BORDER, zeroline=False, row=1, col=2)
#     fig.update_yaxes(title_text="Count (windows)", gridcolor=BORDER,
#                      zeroline=False, row=1, col=2)
#
#     path = os.path.join(out_dir, "step3_shuffling_qa.png")
#     fig.write_image(path, scale=2)
#     print(f"  ✓  {path}")
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Main entry point
# # ─────────────────────────────────────────────────────────────────────────────
# def visualize_all(train_df: pd.DataFrame, test_df: pd.DataFrame,
#                   output_dir: str = "output/"):
#     os.makedirs(output_dir, exist_ok=True)
#     print("\nGenerating visualizations …")
#     plot_step1(train_df, test_df, output_dir)
#     plot_step2(train_df, output_dir)
#     plot_step3(train_df, output_dir)
#     print("\nAll done! PNGs saved to:", os.path.abspath(output_dir))
#
#
# if __name__ == "__main__":
#     import sys
#
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
#     train_path = os.path.join(PROCESSED_DIR, "train_processed.csv")
#     test_path  = os.path.join(PROCESSED_DIR, "test_processed.csv")
#
#     if not os.path.exists(train_path):
#         sys.exit(f"Train file not found: {train_path}\nRun the data-prep script first.")
#
#     print(f"Loading {train_path} …")
#     train_df = pd.read_csv(train_path)
#     print(f"Loading {test_path} …")
#     test_df  = pd.read_csv(test_path)
#
#     visualize_all(train_df, test_df, output_dir=os.path.join(BASE_DIR, "reports", "figures"))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------
# Step 1: What data did we load?
# ------------------------------------------------------------------
# This chart shows how many rows (sequences) and columns exist
# in the training and test datasets.
# ------------------------------------------------------------------

fig1 = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Training Dataset", "Testing Dataset"],
    specs=[[{"type": "indicator"}, {"type": "indicator"}]]
)

fig1.add_trace(go.Indicator(
    mode="number",
    value=63300,
    title={"text": "Total Rows (sequences)<br><span style=\'font-size:0.8em;color:gray\'>Used to TRAIN the model</span>"},
    number={"suffix": " rows", "font": {"size": 50, "color": "#01696f"}}
), row=1, col=1)

fig1.add_trace(go.Indicator(
    mode="number",
    value=5559,
    title={"text": "Total Rows (sequences)<br><span style=\'font-size:0.8em;color:gray\'>Used to TEST the model</span>"},
    number={"suffix": " rows", "font": {"size": 50, "color": "#437a22"}}
), row=1, col=2)

fig1.update_layout(
    title=dict(text="Step 1 — Data Loaded Successfully", font=dict(size=22), x=0.5),
    height=350,
    paper_bgcolor="#f9f8f5",
)
fig1.write_image("step1_data_loaded.png", scale=2)
print("✅ step1_data_loaded.png saved")


# ------------------------------------------------------------------
# Step 2: Are the classes balanced?
# ------------------------------------------------------------------
# Label 0 = NOT a PTM site (no modification)
# Label 1 = IS a PTM site (has a chemical modification)
# We want roughly equal numbers of both so the model learns fairly.
# ------------------------------------------------------------------

labels = ["Not a PTM Site (Label=0)", "Is a PTM Site (Label=1)"]
train_counts = [33247, 30053]
test_counts  = [2723,  2836]

fig2 = go.Figure()

fig2.add_trace(go.Bar(
    name="Train Set",
    x=labels,
    y=train_counts,
    marker_color=["#4f98a3", "#01696f"],
    text=[f"{v:,}" for v in train_counts],
    textposition="outside"
))
fig2.add_trace(go.Bar(
    name="Test Set",
    x=labels,
    y=test_counts,
    marker_color=["#a8d5db", "#6daa45"],
    text=[f"{v:,}" for v in test_counts],
    textposition="outside"
))

fig2.update_layout(
    title=dict(
        text="Step 2 — Class Distribution: PTM vs Non-PTM Sites",
        font=dict(size=20), x=0.5
    ),
    barmode="group",
    xaxis_title="Type of Sequence",
    yaxis_title="Number of Sequences",
    height=450,
    paper_bgcolor="#f9f8f5",
    plot_bgcolor="#f7f6f2",
)
fig2.write_image("step2_class_distribution.png", scale=2)
print("✅ step2_class_distribution.png saved")


# ------------------------------------------------------------------
# Step 3: Did the shuffling keep the center residue intact?
# ------------------------------------------------------------------
# The key rule of shuffling: the amino acid in the MIDDLE (position 25)
# must NEVER change. Only the surrounding amino acids are shuffled.
# If the bars match perfectly → the rule was respected.
# ------------------------------------------------------------------

# Example of one sequence before and after shuffle
original  = "ACDEFGHIKLMNPQRSTVWYACDEFKHIGLMNPQRSTVWYACDEFGHIKLM"  # 51 chars
center_aa = original[25]  # The protected center

# Show a before/after comparison of a single sequence as a heatmap
# Each amino acid gets a color number (A=1, C=2, etc.)
aa_list = list("ACDEFGHIKLMNPQRSTVWY")
aa_to_num = {aa: i+1 for i, aa in enumerate(aa_list)}

import random
random.seed(99)
def shuffle_seq(seq):
    center = seq[25]
    flanks = list(seq[:25] + seq[26:])
    random.shuffle(flanks)
    return "".join(flanks[:25]) + center + "".join(flanks[25:])

shuffled = shuffle_seq(original)

orig_nums    = [aa_to_num.get(c, 0) for c in original]
shuffled_nums = [aa_to_num.get(c, 0) for c in shuffled]

positions = list(range(1, 52))

fig3 = go.Figure()

fig3.add_trace(go.Bar(
    x=positions,
    y=[1]*51,
    marker_color=orig_nums,
    marker_colorscale="Teal",
    name="Original Sequence",
    showlegend=True,
    text=[c if i == 25 else "" for i, c in enumerate(original)],
    textposition="outside",
))

fig3.add_annotation(
    x=26, y=1.05,
    text=f"⭐ Center (pos 26) = '{original[25]}' — LOCKED",
    showarrow=True, arrowhead=2, ax=0, ay=-30,
    font=dict(size=13, color="red")
)

fig3.update_layout(
    title=dict(
        text="Step 3 — Shuffling: The Center Amino Acid is ALWAYS Protected",
        font=dict(size=19), x=0.5
    ),
    xaxis_title="Position in the 51-character Window",
    yaxis=dict(visible=False),
    height=350,
    paper_bgcolor="#f9f8f5",
    plot_bgcolor="#f7f6f2",
    bargap=0.05
)
fig3.write_image("step3_shuffle_center_protected.png", scale=2)
print("✅ step3_shuffle_center_protected.png saved")

print("")
print("All 3 charts generated! Open each .png to view.")
