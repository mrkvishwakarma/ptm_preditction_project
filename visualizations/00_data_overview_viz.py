import plotly.graph_objects as go


# ------------------------------------------------------------------
# Chart 1: Column overview — table view
# ------------------------------------------------------------------
# Shows each of the 5 columns in the CSV, what type of data it is,
# what it means, and an example value from the actual dataset.
# ------------------------------------------------------------------

columns  = ["Seq", "Label", "UniProtID", "pos", "full_sequence"]
dtypes   = ["text (string)", "number (0 or 1)", "text (string)", "number (float)", "text (string)"]
meanings = [
    "51-letter protein window centered on the site of interest",
    "0 = NOT a PTM site  |  1 = IS a PTM site  (what we want to predict)",
    "Unique ID of the protein (e.g. O75643) — like a barcode",
    "Position of the site inside the full protein (e.g. 46 = 46th amino acid)",
    "The entire protein sequence from start to end"
]
examples = [
    "LQADRSLIDRTRRDEPTGEVL...",
    "1",
    "O75643",
    "46.0",
    "MADVTARSLQYEYKANSNLVLQADRS..."
]
col_colors = ["#01696f", "#da7101", "#006494", "#437a22", "#a12c7b"]

fig1 = go.Figure()
fig1.add_trace(go.Table(
    columnwidth=[120, 140, 400, 250],
    header=dict(
        values=["<b>Column Name</b>", "<b>Data Type</b>",
                "<b>What It Means</b>", "<b>Example Value</b>"],
        fill_color="#28251d",
        font=dict(color="white", size=13, family="Arial"),
        align="left", height=36
    ),
    cells=dict(
        values=[columns, dtypes, meanings, examples],
        fill_color=[
            [f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.12)"
             for c in col_colors],
            ["#f7f6f2"]*5, ["#f9f8f5"]*5, ["#f7f6f2"]*5,
        ],
        font=dict(color="#28251d", size=12, family="Arial"),
        align="left", height=44, line_color="#dcd9d5"
    )
))
fig1.update_layout(
    title=dict(
        text="Before Step 1 — What Columns Are in the Dataset?<br>"
             "<sup>Each row is one candidate PTM site. "
             "These 5 columns describe everything we know about it.</sup>",
        font=dict(size=18), x=0.5),
    paper_bgcolor="#f9f8f5",
    height=380, margin=dict(t=100, b=20, l=20, r=20),
    font=dict(family="Arial", color="#28251d")
)
fig1.write_image("step0_column_overview.png", scale=2)
print("✅ step0_column_overview.png saved")


# ------------------------------------------------------------------
# Chart 2: One row explained — visual card layout
# ------------------------------------------------------------------
# Shows a single data row as color-coded cards, with the column name,
# data type, example value, and plain-English meaning for each field.
# ------------------------------------------------------------------
import plotly.graph_objects as go

fields = [
    ("Seq", "51-letter window", "LQADRSLIDRTRRDEPTGE...", "#01696f",
     "The 51 amino acids around the site.<br>This is the main input."),
    ("Label", "0 or 1", "1", "#da7101",
     "The answer we want to predict.<br>1 = PTM site, 0 = not PTM."),
    ("UniProtID", "protein ID", "O75643", "#006494",
     "The unique ID of the protein.<br>Like an ISBN for a book."),
    ("pos", "position number", "46", "#437a22",
     "Which amino acid position this is<br>in the full protein."),
    ("full_sequence", "whole protein", "MADVTARSLQYEYKANSNLV...", "#a12c7b",
     "The full protein sequence.<br>Used to make the 51-letter window.")
]

fig2 = go.Figure()
row_h = 0.9
gap = 0.12
left_w = 1.5

# The x-coordinate where the text in the right box should start
text_start_x = left_w + 0.1

for i, (name, typ, ex, color, desc) in enumerate(fields):
    y_top = (len(fields) - i) * (row_h + gap)
    y_bot = y_top - row_h

    # Left colored box
    fig2.add_shape(
        type="rect",
        x0=0, x1=left_w, y0=y_bot, y1=y_top,
        fillcolor=color,
        line_width=0,
        layer="below"
    )

    # Right transparent box
    fig2.add_shape(
        type="rect",
        x0=left_w, x1=10, y0=y_bot, y1=y_top,
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
        line=dict(color=color, width=1),
        layer="below"
    )

    # Name label (Seq, Label, etc.)
    fig2.add_annotation(
        x=left_w/2, y=(y_top+y_bot)/2,
        text=f"<b>{name}</b>",
        showarrow=False,
        font=dict(size=14, color="white"),
        xanchor="center",
        yanchor="middle"
    )

    # Type text (e.g. "51-letter window")
    fig2.add_annotation(
        x=text_start_x, y=y_top-0.2,
        text=f"<i>{typ}</i>",
        showarrow=False,
        font=dict(size=12, color=color),
        xanchor="left"
    )

    # Example text
    fig2.add_annotation(
        x=text_start_x + 1.4, y=y_top-0.2,
        text=f"<b>e.g.</b> {ex}",
        showarrow=False,
        font=dict(size=12, color="#28251d"),
        xanchor="left"
    )

    # Description text
    fig2.add_annotation(
        x=text_start_x, y=(y_top+y_bot)/2 - 0.1,
        text=desc,
        showarrow=False,
        font=dict(size=12, color="#28251d"),
        xanchor="left",
        align="left"
    )

fig2.update_layout(
    title=dict(
        text="What does one row mean?<br><sup>Each row is one site from one protein.</sup>",
        font=dict(size=18),
        x=0.5
    ),
    xaxis=dict(visible=False, range=[-0.2, 10.2]),
    yaxis=dict(visible=False, range=[-0.2, 6.5]),
    plot_bgcolor="#f9f8f5",
    paper_bgcolor="#f9f8f5",
    height=500,
    margin=dict(t=90, b=20, l=10, r=10),
    font=dict(family="Arial", color="#28251d")
)

fig2.write_image("step0_one_row_explained.png", scale=2)
print("✅ step0_one_row_explained.png saved")



# ------------------------------------------------------------------
# Chart 3: What is a PTM site? — protein diagram
# ------------------------------------------------------------------
# Shows a protein as a long bar with amino acids as dots.
# PTM sites (orange dots) are marked with a zoomed window showing
# where the Seq column comes from.
# ------------------------------------------------------------------

fig3 = go.Figure()
fig3.add_shape(type="rect", x0=0, x1=10, y0=0.38, y1=0.62,
    fillcolor="#cedcd8", line=dict(color="#01696f", width=2), layer="below")
fig3.add_annotation(x=5, y=0.5,
    text="<b>Full Protein Sequence  (can be hundreds to thousands of amino acids long)</b>",
    showarrow=False, font=dict(size=12, color="#28251d"), xanchor="center")

non_ptm = [1.0, 1.6, 3.0, 3.8, 4.5, 6.6, 7.2, 7.8, 9.0, 9.6]
ptm_pos = [5.8]     # single highlighted example

for xp in non_ptm:
    fig3.add_shape(type="circle", x0=xp-0.18, x1=xp+0.18, y0=0.34, y1=0.66,
        fillcolor="#006494", line=dict(color="white", width=1), layer="above")

for xp in ptm_pos:
    fig3.add_shape(type="circle", x0=xp-0.22, x1=xp+0.22, y0=0.30, y1=0.70,
        fillcolor="#da7101", line=dict(color="white", width=2), layer="above")
    fig3.add_annotation(x=xp, y=0.88, text="⬇️ PTM Site  (Label = 1)",
        showarrow=True, ax=0, ay=-20,
        font=dict(size=11, color="#da7101"), xanchor="center")

# 51-window dashed box
zoom_x = 5.8
fig3.add_shape(type="rect",
    x0=zoom_x-1.5, x1=zoom_x+1.5, y0=0.22, y1=0.78,
    line=dict(color="#da7101", width=1.5, dash="dash"),
    fillcolor="rgba(218,113,1,0.05)")
fig3.add_annotation(x=zoom_x, y=0.10,
    text="⬆️ 51-letter window around this site → stored as the 'Seq' column",
    showarrow=False, font=dict(size=11, color="#da7101"), xanchor="center")

fig3.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
    marker=dict(color="#006494", size=12, symbol="circle"),
    name="Normal amino acid  (Label = 0)"))
fig3.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
    marker=dict(color="#da7101", size=14, symbol="circle"),
    name="PTM site  (Label = 1) — gets chemically modified"))

fig3.update_layout(
    title=dict(
        text="Before Step 1 — What is a PTM Site?<br>"
             "<sup>A Post-Translational Modification (PTM) is a chemical change at a specific position on a protein. "
             "</sup>",
        font=dict(size=17), x=0.5),
    xaxis=dict(visible=False, range=[-0.3, 10.3]),
    yaxis=dict(visible=False, range=[-0.05, 1.15]),
    plot_bgcolor="#f9f8f5", paper_bgcolor="#f9f8f5",
    legend=dict(font=dict(size=12), x=0.5, xanchor="center", y=-0.08, orientation="h"),
    height=420, margin=dict(t=120, b=70, l=20, r=20),
    font=dict(family="Arial", color="#28251d")
)
fig3.write_image("step0_what_is_ptm.png", scale=2)
print("✅ step0_what_is_ptm.png saved")

print("\nAll 3 Step-0 charts generated successfully!")
