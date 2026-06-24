import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_rrns_distribution(df, output_dir):
    """
    Generates a dual-panel visualization showing the continuous distribution
    of rRNS scores and the final grouped bin counts.
    """
    # Create a 1x2 subplot layout
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Isolate only the candidate negatives (which have calculated rRNS scores)
    neg_df = df[df['Label'] == 0].copy()

    # ---------------------------------------------------------
    # Panel 1: Histogram of rRNS Scores with Threshold Overlays
    # ---------------------------------------------------------
    # Since K=10, scores are in increments of 0.1. We define custom bins to center the bars.
    bins = np.arange(-0.05, 1.15, 0.1)
    sns.histplot(data=neg_df, x='rRNS_Score', bins=bins, color='slategray', edgecolor='black', ax=axes[0])

    # Overlay the Uncertainty Thresholds as colored background spans
    axes[0].axvspan(-0.05, 0.15, color='dodgerblue', alpha=0.15, label='Low (≤ 0.10)')
    axes[0].axvspan(0.15, 0.25, color='gray', alpha=0.1, label='Unassigned')
    axes[0].axvspan(0.25, 0.65, color='mediumseagreen', alpha=0.2, label='Boundary (0.30 - 0.60)')
    axes[0].axvspan(0.65, 1.05, color='crimson', alpha=0.15, label='High (≥ 0.60)')

    axes[0].set_title('Distribution of rRNS Scores (k=10)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('rRNS Score (Proportion of randomized neighbors)')
    axes[0].set_ylabel('Number of Candidate Negatives')
    axes[0].set_xticks(np.arange(0, 1.1, 0.1))
    axes[0].legend(loc='upper right', title='Threshold Regions')

    # ---------------------------------------------------------
    # Panel 2: Final Bin Counts (Including Positives)
    # ---------------------------------------------------------
    # Define a custom color palette for the bins
    palette = {
        'Positive': 'gold',
        'Low': 'dodgerblue',
        'Unassigned': 'lightgray',
        'Boundary': 'mediumseagreen',
        'High': 'crimson'
    }
    order = ['Positive', 'Low', 'Unassigned', 'Boundary', 'High']

    sns.countplot(data=df, x='Uncertainty_Bin', order=order, palette=palette, ax=axes[1], edgecolor='black')

    axes[1].set_title('Final Dataset Composition by Uncertainty Bin', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Assigned Category')
    axes[1].set_ylabel('Total Count')

    # Add count labels on top of the bars
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%d', label_type='edge', padding=3)

    plt.suptitle('Steps 6 & 7: Latent Space Uncertainty Scoring and Negative Sampling', fontsize=16, y=1.02)
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, 'step6_7_rrns_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved rRNS visualization to: {output_path}")


if __name__ == "__main__":
    print("--- Creating rRNS & Negative Sampling Visualizations ---")

    # 1. Setup Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SCORED_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train_scored_with_bins.csv")
    VIS_DIR = os.path.join(BASE_DIR, "visualizations")

    os.makedirs(VIS_DIR, exist_ok=True)

    # 2. Load Data
    if not os.path.exists(SCORED_TRAIN_PATH):
        raise FileNotFoundError(f"Cannot find {SCORED_TRAIN_PATH}. Run 03_rRNS_scoring.py first.")

    print("Loading scored dataset...")
    df = pd.read_csv(SCORED_TRAIN_PATH)

    # 3. Generate Plot
    plot_rrns_distribution(df, VIS_DIR)

    print("\nExecution complete.")