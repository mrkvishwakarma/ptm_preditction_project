import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_training_compositions(df, output_dir, baseline_sample_size=4414):
    """
    Visualizes the exact number of Positive and Negative samples used to
    train each of the four Random Forest models.
    """
    # 1. Calculate the counts based on the exact logic from your Step 8 script
    pos_count = (df['Label'] == 1).sum()

    # Extract negative counts for each bin
    low_neg = (df['Uncertainty_Bin'] == 'Low').sum()
    boundary_neg = (df['Uncertainty_Bin'] == 'Boundary').sum()
    high_neg = (df['Uncertainty_Bin'] == 'High').sum()

    # 2. Structure the data for Seaborn
    data = [
        {'Model': 'Baseline', 'Class': 'Positives (Fixed)', 'Count': pos_count},
        {'Model': 'Baseline', 'Class': 'Negatives (Sampled)', 'Count': baseline_sample_size},

        {'Model': 'Low Uncertainty', 'Class': 'Positives (Fixed)', 'Count': pos_count},
        {'Model': 'Low Uncertainty', 'Class': 'Negatives (Sampled)', 'Count': low_neg},

        {'Model': 'Boundary Uncertainty', 'Class': 'Positives (Fixed)', 'Count': pos_count},
        {'Model': 'Boundary Uncertainty', 'Class': 'Negatives (Sampled)', 'Count': boundary_neg},

        {'Model': 'High Uncertainty', 'Class': 'Positives (Fixed)', 'Count': pos_count},
        {'Model': 'High Uncertainty', 'Class': 'Negatives (Sampled)', 'Count': high_neg},
    ]

    plot_df = pd.DataFrame(data)

    # 3. Create the Visualization
    plt.figure(figsize=(12, 6))

    # Grouped bar chart
    ax = sns.barplot(data=plot_df, x='Model', y='Count', hue='Class',
                     palette={'Positives (Fixed)': 'mediumpurple', 'Negatives (Sampled)': 'lightseagreen'},
                     edgecolor='black', linewidth=1)

    # Add count labels on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3, fontsize=10)

    plt.title('Step 8: Training Set Composition and Class Imbalance per Model', fontsize=15, fontweight='bold')
    plt.xlabel('Classifier Configuration', fontsize=12)
    plt.ylabel('Number of Samples in Training Set', fontsize=12)

    # Format the y-axis with commas for readability
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    # Add an explanatory text box
    plt.text(0.5, pos_count * 0.85,
             "Note: The positive class remains identical across all models.\nOnly the negative sampling strategy changes.",
             fontsize=10, bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))

    plt.legend(title='Sample Type', loc='upper left')
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, 'step8_training_compositions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved training composition visualization to: {output_path}")


if __name__ == "__main__":
    print("--- Creating Model Training Visualizations ---")

    # 1. Setup Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SCORED_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train_scored_with_bins.csv")
    VIS_DIR = os.path.join(BASE_DIR, "visualizations")

    os.makedirs(VIS_DIR, exist_ok=True)

    # 2. Load Data
    if not os.path.exists(SCORED_TRAIN_PATH):
        raise FileNotFoundError(f"Cannot find {SCORED_TRAIN_PATH}.")

    print("Loading scored dataset...")
    df = pd.read_csv(SCORED_TRAIN_PATH)

    # 3. Generate Plot
    # (Passing 4414 based on your terminal output for the Baseline model)
    plot_training_compositions(df, VIS_DIR, baseline_sample_size=4414)

    print("\nExecution complete.")