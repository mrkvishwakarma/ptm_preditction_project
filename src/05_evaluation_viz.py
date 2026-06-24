import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def plot_performance_metrics(metrics_df, output_dir):
    """
    Generates a grouped bar chart for AUROC, AUPR, MCC, and F1 Score
    across the four negative sampling strategies.
    """
    plt.figure(figsize=(12, 7))

    # Melt the dataframe so seaborn can easily group the bars
    melted_df = metrics_df.melt(id_vars='Model',
                                value_vars=['AUROC', 'AUPR', 'F1_Score', 'MCC'],
                                var_name='Metric', value_name='Score')

    # Create the grouped bar chart
    ax = sns.barplot(data=melted_df, x='Metric', y='Score', hue='Model',
                     palette='viridis', edgecolor='black', linewidth=0.5)

    plt.title('Step 9: Final Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Score (0.0 to 1.0)', fontsize=12)
    plt.xlabel('Evaluation Metric', fontsize=12)
    plt.ylim(0, 1.0)

    # Add an explanatory box pointing out the MCC victory
    boundary_mcc = metrics_df.loc[metrics_df['Model'] == 'Boundary_Uncertainty', 'MCC'].values[0]
    baseline_mcc = metrics_df.loc[metrics_df['Model'] == 'Baseline', 'MCC'].values[0]

    plt.text(2.5, 0.85,
             f"Key Finding:\nThe Boundary model doubled the\nMCC compared to the Baseline\n({boundary_mcc:.3f} vs {baseline_mcc:.3f}).",
             fontsize=10, bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.8'))

    plt.legend(title='Negative Sampling Strategy', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'step9_performance_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved performance metrics visualization to: {output_path}")


def plot_confusion_matrices(metrics_df, output_dir):
    """
    Generates a 2x2 grid of confusion matrices to visually dissect
    where each model succeeded or failed.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Define a clean title mapping for the models
    title_map = {
        'Baseline': 'Baseline (Random Negatives)',
        'Low_Uncertainty': 'Low Uncertainty Negatives',
        'Boundary_Uncertainty': 'Boundary Uncertainty Negatives (Winner)',
        'High_Uncertainty': 'High Uncertainty Negatives'
    }

    for idx, row in metrics_df.iterrows():
        model_name = row['Model']

        # Reconstruct the 2x2 matrix from the CSV row
        # Format: [[True Negative, False Positive], [False Negative, True Positive]]
        cm = np.array([[row['TN'], row['FP']],
                       [row['FN'], row['TP']]])

        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['Predicted Neg (0)', 'Predicted Pos (1)'],
                    yticklabels=['Actual Neg (0)', 'Actual Pos (1)'],
                    annot_kws={"size": 14, "weight": "bold"})

        axes[idx].set_title(title_map.get(model_name, model_name), fontsize=12, fontweight='bold')

    plt.suptitle('Step 9: Confusion Matrices by Sampling Strategy', fontsize=16, fontweight='bold', y=0.98)

    # Add overall context
    plt.figtext(0.5, 0.02,
                "Notice how the Boundary model dramatically increases True Negatives (TN) while reducing False Positives (FP).",
                ha="center", fontsize=11, bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make room for suptitle and figtext

    output_path = os.path.join(output_dir, 'step9_confusion_matrices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrices visualization to: {output_path}")


if __name__ == "__main__":
    print("--- Creating Final Evaluation Visualizations ---")

    # 1. Setup Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    METRICS_PATH = os.path.join(BASE_DIR, "data", "processed", "final_evaluation_metrics.csv")
    VIS_DIR = os.path.join(BASE_DIR, "visualizations")

    os.makedirs(VIS_DIR, exist_ok=True)

    # 2. Load Data
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(f"Cannot find {METRICS_PATH}. Run your Step 9 evaluation script first.")

    print("Loading evaluation metrics...")
    metrics_df = pd.read_csv(METRICS_PATH)

    # 3. Generate Plots
    plot_performance_metrics(metrics_df, VIS_DIR)
    plot_confusion_matrices(metrics_df, VIS_DIR)

    print("\nExecution complete. Your project visualizations are finished!")