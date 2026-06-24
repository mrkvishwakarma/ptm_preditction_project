import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA


def plot_sequence_lengths(raw_train_df, max_length, output_dir):
    """
    Visualizes the distribution of protein sequence lengths and the impact of the cutoff.
    """
    plt.figure(figsize=(10, 6))

    # Calculate lengths of the full sequences
    lengths = raw_train_df['full_sequence'].dropna().apply(len)

    # Plot histogram/KDE
    sns.histplot(lengths, bins=50, color='teal', kde=True, alpha=0.6)

    # Draw the truncation line
    plt.axvline(x=max_length, color='red', linestyle='--', linewidth=2,
                label=f'Model MAX_LENGTH Cutoff ({max_length})')

    # Add annotations
    dropped = (lengths >= max_length).sum()
    kept = (lengths < max_length).sum()
    plt.text(max_length + 50, plt.ylim()[1] * 0.8,
             f"Dropped: {dropped}\nKept: {kept}",
             color='darkred', bbox=dict(facecolor='white', alpha=0.8))

    plt.title('Distribution of Protein Sequence Lengths (Training Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Amino Acids in Protein')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    output_path = os.path.join(output_dir, 'step4_sequence_length_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved length distribution to: {output_path}")


def plot_embedding_pca(real_embeds, shuffled_embeds, output_dir, sample_size=2000):
    """
    Uses PCA to reduce the 320-dimensional embeddings to 2D for visual comparison
    between Real and Shuffled contexts.
    """
    print(f"Running PCA on a sample of {sample_size} embeddings...")

    # Sample the embeddings to keep the scatter plot readable and computation fast
    idx = np.random.choice(real_embeds.shape[0], size=min(sample_size, real_embeds.shape[0]), replace=False)
    sampled_real = real_embeds[idx]
    sampled_shuff = shuffled_embeds[idx]

    # Combine them for PCA fitting
    combined = np.vstack((sampled_real, sampled_shuff))
    labels = ['Real Context'] * len(sampled_real) + ['Shuffled Context'] * len(sampled_shuff)

    # Reduce to 2 dimensions
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(combined)

    # Create DataFrame for plotting
    pca_df = pd.DataFrame({
        'PC1': components[:, 0],
        'PC2': components[:, 1],
        'Source': labels
    })

    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Source',
                    palette={'Real Context': 'royalblue', 'Shuffled Context': 'darkorange'},
                    alpha=0.6, edgecolor=None, s=30)

    variance_ratio = pca.explained_variance_ratio_ * 100
    plt.xlabel(f'Principal Component 1 ({variance_ratio[0]:.1f}% Variance)')
    plt.ylabel(f'Principal Component 2 ({variance_ratio[1]:.1f}% Variance)')
    plt.title('PCA Projection of Embeddings: Real vs. Shuffled Windows', fontsize=14, fontweight='bold')

    # Add an explanatory box
    plt.text(pca_df['PC1'].min(), pca_df['PC2'].max(),
             "Separation between colors visually confirms that shuffling the local\nwindow changes the ESM-2 embedding of the center residue.",
             fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.9, pad=5))

    output_path = os.path.join(output_dir, 'step5_embedding_pca_projection.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA projection to: {output_path}")


if __name__ == "__main__":
    print("--- Creating Embedding Visualizations ---")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train_processed.csv")
    EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
    VIS_DIR = os.path.join(BASE_DIR, "visualizations")
    os.makedirs(VIS_DIR, exist_ok=True)

    # 1. Load Data for Length Plot (Needs the raw un-filtered data)
    print("Loading data...")
    raw_train_df = pd.read_csv(PROCESSED_TRAIN_PATH)

    # 2. Load Embeddings for PCA
    real_embeddings = np.load(os.path.join(EMBEDDINGS_DIR, "train_real_embeddings.npy"))
    shuffled_embeddings = np.load(os.path.join(EMBEDDINGS_DIR, "train_shuffled_embeddings.npy"))

    # 3. Generate Plots
    plot_sequence_lengths(raw_train_df, max_length=1024, output_dir=VIS_DIR)
    plot_embedding_pca(real_embeddings, shuffled_embeddings, VIS_DIR)

    print("\nExecution complete.")