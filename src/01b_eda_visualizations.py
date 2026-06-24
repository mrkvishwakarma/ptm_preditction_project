import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter


def plot_amino_acid_composition(df, output_dir):
    """
    Compares the relative frequency of amino acids in the local window (Seq)
    between Positive and Negative sites.
    """
    print("Calculating amino acid frequencies...")

    # Isolate sequences by class and drop NaNs
    pos_seqs = "".join(df[df['Label'] == 1]['Seq'].dropna().tolist())
    neg_seqs = "".join(df[df['Label'] == 0]['Seq'].dropna().tolist())

    # Count occurrences
    pos_counts = Counter(pos_seqs)
    neg_counts = Counter(neg_seqs)

    # Standard 20 amino acids (ignoring unknown 'X' or gaps '-')
    standard_aas = list("ACDEFGHIKLMNPQRSTVWY")

    # Calculate relative frequencies (percentages)
    pos_total = sum(pos_counts[aa] for aa in standard_aas)
    neg_total = sum(neg_counts[aa] for aa in standard_aas)

    data = []
    for aa in standard_aas:
        data.append(
            {'Amino Acid': aa, 'Frequency (%)': (pos_counts[aa] / pos_total) * 100, 'Class': 'Positive (Verified PTM)'})
        data.append(
            {'Amino Acid': aa, 'Frequency (%)': (neg_counts[aa] / neg_total) * 100, 'Class': 'Negative (Candidate)'})

    freq_df = pd.DataFrame(data)

    # Plotting
    plt.figure(figsize=(14, 6))
    sns.barplot(data=freq_df, x='Amino Acid', y='Frequency (%)', hue='Class',
                palette={'Positive (Verified PTM)': 'mediumorchid', 'Negative (Candidate)': 'silver'},
                edgecolor='black', linewidth=0.5)

    plt.title('EDA: Amino Acid Composition in Local Sequence Windows', fontsize=14, fontweight='bold')
    plt.ylabel('Relative Frequency (%)')
    plt.xlabel('Amino Acid')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.legend(title='Site Type')

    output_path = os.path.join(output_dir, 'eda_aa_composition.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved AA composition plot to: {output_path}")


def plot_sites_per_protein(df, output_dir):
    """
    Visualizes how many target residues (rows) belong to each unique protein.
    """
    print("Calculating sites per unique protein...")

    # Group by UniProtID and count rows
    sites_per_protein = df.groupby('UniProtID').size()

    plt.figure(figsize=(10, 6))
    sns.histplot(sites_per_protein, bins=50, color='coral', kde=True, edgecolor='black')

    # Add statistics to the plot
    mean_sites = sites_per_protein.mean()
    median_sites = sites_per_protein.median()
    max_sites = sites_per_protein.max()

    plt.axvline(x=mean_sites, color='red', linestyle='--', label=f'Mean: {mean_sites:.1f}')
    plt.axvline(x=median_sites, color='blue', linestyle=':', label=f'Median: {median_sites:.1f}')

    plt.text(0.70, 0.85, f"Total Unique Proteins: {len(sites_per_protein)}\nMax sites on one protein: {max_sites}",
             transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, pad=5))

    plt.title('EDA: Distribution of Target Sites per Protein', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Sites on a Single Protein')
    plt.ylabel('Number of Proteins')
    plt.legend()

    output_path = os.path.join(output_dir, 'eda_sites_per_protein.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sites per protein plot to: {output_path}")


def plot_normalized_positions(df, output_dir):
    """
    Plots the normalized position (0.0 to 1.0) of the PTM sites along the protein chain.
    """
    print("Calculating normalized protein positions...")

    # Drop rows missing full sequences to avoid division by zero errors
    clean_df = df.dropna(subset=['full_sequence', 'pos']).copy()

    # Calculate relative position (Position / Total Length)
    clean_df['Length'] = clean_df['full_sequence'].apply(len)
    clean_df['Normalized_Pos'] = clean_df['pos'] / clean_df['Length']

    # Map labels for the legend
    clean_df['Class'] = clean_df['Label'].map({1: 'Positive', 0: 'Negative'})

    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=clean_df, x='Normalized_Pos', hue='Class', fill=True,
                palette={'Positive': 'mediumorchid', 'Negative': 'silver'}, alpha=0.5, common_norm=False)

    plt.title('EDA: Relative Spatial Distribution of Sites Along Protein Chains', fontsize=14, fontweight='bold')
    plt.xlabel('Normalized Position (0.0 = N-terminus, 1.0 = C-terminus)')
    plt.ylabel('Density')
    plt.xlim(0, 1)

    # Add context text
    plt.text(0.5, -0.15,
             "A flat curve indicates sites are distributed evenly across the protein.\nPeaks indicate a structural preference (e.g., near the terminuses).",
             ha='center', transform=plt.gca().transAxes, fontsize=10, fontstyle='italic')

    output_path = os.path.join(output_dir, 'eda_normalized_positions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved normalized position plot to: {output_path}")


if __name__ == "__main__":
    print("--- Running Exploratory Data Analysis (EDA) ---")

    # 1. Setup Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # We use the raw train.csv because we want to see the biological reality before any pipeline filtering
    RAW_TRAIN_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")
    VIS_DIR = os.path.join(BASE_DIR, "visualizations")

    os.makedirs(VIS_DIR, exist_ok=True)

    # 2. Load Data
    if not os.path.exists(RAW_TRAIN_PATH):
        raise FileNotFoundError(f"Cannot find {RAW_TRAIN_PATH}.")

    print(f"Loading raw dataset from {RAW_TRAIN_PATH}...")
    df = pd.read_csv(RAW_TRAIN_PATH)

    # 3. Generate Exploratory Plots
    plot_amino_acid_composition(df, VIS_DIR)
    plot_sites_per_protein(df, VIS_DIR)
    plot_normalized_positions(df, VIS_DIR)

    print("\nEDA complete. Review the generated visualizations to understand your data better!")