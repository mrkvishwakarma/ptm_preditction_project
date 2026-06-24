"""Code for Steps 6 and 7 (Latent space, rRNS calculation, binning)."""
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def calculate_rrns_and_bins(df, real_embeddings, shuffled_embeddings, k=10):
    """
    Calculates the rRNS score for candidate negative sites and assigns uncertainty bins.
    """
    print("Filtering for candidate negative sites (Label == 0)...")
    # 1. Create a boolean mask for negative sites
    neg_mask = (df['Label'] == 0).values

    # 2. Extract dataframe and embeddings for ONLY the negative sites
    neg_df = df[neg_mask].copy()
    real_neg_embeds = real_embeddings[neg_mask]
    shuff_neg_embeds = shuffled_embeddings[neg_mask]

    print(f"Found {len(neg_df)} negative sites. Building latent space...")

    # 3. Combine to build the latent space pool
    # Stack the real and shuffled arrays vertically
    latent_space = np.vstack((real_neg_embeds, shuff_neg_embeds))

    # Create an array tracking the source: 0 for Real, 1 for Randomized
    source_labels = np.array([0] * len(real_neg_embeds) + [1] * len(shuff_neg_embeds))

    # 4. Fit the K-Nearest Neighbors model
    print("Fitting K-Nearest Neighbors model...")
    # We use n_neighbors = k + 1 because the query point will always find itself as the 1st neighbor
    knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1)
    knn.fit(latent_space)

    # 5. Query the model and calculate rRNS
    print(f"Querying {k} nearest neighbors to calculate rRNS...")
    distances, indices = knn.kneighbors(real_neg_embeds)

    rrns_scores = []
    # Iterate through the neighbor indices for each real negative site
    for neighbor_idx in tqdm(indices):
        # Drop the first neighbor (which is the point itself)
        actual_neighbors = neighbor_idx[1:]

        # Count how many of these actual neighbors are from the randomized set (label == 1)
        randomized_count = np.sum(source_labels[actual_neighbors] == 1)

        # Calculate score: (randomized neighbors) / k
        score = randomized_count / k
        rrns_scores.append(score)

    neg_df['rRNS_Score'] = rrns_scores

    # --- Step 7: Assign Uncertainty Bins ---
    print("Assigning Uncertainty Bins based on diagram thresholds...")

    def assign_bin(score):
        if score <= 0.10:
            return 'Low'
        elif 0.30 <= score <= 0.60:
            return 'Boundary'
        elif score >= 0.60:
            return 'High'
        else:
            return 'Unassigned'  # For scores strictly between 0.10-0.30, based on diagram

    neg_df['Uncertainty_Bin'] = neg_df['rRNS_Score'].apply(assign_bin)

    # Merge the scores and bins back into the FULL training dataframe
    # Default all rows to NaN score and 'Positive' bin
    df['rRNS_Score'] = np.nan
    df['Uncertainty_Bin'] = 'Positive'

    # Map the calculated values directly back using the original index
    df.loc[neg_mask, 'rRNS_Score'] = neg_df['rRNS_Score']
    df.loc[neg_mask, 'Uncertainty_Bin'] = neg_df['Uncertainty_Bin']

    return df


if __name__ == "__main__":
    print("--- Steps 6 & 7: Build Latent Space & Negative Sampling ---")

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train_processed.csv")
    EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")

    # Load Data (Make sure you drop NaNs and filter length exactly like in 02_embeddings.py!)
    print("Loading training data and embeddings...")
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    train_df = train_df.dropna(subset=['full_sequence', 'shuffled_full_sequence'])
    train_df = train_df[train_df['pos'] < 1024].reset_index(drop=True)

    real_embeddings = np.load(os.path.join(EMBEDDINGS_DIR, "train_real_embeddings.npy"))
    shuffled_embeddings = np.load(os.path.join(EMBEDDINGS_DIR, "train_shuffled_embeddings.npy"))

    # Calculate!
    # Note: K=10 is standard, but you can adjust it if your literature suggests otherwise.
    scored_train_df = calculate_rrns_and_bins(train_df, real_embeddings, shuffled_embeddings, k=10)

    # Save the highly valuable scored dataset
    output_path = os.path.join(BASE_DIR, "data", "processed", "train_scored_with_bins.csv")
    scored_train_df.to_csv(output_path, index=False)

    print(f"\nSuccessfully saved scored dataset to: {output_path}")

    # Quick sanity check printout
    print("\n--- Distribution of Uncertainty Bins ---")
    print(scored_train_df['Uncertainty_Bin'].value_counts())