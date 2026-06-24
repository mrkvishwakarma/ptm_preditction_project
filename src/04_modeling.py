"""Code for Steps 8 and 9 (Classifier training and evaluation)."""
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import joblib


def extract_training_set(df, embeddings, bin_name=None, is_baseline=False):
    """
    Extracts the Positives and the specified subset of Negatives.
    For the baseline, it takes a random sample of negatives equal to the bin size
    so that class imbalances remain comparable across models.
    """
    pos_mask = df['Label'] == 1

    if is_baseline:
        # Get all negatives, then randomly sample them
        all_neg_mask = df['Label'] == 0
        neg_indices = df[all_neg_mask].index.tolist()

        # Arbitrarily sampling 5000 for the baseline, adjust based on your actual bin sizes!
        sampled_neg_indices = resample(neg_indices, n_samples=5000, random_state=42)
        neg_mask = df.index.isin(sampled_neg_indices)
    else:
        # Filter by the specific uncertainty bin
        neg_mask = df['Uncertainty_Bin'] == bin_name

    # Combine masks
    combined_mask = pos_mask | neg_mask

    # Extract features (X) and labels (y)
    X = embeddings[combined_mask]
    y = df.loc[combined_mask, 'Label'].values

    print(
        f"Dataset '{bin_name if not is_baseline else 'Baseline'}': {sum(pos_mask)} Positives, {sum(neg_mask)} Negatives.")
    return X, y


if __name__ == "__main__":
    print("--- Step 8: Train PTM Predictors ---")

    # 1. Setup Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SCORED_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train_scored_with_bins.csv")
    EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 2. Load Data
    print("Loading data and real embeddings...")
    df = pd.read_csv(SCORED_TRAIN_PATH)
    X_all = np.load(os.path.join(EMBEDDINGS_DIR, "train_real_embeddings.npy"))

    # 3. Define the modeling configurations
    configs = {
        'Baseline': {'is_baseline': True, 'bin_name': None},
        'Low_Uncertainty': {'is_baseline': False, 'bin_name': 'Low'},
        'Boundary_Uncertainty': {'is_baseline': False, 'bin_name': 'Boundary'},
        'High_Uncertainty': {'is_baseline': False, 'bin_name': 'High'}
    }

    trained_models = {}

    # 4. Train identical models on different subsets
    for model_name, params in configs.items():
        print(f"\n--- Training {model_name} Model ---")

        # Ensure the bin exists in the dataframe before proceeding
        if not params['is_baseline'] and params['bin_name'] not in df['Uncertainty_Bin'].values:
            print(f"Skipping {model_name}: No data found for bin '{params['bin_name']}'.")
            continue

        X_train, y_train = extract_training_set(df, X_all, params['bin_name'], params['is_baseline'])

        # Initialize an identical Random Forest for each configuration
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        # Save the model
        model_path = os.path.join(MODELS_DIR, f"{model_name}_rf_model.pkl")
        joblib.dump(clf, model_path)
        print(f"Model saved to {model_path}")

        trained_models[model_name] = clf

    print("\nAll models trained and saved successfully.")