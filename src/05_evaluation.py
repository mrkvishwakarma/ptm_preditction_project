import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import (confusion_matrix, matthews_corrcoef, f1_score,
                             roc_auc_score, average_precision_score)


def evaluate_model(model, X_test, y_test, model_name):
    """
    Calculates all metrics specified in Step 9 of the flow diagram.
    """
    # Get standard predictions (0 or 1) and probability scores for AUROC/AUPR
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the Positive class (1)

    # 1. Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # 2. Advanced Metrics
    mcc = matthews_corrcoef(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)
    aupr = average_precision_score(y_test, y_prob)

    return {
        'Model': model_name,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'MCC': mcc, 'F1_Score': f1, 'AUROC': auroc, 'AUPR': aupr
    }


if __name__ == "__main__":
    print("--- Step 9: Model Evaluation ---")

    # 1. Setup Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test_processed.csv")
    EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    # 2. Load and Prepare Test Data
    print("Loading test data and embeddings...")
    test_df = pd.read_csv(PROCESSED_TEST_PATH)

    # CRITICAL: We must apply the exact same filters we used in 02_embeddings.py
    # so that the DataFrame rows align perfectly with the .npy arrays.
    test_df = test_df.dropna(subset=['full_sequence', 'shuffled_full_sequence'])
    test_df = test_df[test_df['pos'] < 1024].reset_index(drop=True)

    X_test = np.load(os.path.join(EMBEDDINGS_DIR, "test_real_embeddings.npy"))
    y_test = test_df['Label'].values

    print(f"Test Set Ready: {len(y_test)} samples.")

    # 3. Evaluate each model
    results = []
    model_names = ['Baseline', 'Low_Uncertainty', 'Boundary_Uncertainty', 'High_Uncertainty']

    for name in model_names:
        model_path = os.path.join(MODELS_DIR, f"{name}_rf_model.pkl")

        if not os.path.exists(model_path):
            print(f"Warning: Model {name} not found. Skipping...")
            continue

        print(f"Evaluating {name} Model...")
        model = joblib.load(model_path)
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)

    # 4. Display Results
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)

    results_df = pd.DataFrame(results)

    # Reorder columns slightly for better readability
    cols = ['Model', 'AUROC', 'AUPR', 'MCC', 'F1_Score', 'TP', 'FP', 'TN', 'FN']
    results_df = results_df[cols]

    print(results_df.to_string(index=False))

    # Save results to CSV
    output_path = os.path.join(BASE_DIR, "data", "processed", "final_evaluation_metrics.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved evaluation metrics to: {output_path}")