import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
MAX_LENGTH = 1024 # The maximum sequence length for this model
BATCH_SIZE = 16 # Adjust based on your GPU memory

def generate_embeddings(df: pd.DataFrame, sequence_column: str, pos_column: str, model, tokenizer, device) -> np.ndarray:
    """
    Generates embeddings for a specific residue of interest from full protein sequences.
    """
    all_embeddings = []
    
    print(f"Generating embeddings for column: '{sequence_column}'...")
    
    # Process the dataframe in batches
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch_df = df.iloc[i:i+BATCH_SIZE]
        
        sequences = batch_df[sequence_column].tolist()
        positions = batch_df[pos_column].astype(int).tolist()

        # Tokenize the batch of sequences
        inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings_tensor = outputs.last_hidden_state.cpu()
        
        for j, pos in enumerate(positions):
            # The model adds a <cls> token at the beginning (index 0).
            # So, the amino acid at string position `k` is at tensor index `k+1`.
            # Since our 'pos' column is 1-indexed, the amino acid at `pos` is at tensor index `pos`.
            # The check `pos < MAX_LENGTH` before calling this function ensures this is safe.
            target_embedding = embeddings_tensor[j, pos, :].numpy()
            all_embeddings.append(target_embedding)
            
    return np.array(all_embeddings)


if __name__ == "__main__":
    print("--- Steps 4 & 5: Generate Embeddings ---")
    
    # --- 1. Setup Environment ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 2. Load Model and Tokenizer ---
    print(f"Loading model: {MODEL_NAME}...")
    # Note: The 'UNEXPECTED' and 'MISSING' keys in the load report are normal and expected
    # when loading EsmModel without a specific head. This can be safely ignored.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # --- 3. Define Paths and Load Data ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train_processed.csv")
    PROCESSED_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test_processed.csv")
    EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_PATH)

    # --- 4. Filter out-of-bounds sites ---
    print("\nFiltering out sites beyond the model's maximum length...")
    initial_train_count = len(train_df)
    initial_test_count = len(test_df)
    
    train_df = train_df[train_df['pos'] < MAX_LENGTH].reset_index(drop=True)
    test_df = test_df[test_df['pos'] < MAX_LENGTH].reset_index(drop=True)
    
    print(f"Train set: Removed {initial_train_count - len(train_df)} sites. Kept {len(train_df)} sites.")
    print(f"Test set:  Removed {initial_test_count - len(test_df)} sites. Kept {len(test_df)} sites.")

    # --- 5. Generate and Save Embeddings ---
    
    # For Training Data
    train_real_embeddings = generate_embeddings(train_df, 'full_sequence', 'pos', model, tokenizer, device)
    np.save(os.path.join(EMBEDDINGS_DIR, "train_real_embeddings.npy"), train_real_embeddings)
    print(f"Saved train_real_embeddings.npy to {EMBEDDINGS_DIR}")

    train_shuffled_embeddings = generate_embeddings(train_df, 'shuffled_full_sequence', 'pos', model, tokenizer, device)
    np.save(os.path.join(EMBEDDINGS_DIR, "train_shuffled_embeddings.npy"), train_shuffled_embeddings)
    print(f"Saved train_shuffled_embeddings.npy to {EMBEDDINGS_DIR}")

    # For Testing Data
    test_real_embeddings = generate_embeddings(test_df, 'full_sequence', 'pos', model, tokenizer, device)
    np.save(os.path.join(EMBEDDINGS_DIR, "test_real_embeddings.npy"), test_real_embeddings)
    print(f"Saved test_real_embeddings.npy to {EMBEDDINGS_DIR}")

    test_shuffled_embeddings = generate_embeddings(test_df, 'shuffled_full_sequence', 'pos', model, tokenizer, device)
    np.save(os.path.join(EMBEDDINGS_DIR, "test_shuffled_embeddings.npy"), test_shuffled_embeddings)
    print(f"Saved test_shuffled_embeddings.npy to {EMBEDDINGS_DIR}")

    print("\nEmbedding generation complete.")
