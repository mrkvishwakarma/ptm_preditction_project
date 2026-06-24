"""Code for Steps 1 and 2 data prep"""
import pandas as pd
import os
import random

def create_randomized_window(seq: str) -> str:
    """
    Creates a randomized version of an 11-aa window, preserving the center residue.
    
    Args:
        seq: An 11-amino acid sequence string.
        
    Returns:
        A new 11-amino acid sequence with the flanks shuffled.
    """
    if not isinstance(seq, str) or len(seq) != 51:
        # Assuming a window size of 51 based on the data
        return seq
        
    center_residue_index = 25  # Center of a 51-char sequence
    center_residue = seq[center_residue_index]
    
    flanks = list(seq[:center_residue_index] + seq[center_residue_index+1:])
    random.shuffle(flanks)
    
    shuffled_flanks_str = "".join(flanks)
    
    new_seq = (
        shuffled_flanks_str[:center_residue_index] + 
        center_residue + 
        shuffled_flanks_str[center_residue_index:]
    )
    return new_seq

def reconstruct_full_sequence(row):
    """
    Replaces the original window in the full sequence with the shuffled one,
    handling boundary conditions for sequences near the start/end of a protein.
    """
    full_seq = row['full_sequence']
    shuffled_window = row['shuffled_Seq']
    pos = int(row['pos'])
    
    window_radius = 25
    
    # Determine the actual start and end of the window in the full sequence
    actual_start_in_full = max(0, pos - 1 - window_radius)
    actual_end_in_full = min(len(full_seq), pos + window_radius)
    
    # Determine the corresponding start and end in the 51-char shuffled window
    start_in_shuffled = max(0, window_radius - (pos - 1))
    end_in_shuffled = start_in_shuffled + (actual_end_in_full - actual_start_in_full)
    
    # Extract the correct segment from the shuffled window
    segment_to_insert = shuffled_window[start_in_shuffled:end_in_shuffled]
    
    # Reconstruct the full sequence
    reconstructed_seq = (
        full_seq[:actual_start_in_full] + 
        segment_to_insert + 
        full_seq[actual_end_in_full:]
    )

    return reconstructed_seq

def load_and_prepare_data(raw_train_path: str, raw_test_path: str, output_dir: str):
    """
    Loads raw data, creates shuffled windows, and saves the prepared data.
    """
    print(f"Loading raw train data from {raw_train_path}...")
    train_df = pd.read_csv(raw_train_path)
    
    print(f"Loading raw test data from {raw_test_path}...")
    test_df = pd.read_csv(raw_test_path)
    
    # --- Step 3: Create Randomized Window ---
    print("\n--- Step 3: Creating Randomized Windows ---")
    print("Applying shuffling to training data...")
    train_df['shuffled_Seq'] = train_df['Seq'].apply(create_randomized_window)
    train_df['shuffled_full_sequence'] = train_df.apply(reconstruct_full_sequence, axis=1)
    
    print("Applying shuffling to testing data...")
    test_df['shuffled_Seq'] = test_df['Seq'].apply(create_randomized_window)
    test_df['shuffled_full_sequence'] = test_df.apply(reconstruct_full_sequence, axis=1)
    print("Shuffling complete.")

    # Save to processed directory
    os.makedirs(output_dir, exist_ok=True)
    processed_train_path = os.path.join(output_dir, "train_processed.csv")
    processed_test_path = os.path.join(output_dir, "test_processed.csv")
    
    train_df.to_csv(processed_train_path, index=False)
    test_df.to_csv(processed_test_path, index=False)
    
    print(f"\nData successfully prepared and saved to:")
    print(f" - {processed_train_path}")
    print(f" - {processed_test_path}")
    
    return train_df, test_df

if __name__ == "__main__":
    print("--- Steps 1, 2 & 3: Data Preparation and Shuffling ---")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_TRAIN = os.path.join(BASE_DIR, "data", "raw", "train.csv")
    RAW_TEST = os.path.join(BASE_DIR, "data", "raw", "test.csv")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    load_and_prepare_data(RAW_TRAIN, RAW_TEST, PROCESSED_DIR)
