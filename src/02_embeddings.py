import pandas as pd
import numpy as np
import torch
import re
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel

WINDOW_LEN = 51
CENTER_IDX = 25  # 0-indexed center of a 51-aa window (target residue)

MODEL_CONFIGS = {
    "ESM2_650M": {
        "name": "facebook/esm2_t33_650M_UR50D",
        "type": "esm",
        "dim": 1280,
        "batch_size": 8,
    },
    "ProtT5_XL": {
        "name": "Rostlab/prot_t5_xl_uniref50",
        "type": "t5",
        "dim": 1024,
        "batch_size": 4,  # ProtT5-XL is large; lower batch size
    },
}

def load_model(cfg, device):
    if cfg["type"] == "esm":
        tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
        model = AutoModel.from_pretrained(cfg["name"]).to(device)
    else:  # t5
        tokenizer = T5Tokenizer.from_pretrained(cfg["name"], do_lower_case=False)
        model = T5EncoderModel.from_pretrained(cfg["name"]).to(device)
    model.eval()
    # Use half precision on GPU to save memory (esp. for ProtT5-XL)
    if device.type == "cuda":
        model = model.half()
    return tokenizer, model

def prep_sequences(sequences, model_type):
    if model_type == "t5":
        # ProtT5 needs spaced residues + rare AA replacement
        cleaned = [re.sub(r"[UZOB]", "X", s) for s in sequences]
        return [" ".join(list(s)) for s in cleaned]
    return sequences  # ESM2 takes raw strings

def extract_center_embedding(seq_batch, tokenizer, model, model_type, device):
    inputs = tokenizer(seq_batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=WINDOW_LEN + 2).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state.float().cpu()

    embeddings = []
    for j in range(len(seq_batch)):
        if model_type == "esm":
            # ESM2 prepends <cls>: token index = CENTER_IDX + 1
            idx = CENTER_IDX + 1
        else:
            # ProtT5 has no leading special token, appends </s> at the end
            idx = CENTER_IDX
        embeddings.append(hidden[j, idx, :].numpy())
    return np.array(embeddings)

def generate_embeddings_for_model(df, sequence_column, model_key, device):
    cfg = MODEL_CONFIGS[model_key]
    tokenizer, model = load_model(cfg, device)
    batch_size = cfg["batch_size"]

    all_embeds = []
    print(f"Extracting {model_key} embeddings from '{sequence_column}'...")
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df[sequence_column].iloc[i:i+batch_size].tolist()
        prepped = prep_sequences(batch, cfg["type"])
        emb = extract_center_embedding(prepped, tokenizer, model, cfg["type"], device)
        all_embeds.append(emb)

    del model
    torch.cuda.empty_cache()
    return np.vstack(all_embeds)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train_processed.csv")
    PROCESSED_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test_processed.csv")
    EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_PATH)

    # Filter to valid 51-aa windows only (peptide-level, no full-sequence pos filtering needed)
    train_df = train_df[train_df["Seq"].str.len() == WINDOW_LEN].reset_index(drop=True)
    test_df = test_df[test_df["Seq"].str.len() == WINDOW_LEN].reset_index(drop=True)

    for model_key in MODEL_CONFIGS:
        for split_name, df, real_col, shuf_col in [
            ("train", train_df, "Seq", "shuffled_Seq"),
            ("test", test_df, "Seq", "shuffled_Seq"),
        ]:
            real_emb = generate_embeddings_for_model(df, real_col, model_key, device)
            np.save(os.path.join(EMBEDDINGS_DIR, f"{split_name}_real_{model_key}.npy"), real_emb)

            shuf_emb = generate_embeddings_for_model(df, shuf_col, model_key, device)
            np.save(os.path.join(EMBEDDINGS_DIR, f"{split_name}_shuffled_{model_key}.npy"), shuf_emb)

            print(f"Saved {split_name} embeddings for {model_key} (dim={MODEL_CONFIGS[model_key]['dim']})")

    print("\nDual-model embedding generation complete.")