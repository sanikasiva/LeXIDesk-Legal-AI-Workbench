# In src/crf_model.py (FINAL VERSION - SAVES BOTH MODELS)

import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from tqdm import tqdm
import re
import joblib # Import joblib for saving the CRF models

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Import our own modules
from src.cnn_model import LegalSBD_CNN, SBDDataset
from src.feature_extractor import token_to_features, add_neighboring_token_features

# --- 1. Configuration ---
CONTEXT_WINDOW_SIZE = 6
CNN_MODEL_PATH = 'saved_models/cnn_model.pth'
CRF_BASELINE_MODEL_PATH = 'saved_models/crf_baseline_model.joblib' # Path to save the baseline model
CRF_HYBRID_MODEL_PATH = 'saved_models/crf_hybrid_model.joblib'   # Path to save the hybrid model
PERFORMANCE_REPORT_PATH = 'saved_models/performance_report.json'
DELIMITERS = {'.', '?', '!', ';', ':'}
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

# CNN architecture constants
EMBEDDING_DIM = 128
HIDDEN_DIM = 250
NUM_FILTERS = 6
KERNEL_SIZE = 5
DROPOUT_PROB = 0.2

# --- 2. Helper Functions ---
def load_cnn_model(model_path, vocab_size, device):
    """Loads the trained CNN model from a file."""
    model = LegalSBD_CNN(vocab_size, EMBEDDING_DIM, NUM_FILTERS, KERNEL_SIZE, HIDDEN_DIM, DROPOUT_PROB).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def get_cnn_prediction_from_context(text, token_start_idx, cnn_model, char_to_idx, device):
    """
    Gets the CNN's prediction using the TRUE character context from the document.
    """
    token = text[token_start_idx]
    if token not in DELIMITERS:
        return 0.0

    start_left = max(0, token_start_idx - CONTEXT_WINDOW_SIZE)
    left_context = text[start_left : token_start_idx]

    end_right = token_start_idx + 1 + CONTEXT_WINDOW_SIZE
    right_context = text[token_start_idx + 1 : end_right]

    sample_text = left_context + token + right_context
    
    max_len = (CONTEXT_WINDOW_SIZE * 2) + 1
    pad_idx = char_to_idx['<PAD>']
    indexed_text = [char_to_idx.get(char, char_to_idx['<UNK>']) for char in sample_text]
    padded_text = indexed_text[:max_len] + [pad_idx] * (max_len - len(indexed_text))
    text_tensor = torch.tensor([padded_text], dtype=torch.long).to(device)

    with torch.no_grad():
        prediction = cnn_model(text_tensor).item()
        
    return prediction

def prepare_data_for_crf(file_path, cnn_model=None, char_to_idx=None, device=None):
    """
    Processes a raw .jsonl file into token sequences (X) and label sequences (y)
    for the CRF model.
    """
    X = []
    y = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Preparing CRF data from {file_path}"):
            data = json.loads(line)
            text = data['text']
            
            try:
                true_boundary_offsets = {span['end'] for span in data['spans']}
            except KeyError:
                continue

            #tokens_with_spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+|\n", text)]
            # This regex separates words from punctuation, which is critical.
            tokens_with_spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[\w'-]+|[.,!?;:()]|\S+", text)]

            if not tokens_with_spans:
                continue
            
            sentence_features = []
            for token, start, end in tokens_with_spans:
                features = token_to_features(token, text, start, end)
                
                if cnn_model and token in DELIMITERS:
                    delimiter_char_index = text.find(token, start)
                    if delimiter_char_index != -1:
                         cnn_prob = get_cnn_prediction_from_context(text, delimiter_char_index, cnn_model, char_to_idx, device)
                         features['cnn_prob'] = round(cnn_prob, 4)
                
                sentence_features.append(features)

            sentence_features = add_neighboring_token_features(sentence_features)
            
            labels = []
            for token, start, end in tokens_with_spans:
                if end in true_boundary_offsets and token in DELIMITERS:
                    labels.append('B')
                else:
                    labels.append('O')
            
            X.append(sentence_features)
            y.append(labels)
            
    return X, y

# --- 3. Main Execution Block ---
if __name__ == '__main__':
    # --- Part 1: Retrain and Save CNN Model ---
    print("--- Part 1: Retraining and Saving CNN Model ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_df = pd.read_csv('data/processed/train_data.csv')
    all_chars = set()
    for index, row in train_df.iterrows():
        all_chars.update(str(row['left_context']))
        all_chars.update(str(row['delimiter']))
        all_chars.update(str(row['right_context']))
    char_to_idx = {char: i+2 for i, char in enumerate(sorted(list(all_chars)))}
    char_to_idx['<PAD>'] = 0
    char_to_idx['<UNK>'] = 1
    vocab_size = len(char_to_idx)
    max_len = (CONTEXT_WINDOW_SIZE * 2) + 1
    
    train_dataset = SBDDataset('data/processed/train_data.csv', char_to_idx, max_len)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    cnn_model_to_save = LegalSBD_CNN(vocab_size, EMBEDDING_DIM, NUM_FILTERS, KERNEL_SIZE, HIDDEN_DIM, DROPOUT_PROB).to(device)
    optimizer = optim.Adam(cnn_model_to_save.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    cnn_model_to_save.train()
    for epoch in range(15):
        print(f"CNN pre-training epoch {epoch+1}/15...")
        for text, labels in tqdm(train_loader):
            text, labels = text.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = cnn_model_to_save(text)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
    
    torch.save(cnn_model_to_save.state_dict(), CNN_MODEL_PATH)
    print(f"CNN model saved to {CNN_MODEL_PATH}")

    # --- Part 2: Train and Evaluate CRF Models ---
    print("\n--- Part 2: Training and Evaluating CRF Models ---")
    
    raw_train_files = ['data/raw/CD_bva.jsonl', 'data/raw/CD_intellectual_property.jsonl', 'data/raw/CD_scotus.jsonl']
    raw_test_file = 'data/raw/CD_cyber_crime.jsonl'
    
    performance_scores = {}

    # 1. Baseline CRF
    print("\n[1] Preparing data for Baseline CRF model...")
    X_train_base, y_train_base = [], []
    for file in raw_train_files:
        X_docs, y_docs = prepare_data_for_crf(file)
        X_train_base.extend(X_docs)
        y_train_base.extend(y_docs)
    X_test_base, y_test_base = prepare_data_for_crf(raw_test_file)
    
    print("Training Baseline CRF model...")
    crf_base = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    crf_base.fit(X_train_base, y_train_base)
    
    print("\n--- Evaluating Baseline CRF model ---")
    y_pred_base = crf_base.predict(X_test_base)
    print(flat_classification_report(y_test_base, y_pred_base, labels=['B', 'O'], digits=4))

    # --- In src/crf_model.py ---

# --- NEW CORRECTED CODE for baseline metrics ---
# Flatten the lists of lists into single lists
    y_test_flat = [label for doc in y_test_base for label in doc]
    y_pred_flat = [label for doc in y_pred_base for label in doc]

# Calculate metrics for the 'B' class specifically
    p, r, f1, s = precision_recall_fscore_support(y_test_flat, y_pred_flat, labels=['B'], average="macro")
    accuracy = accuracy_score(y_test_flat, y_pred_flat)

    performance_scores['baseline_crf'] = {
        'precision_B': p,
        'recall_B': r,
        'f1_score_B': f1,
        'overall_accuracy': accuracy
    }
# --- END OF CORRECTION ---

    print("Saving the baseline CRF model to disk...")
    joblib.dump(crf_base, CRF_BASELINE_MODEL_PATH)
    print(f"Model saved to {CRF_BASELINE_MODEL_PATH}")

    # 2. Hybrid CNN-CRF
    print("\n[2] Preparing data for Hybrid CNN-CRF model...")
    cnn_model = load_cnn_model(CNN_MODEL_PATH, vocab_size, device)
    
    X_train_hybrid, y_train_hybrid = [], []
    for file in raw_train_files:
        X_docs, y_docs = prepare_data_for_crf(file, cnn_model=cnn_model, char_to_idx=char_to_idx, device=device)
        X_train_hybrid.extend(X_docs)
        y_train_hybrid.extend(y_docs)
    X_test_hybrid, y_test_hybrid = prepare_data_for_crf(raw_test_file, cnn_model=cnn_model, char_to_idx=char_to_idx, device=device)
    
    print("Training Hybrid CNN-CRF model...")
    crf_hybrid = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    crf_hybrid.fit(X_train_hybrid, y_train_hybrid)
    
    print("\n--- Evaluating Hybrid CNN-CRF model ---")
    y_pred_hybrid = crf_hybrid.predict(X_test_hybrid)
    print(flat_classification_report(y_test_hybrid, y_pred_hybrid, labels=['B', 'O'], digits=4))

    

# --- NEW CORRECTED CODE for hybrid metrics ---
# Flatten the lists for the hybrid model results
    y_test_hybrid_flat = [label for doc in y_test_hybrid for label in doc]
    y_pred_hybrid_flat = [label for doc in y_pred_hybrid for label in doc]

# Calculate metrics for the 'B' class
    p_h, r_h, f1_h, s_h = precision_recall_fscore_support(y_test_hybrid_flat, y_pred_hybrid_flat, labels=['B'], average="macro")
    accuracy_h = accuracy_score(y_test_hybrid_flat, y_pred_hybrid_flat)

    performance_scores['hybrid_cnn_crf'] = {
        'precision_B': p_h,
        'recall_B': r_h,
        'f1_score_B': f1_h,
        'overall_accuracy': accuracy_h
    }
# --- END OF CORRECTION ---

    print("Saving the hybrid CRF model to disk...")
    joblib.dump(crf_hybrid, CRF_HYBRID_MODEL_PATH)
    print(f"Model saved to {CRF_HYBRID_MODEL_PATH}")

    # --- NEW CODE: Final comparison and saving the report ---
    print("\n--- Final Performance Summary ---")
    print(json.dumps(performance_scores, indent=2))
    
    with open(PERFORMANCE_REPORT_PATH, 'w') as f:
        json.dump(performance_scores, f, indent=4)
        
    print(f"\nPerformance metrics saved to {PERFORMANCE_REPORT_PATH}")
    # --- END NEW CODE ---