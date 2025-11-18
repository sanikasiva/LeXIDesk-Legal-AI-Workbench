# In predict.py (in the root LexiDesk folder)

import joblib
import re
import torch
import pandas as pd
from src.feature_extractor import token_to_features, add_neighboring_token_features
from src.cnn_model import LegalSBD_CNN # Import the CNN class definition
from src.crf_model import CONTEXT_WINDOW_SIZE, DELIMITERS # Import constants

# --- 1. Define Model Paths ---
BASELINE_MODEL_PATH = 'saved_models/crf_baseline_model.joblib'
HYBRID_MODEL_PATH = 'saved_models/crf_hybrid_model.joblib'
CNN_MODEL_PATH = 'saved_models/cnn_model.pth'

# --- 2. Load All Trained Models ---
print("Loading all trained models...")
try:
    # Load the CRF models
    baseline_crf_model = joblib.load(BASELINE_MODEL_PATH)
    hybrid_crf_model = joblib.load(HYBRID_MODEL_PATH)
    
    # --- Load components needed for the Hybrid Model's feature extraction ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Rebuild the character vocabulary exactly as done in training
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
    
    # Load the CNN model architecture and its trained weights
    cnn_model = LegalSBD_CNN(
        vocab_size=vocab_size, 
        embedding_dim=128, 
        num_filters=6, 
        kernel_size=5, 
        hidden_dim=250, 
        dropout_prob=0.2
    ).to(device)
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    cnn_model.eval()

except FileNotFoundError as e:
    print(f"Error: A model file was not found: {e}")
    print("Please run 'python -m src.crf_model' first to train and save all models.")
    exit()

print("All models loaded successfully.")

# --- 3. Prediction and Feature Extraction Functions ---

def get_cnn_prediction_from_context(text, token_start_idx, cnn_model, char_to_idx, device):
    """
    (This function remains the same)
    Identical to the function in crf_model.py. Gets the CNN's prediction.
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

 # --- CHANGE HERE: REPLACING THE ENTIRE segment_text FUNCTION ---

def segment_text(text, model, use_hybrid_features=False):
    """
    Takes raw text and uses a trained CRF model to split it into sentences.
    This version uses the CORRECT tokenizer and has ROBUST sentence reconstruction.
    """
    # 1. Get tokens and their spans using the CORRECT regex that separates punctuation.
    tokens_with_spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[\w'-]+|[.,!?;:()]|\S+", text)]
    
    if not tokens_with_spans:
        return []

    # 2. Extract features for the tokens, exactly as done in training.
    sentence_features = []
    for token, start, end in tokens_with_spans:
        features = token_to_features(token, text, start, end)
        
        # If using the hybrid model, generate and add the CNN feature
        if use_hybrid_features and token in DELIMITERS:
            # We need the character index, which is the start of the token span
            cnn_prob = get_cnn_prediction_from_context(text, start, cnn_model, char_to_idx, device)
            features['cnn_prob'] = round(cnn_prob, 4)
        
        sentence_features.append(features)
    
    sentence_features = add_neighboring_token_features(sentence_features)

    # 3. Predict the labels ('B' for Boundary, 'O' for Other) for the sequence
    labels = model.predict([sentence_features])[0]

    # 4. Reconstruct sentences based on the predicted 'B' labels. This logic is much cleaner.
    sentences = []
    current_sentence_start_index = 0
    for i, label in enumerate(labels):
        # A 'B' label means the token AT THIS INDEX is the end of a sentence.
        if label == 'B':
            # The sentence runs from the start index up to and including the current token.
            sentence_end_char_index = tokens_with_spans[i][2] # Get the 'end' span of the boundary token
            sentence_start_char_index = tokens_with_spans[current_sentence_start_index][1] # Get 'start' span
            
            # Slice the original text to get the sentence perfectly formatted.
            sentence = text[sentence_start_char_index:sentence_end_char_index].strip()
            sentences.append(sentence)
            
            # The next sentence will start at the next token.
            current_sentence_start_index = i + 1
    
    # After the loop, check if there are any leftover tokens that form a final sentence.
    if current_sentence_start_index < len(tokens_with_spans):
        # The final sentence runs from the start index to the very end of the text.
        sentence_start_char_index = tokens_with_spans[current_sentence_start_index][1]
        sentence = text[sentence_start_char_index:].strip()
        if sentence: # Make sure it's not just whitespace
             sentences.append(sentence)
        
    return sentences

# --- END OF CHANGES ---


# --- 4. Define a sample legal text and Run Predictions ---
sample_legal_text = """
The court's decision in Marbury v. Madison, 5 U.S. 137 (1803), established the principle of judicial review. This principle is outlined in § 1.3(a) of the legal code. The defendant, Mr. Smith, was subsequently charged under 18 U.S.C. § 1001. All proceedings were documented by the F.B.I. for review.
"""

print("\n" + "="*50)
print("--- Segmenting Sample Legal Text ---")
print(f"\nOriginal Text:\n---\n{sample_legal_text.strip()}\n---")

# --- Run prediction with the Baseline Model ---
print("\n--- Detected Sentences (Baseline CRF Model) ---")
baseline_sentences = segment_text(sample_legal_text, baseline_crf_model, use_hybrid_features=False)
for i, sent in enumerate(baseline_sentences):
    print(f"[{i+1}]: {sent}")

# --- Run prediction with the Hybrid Model ---
print("\n--- Detected Sentences (Hybrid CNN-CRF Model) ---")
hybrid_sentences = segment_text(sample_legal_text, hybrid_crf_model, use_hybrid_features=True)
for i, sent in enumerate(hybrid_sentences):
    print(f"[{i+1}]: {sent}")

print("\n" + "="*50)