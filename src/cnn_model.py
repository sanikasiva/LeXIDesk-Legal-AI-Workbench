# In src/cnn_model.py

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import numpy as np

# --- 1. Configuration and Constants ---
# Hyperparameters from the paper (Sections 4.1.5 and 5.1)
CONTEXT_WINDOW_SIZE = 6
EMBEDDING_DIM = 128
HIDDEN_DIM = 250
NUM_FILTERS = 6
KERNEL_SIZE = 5
DROPOUT_PROB = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 15 # Paper found this was optimal for the CNN

# Paths to our processed data
TRAIN_CSV_PATH = 'data/processed/train_data.csv'
TEST_CSV_PATH = 'data/processed/test_data.csv'

# --- 2. Create a Custom PyTorch Dataset ---
class SBDDataset(Dataset):
    def __init__(self, csv_file, char_to_idx, max_len):
        self.data = pd.read_csv(csv_file)
        self.char_to_idx = char_to_idx
        self.max_len = max_len
        self.pad_idx = char_to_idx['<PAD>']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Combine contexts and delimiter into a single string
        left = str(row['left_context'])
        delimiter = str(row['delimiter'])
        right = str(row['right_context'])
        
        sample_text = left + delimiter + right
        
        # Convert text to numerical indices
        indexed_text = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in sample_text]
        
        # Pad the sequence to max_len
        padded_text = indexed_text[:self.max_len] + [self.pad_idx] * (self.max_len - len(indexed_text))
        
        # Convert to tensors
        text_tensor = torch.tensor(padded_text, dtype=torch.long)
        label_tensor = torch.tensor(row['label'], dtype=torch.float32)
        
        return text_tensor, label_tensor

# --- 3. Define the CNN Model Architecture ---
# As described in Section 4.1.5 and Figure 2 of the paper
class LegalSBD_CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_size, hidden_dim, dropout_prob):
        super(LegalSBD_CNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 1D Convolutional Layer
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size)
        
        # ReLU activation is applied after convolution
        self.relu = nn.ReLU()
        
        # Global Max Pooling is implicitly handled by adaptive pooling or a manual max operation
        
        self.dropout = nn.Dropout(dropout_prob)
        
        # Hidden Layer and Output Layer
        self.fc1 = nn.Linear(num_filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # Output is a single value for binary classification
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        # text shape: [batch_size, sequence_length]
        embedded = self.embedding(text)
        # embedded shape: [batch_size, sequence_length, embedding_dim]
        
        # Conv1d expects [batch_size, in_channels, sequence_length], so we permute
        embedded = embedded.permute(0, 2, 1)
        # embedded shape: [batch_size, embedding_dim, sequence_length]
        
        conved = self.conv1d(embedded)
        # conved shape: [batch_size, num_filters, new_sequence_length]
        
        activated = self.relu(conved)
        
        # Global Max Pooling over the sequence dimension
        pooled = torch.max(activated, dim=2).values
        # pooled shape: [batch_size, num_filters]
        
        # Pass through the fully connected layers
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Apply sigmoid to get a probability
        output = self.sigmoid(x)
        
        return output.squeeze(-1)


# --- 4. Training and Evaluation Functions ---
def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for text, labels in tqdm(data_loader, desc="Training"):
        text, labels = text.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(text)
        
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Store predictions and labels for metric calculation
        preds = (predictions > 0.5).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    
    return avg_loss, precision, recall, f1

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for text, labels in tqdm(data_loader, desc="Evaluating"):
            text, labels = text.to(device), labels.to(device)
            
            predictions = model(text)
            loss = criterion(predictions, labels)
            
            total_loss += loss.item()

            preds = (predictions > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    
    return avg_loss, precision, recall, f1

# --- 5. Main Execution Block ---
if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build Vocabulary from training data
    print("Building vocabulary...")
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    all_chars = set()
    for index, row in train_df.iterrows():
        all_chars.update(str(row['left_context']))
        all_chars.update(str(row['delimiter']))
        all_chars.update(str(row['right_context']))
    
    char_to_idx = {char: i+2 for i, char in enumerate(sorted(list(all_chars)))}
    char_to_idx['<PAD>'] = 0
    char_to_idx['<UNK>'] = 1 # For unknown characters in the test set
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    # The max length of our input sequence (left + delimiter + right)
    max_len = (CONTEXT_WINDOW_SIZE * 2) + 1

    # Create Datasets and DataLoaders
    print("Creating datasets and dataloaders...")
    train_dataset = SBDDataset(TRAIN_CSV_PATH, char_to_idx, max_len)
    test_dataset = SBDDataset(TEST_CSV_PATH, char_to_idx, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, optimizer, and loss function
    print("Initializing model...")
    model = LegalSBD_CNN(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        hidden_dim=HIDDEN_DIM,
        dropout_prob=DROPOUT_PROB
    ).to(device)
    
    # Optimizer and Loss from the paper
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_p, train_r, train_f1 = train_model(model, train_loader, optimizer, criterion, device)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train P: {train_p:.4f} | Train R: {train_r:.4f} | Train F1: {train_f1:.4f}")

    print("\nTraining finished. Starting final evaluation on the test set...")
    test_loss, test_p, test_r, test_f1 = evaluate_model(model, test_loader, criterion, device)
    print("\n--- Test Set Performance ---")
    print(f"  Test Loss: {test_loss:.4f} | Test P: {test_p:.4f} | Test R: {test_r:.4f} | Test F1: {test_f1:.4f}")

    # Optional: Save the trained model
    torch.save(model.state_dict(), 'saved_models/cnn_model.pth')
    print("\nModel saved to 'saved_models/cnn_model.pth'")