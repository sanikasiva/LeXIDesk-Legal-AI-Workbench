# In src/process_data.py

import json
import pandas as pd
from tqdm import tqdm # A library to show progress bars

def create_training_samples(file_path, context_window_size=6):
    """
    Reads a .jsonl file and processes it to create training samples for
    Sentence Boundary Detection.
    """
    processed_samples = []
    potential_delimiters = {'.', '?', '!', ';', ':'}

    print(f"Reading and processing file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing documents"):
            data = json.loads(line)
            text = data['text']
            
            # --- THIS IS THE FIXED PART ---
            # Instead of looking for 'sents_end_offs', we now build the set
            # of offsets by looping through the 'spans' list and getting the 'end' value.
            # Using a set comprehension for efficiency.
            try:
                true_boundary_offsets = {span['end'] for span in data['spans']}
            except KeyError:
                print(f"Warning: A document in {file_path} is missing the 'spans' key. Skipping this document.")
                continue # Move to the next line/document
            # --- END OF FIX ---

            for i, char in enumerate(text):
                if char in potential_delimiters:
                    # The offset is the position *after* the character.
                    # In this dataset, the 'end' offset in a span points directly
                    # to the position after the last character of the sentence.
                    # So, if a delimiter is at index `i`, its offset is `i + 1`.
                    # However, some datasets might define the 'end' offset as the
                    # index of the final character itself. Let's stick with the
                    # most common interpretation: offset = index + 1.
                    # The provided data's 'spans' seem to use this convention.
                    # Example: text "Hello." -> '.' is at index 5, offset is 6.
                    label = 1 if (i + 1) in true_boundary_offsets else 0
                    
                    start_left = max(0, i - context_window_size)
                    left_context = text[start_left : i]

                    end_right = i + 1 + context_window_size
                    right_context = text[i + 1 : end_right]

                    processed_samples.append({
                        'left_context': left_context,
                        'delimiter': char,
                        'right_context': right_context,
                        'label': label
                    })

    return pd.DataFrame(processed_samples)

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Using relative paths, which is the correct practice.
    # This assumes you run the script from the root 'LexiDesk' folder.
    file_paths = {
        'bva': 'C:/Users/acer/Downloads\LexiDesk\data/raw/CD_bva.jsonl',
        'ip': 'C:/Users/acer/Downloads/LexiDesk/data/raw/CD_intellectual_property.jsonl',
        'sc': 'C:/Users/acer/Downloads/LexiDesk/data/raw/CD_scotus.jsonl',
        'cc': 'C:/Users/acer/Downloads/LexiDesk/data/raw/CD_cyber_crime.jsonl'
    }

    all_dataframes = {}
    for name, path in file_paths.items():
        try:
            df = create_training_samples(path)
            all_dataframes[name] = df
            print(f"Finished processing {name}. Found {len(df)} samples.")
            print("Sample data:")
            print(df.head())
            print("\nClass distribution (1 = True boundary, 0 = False):")
            print(df['label'].value_counts(normalize=True)) # normalize=True gives percentages
            print("-" * 50)
        except FileNotFoundError:
            print(f"ERROR: File not found at '{path}'. Please check the path and filename.")
            print("Make sure your .jsonl files are inside the 'data/raw/' directory.")
            # Stop execution if a file is missing
            exit()

    try:
        train_df = pd.concat(
            [all_dataframes['bva'], all_dataframes['ip'], all_dataframes['sc']],
            ignore_index=True
        )
        
        # Saving processed data to the 'data/processed' directory
        train_df.to_csv('data/processed/train_data.csv', index=False)
        all_dataframes['cc'].to_csv('data/processed/test_data.csv', index=False)
        
        print(f"Successfully created 'data/processed/train_data.csv' with {len(train_df)} samples.")
        print(f"Successfully created 'data/processed/test_data.csv' with {len(all_dataframes['cc'])} samples.")
    except KeyError as e:
        print(f"\nCould not create final CSV files. A required dataframe is missing: {e}")