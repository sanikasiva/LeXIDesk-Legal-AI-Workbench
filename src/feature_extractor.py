# In src/feature_extractor.py

import re

def get_token_signature(token):
    """Generates a 'signature' of a token by replacing character types."""
    return "".join(['C' if char.isupper() else 'c' if char.islower() else 'D' if char.isdigit() else char for char in token])

def get_token_length_feature(token):
    """Categorizes token length."""
    length = len(token)
    if length < 4:
        return 'short'
    elif length <= 6:
        return 'normal'
    else:
        return 'long'

def token_to_features(token, text, start, end):
    """
    The core feature extraction function.
    This version uses GENERALIZED PATTERNS instead of a hardcoded list.
    """
    # Basic contextual features
    char_before = text[start - 1] if start > 0 else '<BOS>'
    char_after = text[end] if end < len(text) else '<EOS>'
    
    # --- GENERALIZED ABBREVIATION PATTERNS ---
    # We create several boolean flags based on the token's structure.
    
    # Pattern 1: Is it a single capital letter with a period? (e.g., "S.")
    is_initial = re.fullmatch(r'[A-Z]\.', token) is not None
    
    # Pattern 2: Is it an acronym like "U.S." or "F.B.I."?
    # (A sequence of one or more capital letters-and-periods)
    is_acronym = re.fullmatch(r'([A-Z]\.)+', token) is not None
    
    # Pattern 3: Is it a title like "Mr." or "Dr."?
    is_title_like = re.fullmatch(r'(Mr|Mrs|Ms|Dr)\.', token, re.IGNORECASE) is not None
    
    # We can combine these into a single powerful feature for the model
    is_likely_abbreviation = is_initial or is_acronym or is_title_like

    features = {
        'bias': 1.0,
        'token': token,
        'lower': token.lower(),
        'sig': get_token_signature(token),
        'len_cat': get_token_length_feature(token),
        'is_lower': token.islower(),
        'is_upper': token.isupper(),
        'is_title': token.istitle(),
        'is_digit': token.isdigit(),
        'char_before': char_before,
        'is_space_before': char_before.isspace(),
        'char_after': char_after,
        'is_space_after': char_after.isspace(),
        
        # --- THIS IS THE NEW GENERALIZED FEATURE ---
        # It replaces the hardcoded list lookup.
        'is_likely_abbreviation': is_likely_abbreviation,
        
        # We can still keep other helpful patterns
        'ends_with_period': token.endswith('.'),
        'is_numeric_with_period': re.fullmatch(r'\d+\.', token) is not None,
    }
    
    # Is the character after this token a capital letter? (Still a very strong signal)
    if char_after.isupper():
        features['next_char_is_upper'] = True
        
    return features

# ... (all other functions in the file stay the same) ...

def add_neighboring_token_features(sentence_features):
    """
    Adds features from neighboring tokens.
    This corrected version builds a new list of features to prevent the
    "snowball" effect that causes MemoryErrors.
    """
    # Create a new list to store the output, preventing in-place modification bugs.
    expanded_sentence_features = []
    
    # Pad the features with special markers for the beginning and end
    # This makes the loop simpler and avoids lots of if/else checks inside.
    sentence_features = [None] + sentence_features + [None]

    for i in range(1, len(sentence_features) - 1):
        # Create a fresh copy of the current token's features
        new_features = sentence_features[i].copy()
        
        # Features for the previous token
        prev_feats = sentence_features[i-1]
        if prev_feats is not None:
            # Always read from the original, simple feature dictionaries
            for key, value in prev_feats.items():
                new_features['-1:' + key] = value
        else:
            # It's the beginning of the sentence
            new_features['BOS'] = True

        # Features for the next token
        next_feats = sentence_features[i+1]
        if next_feats is not None:
            # Always read from the original, simple feature dictionaries
            for key, value in next_feats.items():
                new_features['+1:' + key] = value
        else:
            # It's the end of the sentence
            new_features['EOS'] = True
            
        expanded_sentence_features.append(new_features)
        
    return expanded_sentence_features