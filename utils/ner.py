import spacy
import re
import yaml
from pathlib import Path

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Load patterns from config
config_path = Path(__file__).parent.parent / "config" / "entity_patterns.yaml"
with open(config_path, 'r') as f:
    PATTERNS = yaml.safe_load(f)

def extract_entities(text):
    """
    Extract named entities and technical patterns from text
    
    Args:
        text (str): Input text to process
        
    Returns:
        dict: Dictionary of entity types and their values
    """
    doc = nlp(text)
    entities = {}
    
    # Extract spaCy named entities
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        if ent.text not in entities[ent.label_]:
            entities[ent.label_].append(ent.text)
    
    # Apply custom patterns from config
    for pattern_name, pattern_info in PATTERNS['patterns'].items():
        # Handle regex patterns
        if 'pattern' in pattern_info:
            matches = re.findall(pattern_info['pattern'], text, re.IGNORECASE)
            if matches:
                # Flatten if the matches are tuples (from regex groups)
                matches = [m[0] if isinstance(m, tuple) else m for m in matches]
                # Remove duplicates while preserving order
                unique_matches = []
                for m in matches:
                    if m not in unique_matches:
                        unique_matches.append(m)
                entities[pattern_name.upper()] = unique_matches
        
        # Handle custom term lists
        if 'custom_terms' in pattern_info:
            matches = []
            for term in pattern_info['custom_terms']:
                if term.lower() in text.lower():
                    matches.append(term)
            if matches:
                entities[pattern_name.upper()] = matches
    
    return entities