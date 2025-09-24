from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import torch
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path to allow imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.query_docs import load_search_system, search_documents
from utils.ner import extract_entities

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchSystem:
    _instance = None
    
    def __init__(self):
        if SearchSystem._instance is None:
            logger.info("Loading FAISS search system...")
            self.faiss_index, self.chunks_meta, self.embed_model = load_search_system()
            logger.info("FAISS search system loaded successfully")
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

# Initialize model and tokenizer
MODEL_NAME = "google/flan-t5-base"
logger.info(f"Loading generation model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
logger.info(f"Model loaded and moved to device: {device} (CUDA available: {torch.cuda.is_available()})")
if torch.cuda.is_available():
    logger.info(f"GPU Model: {torch.cuda.get_device_name(0)}")

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Data directory: {DATA_DIR}")

def generate_answer(query: str, contexts: list[str] = None, max_length: int = 1024, max_new_tokens: int = 200, top_k: int = 4) -> str:
    """
    Generate an answer using Flan-T5 model based on the query and retrieved contexts.
    If contexts are not provided, it will use FAISS search to retrieve relevant contexts.
    """
    # If no contexts provided, search using FAISS directly (via SearchSystem singleton)
    if contexts is None:
        logger.info("No contexts provided, searching using FAISS...")

        # Extract entities for better search (optional, still useful)
        entities = extract_entities(query)
        logger.info(f"Extracted entities: {entities}")

        # Use SearchSystem singleton instead of calling search_documents
        search_system = SearchSystem.get_instance()
        index, chunks, embed_model = (
            search_system.faiss_index,
            search_system.chunks_meta,
            search_system.embed_model,
        )

        # Encode query
        base_emb = embed_model.encode([query])
        D, I = index.search(base_emb, top_k)

        # Collect contexts directly
        contexts = [chunks[i]["text"] for i in I[0]]
        logger.info(f"Retrieved {len(contexts)} context chunks from FAISS")
    # Construct the prompt with system instruction and context
    prompt = (
        "You are an expert Windows support assistant. Answer the question based on the "
        "provided context. Provide a detailed, comprehensive answer that covers all relevant points. "
        "Structure your answer with clear steps or bullet points when appropriate. "
        "Cite the source chunk indices [X] used in your answer.\n\n"
        "Context:\n"
    )
    
    # Add enumerated contexts
    for i, context in enumerate(contexts):
        prompt += f"[{i}] {context.strip()}\n"
    
    # Add the query
    prompt += f"\nQuestion: {query}\nAnswer:"
    
    # Tokenize input
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length
    ).to(device)
    
    # Generate answer
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,  # Beam search for better quality
            temperature=0.7,  # Add some randomness (0.7 is a good balance)
            no_repeat_ngram_size=2,  # Avoid repeating 2-grams
            early_stopping=True
        )
    
    # Decode and return the generated answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    # Example 1: With provided contexts
    logger.info("Example 1: Using provided contexts")
    sample_query = "How do I fix the Windows error 0x80070005?"
    sample_contexts = [
        "Error 0x80070005 typically indicates access denied. This occurs when you don't have the necessary permissions.",
        "To fix error 0x80070005: 1) Run as administrator 2) Check file permissions 3) Disable antivirus temporarily",
    ]
    
    answer = generate_answer(sample_query, sample_contexts)
    print(f"\nQuery: {sample_query}")
    print(f"Generated Answer: {answer}")
    
    # Example 2: Using FAISS search
    logger.info("\nExample 2: Using FAISS search")
    sample_query = "What are the steps to improve Windows performance?"
    
    answer = generate_answer(sample_query)  # Will automatically search using FAISS
    print(f"\nQuery: {sample_query}")
    print(f"Generated Answer: {answer}")