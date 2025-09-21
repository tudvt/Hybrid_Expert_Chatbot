from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import pickle
import os
import shutil
from datetime import datetime

def check_required_files():
    """Check if chunks.json exists before proceeding"""
    if not os.path.exists("data/chunks.json"):
        raise FileExistsError("chunks.json not found! Please run chunking process first.")

def backup_files():
    """Backup existing index and metadata files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join("data", "backups", timestamp)
    os.makedirs(backup_dir, exist_ok=True)
    
    for file in ["faiss_index.idx", "chunks_meta.pkl"]:
        src = os.path.join("data", file)
        if os.path.exists(src):
            dst = os.path.join(backup_dir, file)
            shutil.copy2(src, dst)
            print(f"Backed up {file} to {backup_dir}")
    return backup_dir

def restore_backup(backup_dir):
    """Restore index and metadata from a backup"""
    print(f"Restoring from backup: {backup_dir}")
    for file in ["faiss_index.idx", "chunks_meta.pkl"]:
        src = os.path.join(backup_dir, file)
        if os.path.exists(src):
            dst = os.path.join("data", file)
            shutil.copy2(src, dst)
            print(f"Restored {file} from backup")
        else:
            print(f"Warning: {file} not found in backup directory")

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)
os.makedirs(os.path.join("data", "backups"), exist_ok=True)

# Check and backup
print("Checking required files...")
check_required_files()
print("Creating backups...")
backup_files()

# Load chunks
print("Loading chunks...")
with open("data/chunks.json", "r", encoding='utf-8') as f:
    chunks = json.load(f)
texts = [c["text"] for c in chunks]

# Initialize the embedding model
print("Initializing embedding model...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings
print("Generating embeddings...")
embeddings = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# Create and save FAISS index
print("Creating FAISS index...")
dim = embeddings.shape[1]  # Get the dimension of the embeddings
index = faiss.IndexFlatL2(dim)  # Create a new index using L2 distance
index.add(embeddings)  # Add vectors to the index

# Save the FAISS index
print("Saving FAISS index...")
faiss.write_index(index, "data/faiss_index.idx")

# Save metadata (chunks) for later retrieval
print("Saving metadata...")
with open("data/chunks_meta.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Done! Index and metadata saved successfully.")

# Example query function
def search_documents(query, k=4):
    """
    Search for similar documents using the FAISS index
    Args:
        query (str): The search query
        k (int): Number of results to return
    Returns:
        list: List of relevant text chunks
    """
    # Load the index and metadata
    index = faiss.read_index("data/faiss_index.idx")
    with open("data/chunks_meta.pkl", "rb") as f:
        chunks = pickle.load(f)
    
    # Encode the query
    query_emb = embed_model.encode([query])
    
    # Search the index
    D, I = index.search(query_emb, k)
    
    # Return the results
    results = [{"text": chunks[i]["text"], "score": float(d)} 
              for i, d in zip(I[0], D[0])]
    return results

# Test the search if running as main script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', help='Restore from a specific backup directory')
    parser.add_argument('--query', default="How to fix installation errors?", 
                       help='Test query to run after build/restore')
    args = parser.parse_args()

    if args.restore:
        restore_backup(args.restore)
    else:
        # Regular build process
        print("Building new index...")
        backup_dir = backup_files()
        
        try:
            # Load chunks and build index
            print("Loading chunks...")
            with open("data/chunks.json", "r", encoding='utf-8') as f:
                chunks = json.load(f)
            texts = [c["text"] for c in chunks]
            
            print("Generating embeddings...")
            embeddings = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            
            print("Creating FAISS index...")
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
            
            print("Saving files...")
            faiss.write_index(index, "data/faiss_index.idx")
            with open("data/chunks_meta.pkl", "wb") as f:
                pickle.dump(chunks, f)
                
            print("Build completed successfully!")
        except Exception as e:
            print(f"Error during build: {str(e)}")
            print("Restoring from backup...")
            restore_backup(backup_dir)
            print("Restored from backup due to error")
    
    # Test the search
    print("\nTesting search functionality...")
    print(f"Query: {args.query}")
    results = search_documents(args.query)
    print("\nResults:")
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (distance: {result['score']:.2f}) ---")
        print(result["text"])