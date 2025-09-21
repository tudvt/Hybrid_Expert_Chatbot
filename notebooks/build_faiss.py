from sentence_transformers import SentenceTransformer
import faiss
import pickle
import argparse
import os
import time

def load_resources():
    print("\n=== Resource Loading Process ===")
    start_time = time.time()
    
    # Check if model is in cache
    cache_dir = os.path.expanduser('~/.cache/torch/sentence_transformers')
    model_name = "all-MiniLM-L6-v2"
    is_cached = os.path.exists(os.path.join(cache_dir, model_name))
    
    if not is_cached:
        print("First run: Downloading model from HuggingFace Hub...")
        print("This may take a few minutes depending on your internet speed...")
    else:
        print("Using cached model from previous run...")
    
    embed_model = SentenceTransformer(f"sentence-transformers/{model_name}")
    
    print("\nLoading FAISS index...")
    index = faiss.read_index("data/faiss_index.idx")
    
    print("Loading metadata...")
    with open("data/chunks_meta.pkl", "rb") as f:
        chunks = pickle.load(f)
        
    load_time = time.time() - start_time
    print(f"\nAll resources loaded in {load_time:.2f} seconds")
    print("=" * 40)
    return embed_model, index, chunks

def search_documents(query, k=4):
    embed_model, index, chunks = load_resources()
    
    print(f"\nSearching for: '{query}'")
    query_emb = embed_model.encode([query])
    D, I = index.search(query_emb, k)
    
    results = []
    for i, (dist, idx) in enumerate(zip(D[0], I[0]), 1):
        results.append({
            "rank": i,
            "text": chunks[idx]["text"],
            "score": float(dist)
        })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="The search query")
    parser.add_argument("--top_k", type=int, default=4, help="Number of results to return")
    args = parser.parse_args()
    
    results = search_documents(args.query, args.top_k)
    
    print("\nSearch Results:")
    print("=" * 80)
    for result in results:
        print(f"\nResult #{result['rank']} (Distance: {result['score']:.2f})")
        print("-" * 40)
        print(result['text'])
        print("-" * 40)
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()