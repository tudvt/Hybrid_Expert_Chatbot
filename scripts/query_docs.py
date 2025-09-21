from sentence_transformers import SentenceTransformer
import faiss
import pickle

def load_search_system():
    """Load the FAISS index and metadata"""
    try:
        print("\n=== Loading Search System ===", flush=True)
        
        print("\nInitializing and loading model...", flush=True)
        # Load the embedding model - will use cache if available
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embed_model = SentenceTransformer(model_name)
            
        # Load the embedding model
        print("Initializing embedding model...", flush=True)
        embed_model = SentenceTransformer(model_name)
        
        # Load the FAISS index
        print("Loading FAISS index...", flush=True)
        index = faiss.read_index("data/faiss_index.idx")
        
        # Load the metadata
        print("Loading metadata...", flush=True)
        with open("data/chunks_meta.pkl", "rb") as f:
            chunks = pickle.load(f)
            
        print("=== System Loaded Successfully ===\n", flush=True)
        return index, chunks, embed_model
    except Exception as e:
        print(f"Error loading search system: {str(e)}")
        return None, None, None

def search_documents(query, k=4):
    """
    Search for documents similar to the query
    Args:
        query (str): The search query
        k (int): Number of results to return
    Returns:
        list: List of relevant text chunks with scores
    """
    # Load the system
    index, chunks, embed_model = load_search_system()
    if not all([index, chunks, embed_model]):
        return []
    
    # Encode the query
    query_emb = embed_model.encode([query])
    
    # Search the index
    D, I = index.search(query_emb, k)
    
    # Return the results
    results = [{"text": chunks[i]["text"], "score": float(d)} 
              for i, d in zip(I[0], D[0])]
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Search through documents using FAISS index")
    parser.add_argument("query", help="The search query")
    parser.add_argument("--top_k", type=int, default=4, help="Number of results to return")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for user input after showing results")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"Searching for: {args.query}")
    print(f"Top {args.top_k} results:")
    print("="*80 + "\n")
    
    results = search_documents(args.query, k=args.top_k)
    
    if not results:
        print("No results found or error loading search system.")
    else:
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} (distance: {result['score']:.2f}) ---")
            print(result["text"])
            print("-"*80)
    
    if not args.no_wait:
        input("\nPress Enter to exit...")