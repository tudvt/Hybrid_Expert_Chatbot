from sentence_transformers import SentenceTransformer
import faiss
import pickle
from utils.ner import extract_entities
from utils.search_utils import merge_results

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

def search_documents(query, k=4, disable_entities=False):
    """
    Search for documents similar to the query
    Args:
        query (str): The search query
        k (int): Number of results to return
        disable_entities (bool): If True, only use base semantic search
    Returns:
        list: List of relevant text chunks with scores and entities
    """
    # Load the system
    index, chunks, embed_model = load_search_system()
    if not all([index, chunks, embed_model]):
        return []
    
    # Base query embedding and search
    base_emb = embed_model.encode([query])
    base_results = index.search(base_emb, k)
    final_D, final_I = base_results
    
    # Extract entities and perform enhanced search if enabled
    entities = None if disable_entities else extract_entities(query)
    
    if entities:
        # Create enhanced query with entities
        enhanced_parts = [query]  # Start with original query
        for ent_type, ent_values in entities.items():
            if isinstance(ent_values, list):
                enhanced_parts.extend(ent_values)
            else:
                enhanced_parts.append(ent_values)
        enhanced_query = ' '.join(enhanced_parts)
        
        # Search with enhanced query
        enhanced_emb = embed_model.encode([enhanced_query])
        enhanced_results = index.search(enhanced_emb, k)
        
        # Merge results
        final_D, final_I = merge_results(base_results, enhanced_results)
    
    # Get final results using the merged D, I
    
    # Return the results with entities
    results = [{
        "text": chunks[i]["text"], 
        "score": float(d),
        "entities": extract_entities(chunks[i]["text"]),
        "query_type": "hybrid" if entities else "base"}
        for i, d in zip(final_I[0], final_D[0])]
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Search through documents using FAISS index")
    parser.add_argument("query", help="The search query")
    parser.add_argument("--top_k", type=int, default=4, help="Number of results to return")
    parser.add_argument("--basic", action="store_true", help="Use basic semantic search without entity enhancement")
    parser.add_argument("--enhanced", action="store_true", help="Use entity-enhanced search (default)")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for user input after showing results")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"Searching for: {args.query}")
    search_type = "Basic semantic search" if args.basic else "Entity-enhanced search"
    print(f"Search type: {search_type}")
    print("="*80 + "\n")

    # Get results with appropriate search type
    results = search_documents(args.query, k=args.top_k, disable_entities=args.basic)
    
    if not results:
        print("No results found or error loading search system.")
    else:
        for i, result in enumerate(results, 1):
            print(f"\n=== Result {i} (distance: {result['score']:.2f}) ===")
            
            # Only show entities for enhanced search
            if not args.basic and result['entities']:
                print("\nEntities found:", result['entities'])
            
            print("\nText:", result["text"])
            print("-"*80)
    
    if not args.no_wait:
        input("\nPress Enter to exit...")