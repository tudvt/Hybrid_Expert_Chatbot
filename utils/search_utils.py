def merge_results(base_results, enhanced_results, threshold=0.9):
    """
    Merge and deduplicate search results from base and enhanced queries
    
    Args:
        base_results: Results from base query (D, I tuples)
        enhanced_results: Results from enhanced query (D, I tuples)
        threshold: Similarity threshold for deduplication
        
    Returns:
        tuple: Combined and deduplicated (distances, indices)
    """
    combined_D = list(base_results[0][0])  # distances
    combined_I = list(base_results[1][0])  # indices
    
    # Add enhanced results if they're significantly different
    for d, i in zip(enhanced_results[0][0], enhanced_results[1][0]):
        if i not in combined_I:  # If index not already included
            combined_D.append(d)
            combined_I.append(i)
    
    # Sort by distance (lower is better)
    sorted_pairs = sorted(zip(combined_D, combined_I))
    sorted_D, sorted_I = zip(*sorted_pairs)
    
    return [list(sorted_D)], [list(sorted_I)]