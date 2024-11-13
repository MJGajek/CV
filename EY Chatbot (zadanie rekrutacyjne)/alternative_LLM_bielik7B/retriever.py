# retriever.py

import faiss
import json
from sentence_transformers import SentenceTransformer
import time
import numpy as np

def initialize_retriever(faiss_path, metadata_path, model_name='distiluse-base-multilingual-cased'):
    """Load FAISS index, metadata, and initialize the embedding model."""
    try:
        index = faiss.read_index(faiss_path)
        print("FAISS index loaded successfully.")
    except Exception as e:
        print("Error loading FAISS index:", e)
        index = None

    metadata = None
    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        print("Metadata loaded successfully.")
    except Exception as e:
        print("Error loading metadata:", e)

    try:
        retriever_model = SentenceTransformer(model_name)
        print("Retriever model initialized successfully.")
    except Exception as e:
        print("Error loading retriever model:", e)
        retriever_model = None

    return index, metadata, retriever_model

def retrieve_top_chunks(query, index, metadata, retriever_model, initial_top_k=30, final_top_k=3, verbose=True):
    """Retrieve the top-k chunks that best match the query using FAISS."""
    # Encode the query
    query_embedding = retriever_model.encode(query).reshape(1, -1)

    # Retrieve initial set of candidate chunks
    distances, indices = index.search(query_embedding, initial_top_k)
    retrieval_time = time.time()

    retrieved_chunks = []
    for i, idx in enumerate(indices[0]):
        # Ensure that idx is within bounds of metadata (in case of index issues)
        if idx < len(metadata):
            chunk = metadata[idx]
            chunk_info = {
                "distance": distances[0][i],
                "start": chunk["start"],
                "end": chunk["end"],
                "speaker": chunk["speaker"],
                "text": chunk["text"]
            }
            retrieved_chunks.append(chunk_info)

    # Apply the final top-k filtering here explicitly
    retrieved_chunks = sorted(retrieved_chunks, key=lambda x: x["distance"])  # Sort by distance, closest first
    retrieved_chunks = retrieved_chunks[:final_top_k]  # Limit to final_top_k items

    # Print to check after filtering to final_top_k
    print(f"Filtered chunks count (should be {final_top_k}): {len(retrieved_chunks)}")
    
    # Verbose mode to print additional debug information, if needed
    if verbose:
        avg_distance = np.mean(distances[0]) if distances[0].size > 0 else float('inf')
        final_top_k_distances = distances[0][:final_top_k]
        avg_final_top_k_distance = np.mean(final_top_k_distances) if final_top_k_distances.size > 0 else float('inf')
        
        # Comment out or enable print statements as needed
        # print(f"Query: {query}")
        # print(f"Retrieved in {retrieval_time:.4f} seconds")
        # print(f"Average similarity distance (initial top-k): {avg_distance:.4f}")
        # print(f"Average similarity distance (final top-k): {avg_final_top_k_distance:.4f}")
        # print(f"Results count after filtering: {len(retrieved_chunks)}\n")

    return retrieved_chunks
