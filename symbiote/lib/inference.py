#
# Method to perform inference given an embedding an a chroma collection
#

import numpy as np
from chromadb import Collection


# The threshold (in cosine similarity difference) between two embeddings to consider them the same
DETECT_THRESHOLD: float = 0.05

def perform_inference(
    embedding: np.ndarray,
    collection: Collection
) -> tuple[int, int]:
    """Perform inference on the embedding by querying the collection and deciding the label
    based on the resulting queried vectors"""
    results = collection.query(query_embeddings=embedding)
    print(results)
    picklists = [results["metadatas"][0][i]["picklist"] for i in range(len(results["ids"][0])) if results["distances"][0][i] <= DETECT_THRESHOLD]
    bins = {}
    for picklist in picklists:
        for c in picklist:
            if c in bins:
                bins[c] += 1
            else:
                bins[c] = 1
    return (max(bins, key=bins.get), len(picklists))
