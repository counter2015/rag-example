from typing import List
import numpy as np


def normalize_embedding(embedding: List[float]) -> List[float]:
    embedding_values_np = np.array(embedding)
    normed_embedding = embedding_values_np / np.linalg.norm(embedding_values_np)
    return normed_embedding
