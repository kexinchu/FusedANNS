"""
Similarity computation functions for MSTM (Multi-modal Similarity with Typed Sets)
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, kendalltau


class TypedSetSimilarity:
    """Unified typed-set similarity computation for MSTM"""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        """
        Args:
            alpha: Weight for primary modality
            beta: Weight for auxiliary modality
        """
        self.alpha = alpha
        self.beta = beta
    
    def max_sim(self, query_vectors: Dict[str, np.ndarray], 
                object_vectors: Dict[str, List[np.ndarray]]) -> float:
        """
        Multi-vector MaxSim: max similarity between query and object vectors
        Only considers target modality (no auxiliary)
        """
        if not query_vectors or not object_vectors:
            return 0.0
        
        max_sim = -np.inf
        for modality, q_vec in query_vectors.items():
            if modality in object_vectors:
                for obj_vec in object_vectors[modality]:
                    sim = 1 - cosine(q_vec, obj_vec)
                    max_sim = max(max_sim, sim)
        
        return max_sim if max_sim != -np.inf else 0.0
    
    def typed_set_similarity(self, query_vectors: Dict[str, np.ndarray],
                            object_vectors: Dict[str, List[np.ndarray]]) -> float:
        """
        Typed-set similarity: considers both primary and auxiliary modalities
        Uses soft-Hausdorff or similar aggregation
        """
        if not query_vectors or not object_vectors:
            return 0.0
        
        # Primary modality similarity (max)
        primary_sims = []
        for modality, q_vec in query_vectors.items():
            if modality in object_vectors:
                for obj_vec in object_vectors[modality]:
                    sim = 1 - cosine(q_vec, obj_vec)
                    primary_sims.append(sim)
        
        primary_score = max(primary_sims) if primary_sims else 0.0
        
        # Auxiliary modality contribution
        auxiliary_sims = []
        for aux_modality, aux_vectors in object_vectors.items():
            if aux_modality not in query_vectors:  # Auxiliary modality
                for q_modality, q_vec in query_vectors.items():
                    for aux_vec in aux_vectors:
                        sim = 1 - cosine(q_vec, aux_vec)
                        auxiliary_sims.append(sim)
        
        auxiliary_score = np.mean(auxiliary_sims) if auxiliary_sims else 0.0
        
        # Combined score
        return self.alpha * primary_score + self.beta * auxiliary_score
    
    def soft_hausdorff(self, set_a: List[np.ndarray], set_b: List[np.ndarray], 
                      temperature: float = 1.0) -> float:
        """
        Soft Hausdorff distance (k-quasimetric approximation)
        """
        if not set_a or not set_b:
            return 0.0
        
        # Forward direction: max over a, min over b
        forward_dists = []
        for vec_a in set_a:
            min_dist = min([1 - cosine(vec_a, vec_b) for vec_b in set_b])
            forward_dists.append(min_dist)
        forward_score = max(forward_dists) if forward_dists else 0.0
        
        # Backward direction: max over b, min over a
        backward_dists = []
        for vec_b in set_b:
            min_dist = min([1 - cosine(vec_a, vec_b) for vec_a in set_a])
            backward_dists.append(min_dist)
        backward_score = max(backward_dists) if backward_dists else 0.0
        
        # Symmetric soft-Hausdorff
        return (forward_score + backward_score) / 2.0


class MUSTSimilarity:
    """MUST baseline: fused vector similarity"""
    
    @staticmethod
    def fuse_vectors(vectors: Dict[str, List[np.ndarray]], 
                    weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Fuse multiple vectors into a single vector (weighted average)
        """
        if not vectors:
            return np.zeros(128)  # Default dimension
        
        if weights is None:
            weights = {mod: 1.0 / len(vectors) for mod in vectors.keys()}
        
        fused = None
        total_weight = 0.0
        
        for modality, vec_list in vectors.items():
            weight = weights.get(modality, 1.0)
            for vec in vec_list:
                if fused is None:
                    fused = np.zeros_like(vec)
                fused += weight * vec
                total_weight += weight
        
        if total_weight > 0:
            fused /= total_weight
        else:
            fused = np.zeros(128)
        
        return fused
    
    @staticmethod
    def compute_similarity(query_fused: np.ndarray, 
                          object_fused: np.ndarray) -> float:
        """Cosine similarity between fused vectors"""
        return 1 - cosine(query_fused, object_fused)


def compute_correlation(scores_a: List[float], scores_b: List[float],
                       ground_truth: List[float]) -> Dict[str, float]:
    """
    Compute correlation between two scoring methods and ground truth
    Returns Spearman and Kendall correlations
    """
    spearman_a, _ = spearmanr(scores_a, ground_truth)
    spearman_b, _ = spearmanr(scores_b, ground_truth)
    kendall_a, _ = kendalltau(scores_a, ground_truth)
    kendall_b, _ = kendalltau(scores_b, ground_truth)
    
    return {
        'spearman_a': spearman_a,
        'spearman_b': spearman_b,
        'kendall_a': kendall_a,
        'kendall_b': kendall_b,
        'spearman_diff': spearman_b - spearman_a,
        'kendall_diff': kendall_b - kendall_a
    }

