"""
Evaluation metrics for retrieval tasks
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def recall_at_k(ground_truth: List[int], retrieved: List[int], k: int) -> float:
    """Compute Recall@k"""
    if not ground_truth:
        return 0.0
    retrieved_k = set(retrieved[:k])
    relevant = set(ground_truth)
    return len(retrieved_k & relevant) / len(relevant)


def ndcg_at_k(ground_truth: List[int], retrieved: List[int], k: int) -> float:
    """Compute NDCG@k"""
    if not ground_truth:
        return 0.0
    
    dcg = 0.0
    for i, obj_id in enumerate(retrieved[:k]):
        if obj_id in ground_truth:
            # Binary relevance
            rel = 1.0
            dcg += rel / np.log2(i + 2)
    
    # Ideal DCG
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k))])
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(queries: List[Dict],
                      objects: List[Dict],
                      similarity_func,
                      ground_truth: List[List[int]],
                      k_values: List[int] = [1, 5, 10, 50, 100]) -> Dict:
    """
    Evaluate retrieval performance
    
    Args:
        queries: List of query objects
        objects: List of object objects
        similarity_func: Function (query, object) -> float
        ground_truth: List of lists, each containing relevant object indices
        k_values: List of k values for evaluation
    """
    results = defaultdict(dict)
    
    all_recalls = defaultdict(list)
    all_ndcgs = defaultdict(list)
    
    for q_idx, query in enumerate(queries):
        # Compute similarities
        scores = []
        for obj_idx, obj in enumerate(objects):
            score = similarity_func(query, obj)
            scores.append((obj_idx, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        retrieved = [idx for idx, _ in scores]
        
        # Evaluate at different k
        gt = ground_truth[q_idx]
        for k in k_values:
            r = recall_at_k(gt, retrieved, k)
            n = ndcg_at_k(gt, retrieved, k)
            all_recalls[k].append(r)
            all_ndcgs[k].append(n)
    
    # Aggregate results
    for k in k_values:
        results[f'recall@{k}'] = {
            'mean': np.mean(all_recalls[k]),
            'std': np.std(all_recalls[k]),
            'values': all_recalls[k]
        }
        results[f'ndcg@{k}'] = {
            'mean': np.mean(all_ndcgs[k]),
            'std': np.std(all_ndcgs[k]),
            'values': all_ndcgs[k]
        }
    
    return results


def compare_methods(results_a: Dict, results_b: Dict, 
                   metric: str = 'recall@10') -> Dict:
    """Compare two methods' results"""
    if metric not in results_a or metric not in results_b:
        return {}
    
    mean_a = results_a[metric]['mean']
    mean_b = results_b[metric]['mean']
    
    improvement = ((mean_b - mean_a) / mean_a * 100) if mean_a > 0 else 0.0
    
    return {
        'method_a_mean': mean_a,
        'method_b_mean': mean_b,
        'improvement_percent': improvement,
        'absolute_improvement': mean_b - mean_a
    }

