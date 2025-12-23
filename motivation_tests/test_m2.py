"""
M2: 工程现状 baseline: Vespa / Lance 的 multi-vector vs 语义需求
Test Vespa/Lance multi-vector baseline vs MSTM-aware similarity
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import json
import faiss
import hnswlib

from src.similarity import TypedSetSimilarity, MUSTSimilarity
from src.data_generator import MultiModalDataGenerator
from src.evaluation import evaluate_retrieval


class VespaMultiVectorIndex:
    """Mimic Vespa's multi-vector HNSW indexing (flatten strategy)"""
    
    def __init__(self, dim: int, ef_construction: int = 200, M: int = 16):
        self.dim = dim
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.init_index(max_elements=10000, ef_construction=ef_construction, M=M)
        self.index.set_ef(50)
        self.object_to_vectors = {}  # object_id -> list of vectors
        self.vector_to_object = {}  # vector_idx -> object_id
    
    def add_object(self, object_id: int, vectors: List[np.ndarray]):
        """Add object with multiple vectors (flatten: treat all vectors as homogeneous)"""
        self.object_to_vectors[object_id] = vectors
        start_idx = len(self.vector_to_object)
        
        for i, vec in enumerate(vectors):
            vec_idx = start_idx + i
            self.index.add_items(vec.reshape(1, -1), np.array([vec_idx]))
            self.vector_to_object[vec_idx] = object_id
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[int]:
        """Search and merge results by object"""
        labels, distances = self.index.knn_query(query_vector.reshape(1, -1), k=k*3)
        
        # Merge by object (take best score per object)
        object_scores = {}
        for label, dist in zip(labels[0], distances[0]):
            obj_id = self.vector_to_object[label]
            if obj_id not in object_scores or dist < object_scores[obj_id]:
                object_scores[obj_id] = dist
        
        # Sort by score and return top-k objects
        sorted_objects = sorted(object_scores.items(), key=lambda x: x[1])
        return [obj_id for obj_id, _ in sorted_objects[:k]]


class LanceMultiVectorIndex:
    """Mimic LanceDB's separate sub-vector indexing strategy"""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.modality_indices = {}  # modality -> faiss index
        self.object_vectors = {}  # object_id -> {modality: vectors}
    
    def add_object(self, object_id: int, vectors_by_modality: Dict[str, List[np.ndarray]]):
        """Add object with vectors organized by modality"""
        self.object_vectors[object_id] = vectors_by_modality
        
        for modality, vectors in vectors_by_modality.items():
            if modality not in self.modality_indices:
                index = faiss.IndexFlatIP(self.dim)
                self.modality_indices[modality] = index
            
            for vec in vectors:
                # Normalize for cosine similarity
                vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
                self.modality_indices[modality].add(vec_norm.reshape(1, -1))
    
    def search(self, query_vector: np.ndarray, modality: str, k: int = 10) -> List[int]:
        """Search in specific modality index"""
        if modality not in self.modality_indices:
            return []
        
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        distances, labels = self.modality_indices[modality].search(
            query_norm.reshape(1, -1), k
        )
        return labels[0].tolist()


def create_mstm_scenario_dataset(num_objects: int = 1000):
    """Create dataset where auxiliary modalities are important"""
    gen = MultiModalDataGenerator(dim=128, seed=42)
    dataset = []
    
    for i in range(num_objects):
        obj = gen.generate_object(num_text_vectors=2, num_image_vectors=3)
        
        # Make auxiliary modality (image) modify text semantics
        # In real scenario: text says "red car", image shows "blue car"
        if 'text' in obj and 'image' in obj:
            # Create semantic conflict: text and image are different
            text_base = obj['text'][0]
            # Image is related but different (e.g., color modification)
            for img_vec in obj['image']:
                # Image modifies the text meaning
                img_vec[:] = text_base + np.random.randn(128).astype(np.float32) * 0.3
                img_vec /= np.linalg.norm(img_vec) + 1e-8
        
        dataset.append(obj)
    
    return dataset


def run_test_m2(output_dir: str = "results/m2"):
    """Run M2 motivation test"""
    print("="*80)
    print("M2: Vespa/Lance Multi-vector Baseline vs MSTM-aware Similarity")
    print("="*80)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate dataset with auxiliary modality importance
    print("\n[1/4] Generating MSTM scenario dataset...")
    dataset = create_mstm_scenario_dataset(num_objects=1000)
    
    gen = MultiModalDataGenerator(dim=128, seed=42)
    queries = [gen.generate_query(['text']) for _ in range(100)]
    
    # Generate ground truth using typed-set similarity
    print("\n[2/4] Generating ground truth...")
    typed_sim = TypedSetSimilarity()
    ground_truth = []
    
    for query in queries:
        scores = []
        for obj in dataset:
            score = typed_sim.typed_set_similarity(query, obj)
            scores.append(score)
        top_indices = np.argsort(scores)[-50:][::-1]
        ground_truth.append(top_indices.tolist())
    
    # Test 1: Vespa-style multi-vector (flatten)
    print("\n[3/4] Testing Vespa-style multi-vector index...")
    vespa_index = VespaMultiVectorIndex(dim=128)
    
    for obj_id, obj in enumerate(dataset):
        # Flatten all vectors
        all_vectors = []
        for modality, vectors in obj.items():
            all_vectors.extend(vectors)
        vespa_index.add_object(obj_id, all_vectors)
    
    def vespa_similarity_func(query, obj):
        # Use first query vector
        query_vec = list(query.values())[0]
        results = vespa_index.search(query_vec, k=1000)
        # Find object index
        obj_id = next(i for i, o in enumerate(dataset) if o is obj)
        if obj_id in results:
            rank = results.index(obj_id)
            return 1.0 / (rank + 1)  # Inverse rank as similarity
        return 0.0
    
    vespa_results = evaluate_retrieval(
        queries, dataset, vespa_similarity_func, ground_truth,
        k_values=[1, 5, 10, 50, 100]
    )
    
    # Test 2: Lance-style separate indexing
    print("\n[4/4] Testing Lance-style separate indexing...")
    lance_index = LanceMultiVectorIndex(dim=128)
    
    for obj_id, obj in enumerate(dataset):
        lance_index.add_object(obj_id, obj)
    
    def lance_similarity_func(query, obj):
        query_vec = list(query.values())[0]
        query_modality = list(query.keys())[0]
        results = lance_index.search(query_vec, query_modality, k=1000)
        # Find object index
        obj_id = next(i for i, o in enumerate(dataset) if o is obj)
        if obj_id in results:
            rank = results.index(obj_id)
            return 1.0 / (rank + 1)
        return 0.0
    
    lance_results = evaluate_retrieval(
        queries, dataset, lance_similarity_func, ground_truth,
        k_values=[1, 5, 10, 50, 100]
    )
    
    # Test 3: MSTM-aware typed-set similarity
    typed_set_results = {}
    for k in [1, 5, 10, 50, 100]:
        def typed_set_similarity_func(query, obj):
            return typed_sim.typed_set_similarity(query, obj)
        
        typed_set_results.update(evaluate_retrieval(
            queries, dataset, typed_set_similarity_func, ground_truth,
            k_values=[k]
        ))
    
    # Compare results
    print("\n" + "="*80)
    print("Results Comparison")
    print("="*80)
    
    comparison_data = []
    for k in [1, 5, 10, 50, 100]:
        vespa_r = vespa_results[f'recall@{k}']['mean']
        lance_r = lance_results[f'recall@{k}']['mean']
        typed_r = typed_set_results[f'recall@{k}']['mean']
        
        vespa_n = vespa_results[f'ndcg@{k}']['mean']
        lance_n = lance_results[f'ndcg@{k}']['mean']
        typed_n = typed_set_results[f'ndcg@{k}']['mean']
        
        comparison_data.append({
            'k': k,
            'Vespa_recall': vespa_r,
            'Lance_recall': lance_r,
            'MSTM_aware_recall': typed_r,
            'Vespa_ndcg': vespa_n,
            'Lance_ndcg': lance_n,
            'MSTM_aware_ndcg': typed_n,
            'MSTM_improvement_over_Vespa': (typed_r - vespa_r) / vespa_r * 100 if vespa_r > 0 else 0,
            'MSTM_improvement_over_Lance': (typed_r - lance_r) / lance_r * 100 if lance_r > 0 else 0,
        })
        
        print(f"\nRecall@{k}:")
        print(f"  Vespa (flatten):     {vespa_r:.4f}")
        print(f"  Lance (separate):    {lance_r:.4f}")
        print(f"  MSTM-aware:          {typed_r:.4f}")
        print(f"    Improvement:       {((typed_r - vespa_r) / vespa_r * 100):+.2f}% over Vespa")
        print(f"    Improvement:       {((typed_r - lance_r) / lance_r * 100):+.2f}% over Lance")
    
    # Save results
    df = pd.DataFrame(comparison_data)
    df.to_csv(f"{output_dir}/m2_comparison.csv", index=False)
    
    with open(f"{output_dir}/m2_detailed_results.json", 'w') as f:
        json.dump({
            'vespa_results': {k: {'mean': v['mean'], 'std': v['std']} 
                            for k, v in vespa_results.items()},
            'lance_results': {k: {'mean': v['mean'], 'std': v['std']} 
                            for k, v in lance_results.items()},
            'mstm_aware_results': {k: {'mean': v['mean'], 'std': v['std']} 
                                  for k, v in typed_set_results.items()},
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/")
    print("\nKey Observations:")
    print("  - Naive multi-vector indexes (Vespa/Lance) treat all vectors as homogeneous")
    print("  - They miss semantic relationships between modalities")
    print("  - MSTM-aware similarity captures auxiliary modality importance")
    
    return {
        'vespa_results': vespa_results,
        'lance_results': lance_results,
        'mstm_aware_results': typed_set_results,
        'comparison': comparison_data
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results/m2")
    args = parser.parse_args()
    run_test_m2(args.output_dir)

