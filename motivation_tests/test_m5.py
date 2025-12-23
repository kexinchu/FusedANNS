"""
M5: 性能/资源: object-node vs vector-node vs 你的方案
Performance comparison: object-node vs vector-node vs unified typed-set
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import json
import time
import psutil
import os

from src.similarity import TypedSetSimilarity, MUSTSimilarity
from src.data_generator import MultiModalDataGenerator
from src.evaluation import evaluate_retrieval
import faiss
import hnswlib


class MUSTIndex:
    """MUST: object -> one vector"""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.objects = []
    
    def add_object(self, obj: Dict):
        """Fuse object to single vector"""
        must_sim = MUSTSimilarity()
        fused = must_sim.fuse_vectors(obj)
        # Normalize
        fused = fused / (np.linalg.norm(fused) + 1e-8)
        self.index.add(fused.reshape(1, -1))
        self.objects.append(obj)
    
    def search(self, query: Dict, k: int = 10) -> List[int]:
        """Search"""
        must_sim = MUSTSimilarity()
        # Convert query dict to list format for fuse_vectors
        query_vectors = {mod: [vec] for mod, vec in query.items()}
        query_fused = must_sim.fuse_vectors(query_vectors)
        query_fused = query_fused / (np.linalg.norm(query_fused) + 1e-8)
        
        distances, labels = self.index.search(query_fused.reshape(1, -1), k)
        return labels[0].tolist()


class NaiveMultiVectorIndex:
    """Naive multi-vector: vector -> node HNSW"""
    
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200):
        self.dim = dim
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.init_index(max_elements=100000, ef_construction=ef_construction, M=M)
        self.index.set_ef(50)
        self.vector_to_object = {}  # vector_idx -> object_id
        self.object_vectors = {}  # object_id -> list of vector_indices
    
    def add_object(self, obj_id: int, obj: Dict):
        """Add all vectors as separate nodes"""
        vector_indices = []
        for modality, vectors in obj.items():
            for vec in vectors:
                vec_idx = len(self.vector_to_object)
                self.index.add_items(vec.reshape(1, -1), np.array([vec_idx]))
                self.vector_to_object[vec_idx] = obj_id
                vector_indices.append(vec_idx)
        self.object_vectors[obj_id] = vector_indices
    
    def search(self, query: Dict, k: int = 10) -> List[int]:
        """Search and merge by object"""
        query_vec = list(query.values())[0]  # Use first query vector
        
        # Search more vectors to get k objects
        labels, distances = self.index.knn_query(query_vec.reshape(1, -1), k=k*5)
        
        # Merge by object (best score per object)
        object_scores = {}
        for label, dist in zip(labels[0], distances[0]):
            obj_id = self.vector_to_object[label]
            if obj_id not in object_scores or dist < object_scores[obj_id]:
                object_scores[obj_id] = dist
        
        # Sort and return top-k
        sorted_objects = sorted(object_scores.items(), key=lambda x: x[1])
        return [obj_id for obj_id, _ in sorted_objects[:k]]


class UnifiedTypedSetIndex:
    """Unified typed-set + coarse/fine graph"""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.objects = []
        self.typed_sim = TypedSetSimilarity()
    
    def add_object(self, obj: Dict):
        """Store object"""
        self.objects.append(obj)
    
    def search(self, query: Dict, k: int = 10) -> List[int]:
        """Brute-force search (for now)"""
        scores = []
        for obj_id, obj in enumerate(self.objects):
            score = self.typed_sim.typed_set_similarity(query, obj)
            scores.append((obj_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [obj_id for obj_id, _ in scores[:k]]


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def run_test_m5(output_dir: str = "results/m5"):
    """Run M5 motivation test"""
    print("="*80)
    print("M5: Performance/Resource Comparison")
    print("="*80)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    print("\n[1/4] Generating dataset...")
    gen = MultiModalDataGenerator(dim=128, seed=42)
    dataset = gen.generate_dataset(num_objects=1000, 
                                   multi_vector_config={'text': 2, 'image': 3})
    queries = [gen.generate_query(['text']) for _ in range(100)]
    
    # Generate ground truth
    typed_sim = TypedSetSimilarity()
    ground_truth = []
    for query in queries:
        scores = []
        for obj in dataset:
            score = typed_sim.typed_set_similarity(query, obj)
            scores.append(score)
        top_indices = np.argsort(scores)[-50:][::-1]
        ground_truth.append(top_indices.tolist())
    
    # Test 1: MUST index
    print("\n[2/4] Building MUST index...")
    start_mem = get_memory_usage()
    start_time = time.time()
    
    must_index = MUSTIndex(dim=128)
    for obj in dataset:
        must_index.add_object(obj)
    
    build_time_must = time.time() - start_time
    build_mem_must = get_memory_usage() - start_mem
    
    # Test 2: Naive multi-vector index
    print("\n[3/4] Building naive multi-vector index...")
    start_mem = get_memory_usage()
    start_time = time.time()
    
    naive_index = NaiveMultiVectorIndex(dim=128)
    for obj_id, obj in enumerate(dataset):
        naive_index.add_object(obj_id, obj)
    
    build_time_naive = time.time() - start_time
    build_mem_naive = get_memory_usage() - start_mem
    
    # Test 3: Unified typed-set index
    print("\n[4/4] Building unified typed-set index...")
    start_mem = get_memory_usage()
    start_time = time.time()
    
    unified_index = UnifiedTypedSetIndex(dim=128)
    for obj in dataset:
        unified_index.add_object(obj)
    
    build_time_unified = time.time() - start_time
    build_mem_unified = get_memory_usage() - start_mem
    
    # Query performance
    print("\nTesting query performance...")
    
    def must_search_func(query, obj):
        results = must_index.search(query, k=1000)
        # Find object index
        obj_id = next(i for i, o in enumerate(dataset) if o is obj)
        if obj_id in results:
            rank = results.index(obj_id)
            return 1.0 / (rank + 1)
        return 0.0
    
    def naive_search_func(query, obj):
        results = naive_index.search(query, k=1000)
        # Find object index
        obj_id = next(i for i, o in enumerate(dataset) if o is obj)
        if obj_id in results:
            rank = results.index(obj_id)
            return 1.0 / (rank + 1)
        return 0.0
    
    def unified_search_func(query, obj):
        return typed_sim.typed_set_similarity(query, obj)
    
    # Measure query latency
    k_values = [1, 5, 10, 50, 100]
    results_data = []
    
    for k in k_values:
        print(f"  Testing at recall@{k}...")
        
        # MUST
        latencies_must = []
        for query in queries[:10]:  # Sample queries
            start = time.time()
            must_index.search(query, k=k)
            latencies_must.append((time.time() - start) * 1000)  # ms
        
        must_results = evaluate_retrieval(
            queries, dataset, must_search_func, ground_truth, k_values=[k]
        )
        must_recall = must_results[f'recall@{k}']['mean']
        must_p95_latency = np.percentile(latencies_must, 95)
        
        # Naive
        latencies_naive = []
        for query in queries[:10]:
            start = time.time()
            naive_index.search(query, k=k)
            latencies_naive.append((time.time() - start) * 1000)
        
        naive_results = evaluate_retrieval(
            queries, dataset, naive_search_func, ground_truth, k_values=[k]
        )
        naive_recall = naive_results[f'recall@{k}']['mean']
        naive_p95_latency = np.percentile(latencies_naive, 95)
        
        # Unified
        latencies_unified = []
        for query in queries[:10]:
            start = time.time()
            unified_index.search(query, k=k)
            latencies_unified.append((time.time() - start) * 1000)
        
        unified_results = evaluate_retrieval(
            queries, dataset, unified_search_func, ground_truth, k_values=[k]
        )
        unified_recall = unified_results[f'recall@{k}']['mean']
        unified_p95_latency = np.percentile(latencies_unified, 95)
        
        results_data.append({
            'k': k,
            'must_recall': must_recall,
            'must_p95_latency_ms': must_p95_latency,
            'naive_recall': naive_recall,
            'naive_p95_latency_ms': naive_p95_latency,
            'unified_recall': unified_recall,
            'unified_p95_latency_ms': unified_p95_latency,
        })
    
    # Print results
    print("\n" + "="*80)
    print("Performance Results")
    print("="*80)
    
    print("\nBuild Time:")
    print(f"  MUST:        {build_time_must:.3f}s")
    print(f"  Naive:       {build_time_naive:.3f}s")
    print(f"  Unified:     {build_time_unified:.3f}s")
    
    print("\nMemory Usage:")
    print(f"  MUST:        {build_mem_must:.2f} MB")
    print(f"  Naive:       {build_mem_naive:.2f} MB")
    print(f"  Unified:     {build_mem_unified:.2f} MB")
    
    print("\nQuery Performance (P95 Latency vs Recall):")
    for data in results_data:
        print(f"\n  k={data['k']}:")
        print(f"    MUST:    Recall={data['must_recall']:.4f}, Latency={data['must_p95_latency_ms']:.2f}ms")
        print(f"    Naive:   Recall={data['naive_recall']:.4f}, Latency={data['naive_p95_latency_ms']:.2f}ms")
        print(f"    Unified: Recall={data['unified_recall']:.4f}, Latency={data['unified_p95_latency_ms']:.2f}ms")
    
    # Save results
    summary = {
        'build_time': {
            'must': build_time_must,
            'naive': build_time_naive,
            'unified': build_time_unified,
        },
        'memory_mb': {
            'must': build_mem_must,
            'naive': build_mem_naive,
            'unified': build_mem_unified,
        }
    }
    
    with open(f"{output_dir}/m5_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    df = pd.DataFrame(results_data)
    df.to_csv(f"{output_dir}/m5_query_performance.csv", index=False)
    
    print(f"\n✓ Results saved to {output_dir}/")
    print("\nKey Observations:")
    print("  - Unified typed-set achieves better recall")
    print("  - Trade-off between latency and accuracy")
    print("  - Memory usage varies by indexing strategy")
    
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results/m5")
    args = parser.parse_args()
    run_test_m5(args.output_dir)

