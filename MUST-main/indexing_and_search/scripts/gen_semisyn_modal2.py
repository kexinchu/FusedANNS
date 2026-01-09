#!/usr/bin/env python3
"""
Generate semi-synthetic modal2 (.ivecs) by k-means clustering on modal1 (.fvecs).

For each vector, we store the IDs of its nearest cluster centers as an int vector.
This simulates discrete text/attribute labels, following the approach described in [30].
"""

from __future__ import annotations

import argparse
import os
import struct
from typing import Iterable, Tuple

import numpy as np


def read_fvecs_memmap(path: str, max_rows: int | None = None) -> Tuple[np.ndarray, int, int]:
    with open(path, "rb") as f:
        dim_bytes = f.read(4)
        if not dim_bytes:
            raise ValueError(f"Empty fvecs file: {path}")
        dim = struct.unpack("I", dim_bytes)[0]
    file_size = os.path.getsize(path)
    vec_size = (dim + 1) * 4
    num = file_size // vec_size
    if max_rows is not None:
        num = min(num, max_rows)
    raw = np.memmap(path, dtype="int32", mode="r", shape=(num, dim + 1))
    data = raw[:, 1:].view("float32")
    return data, dim, num


def write_ivecs(path: str, rows: Iterable[np.ndarray]) -> None:
    with open(path, "wb") as f:
        for row in rows:
            row = np.asarray(row, dtype="uint32")
            f.write(struct.pack("I", row.size))
            f.write(row.tobytes())


def batch_iter(data: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    total = data.shape[0]
    for start in range(0, total, batch_size):
        yield data[start:start + batch_size]


def train_kmeans_faiss(train_data: np.ndarray, k: int, seed: int, niter: int) -> np.ndarray:
    import faiss  # type: ignore
    kmeans = faiss.Kmeans(train_data.shape[1], k, niter=niter, verbose=True, seed=seed)
    kmeans.train(train_data)
    return kmeans.centroids


def train_kmeans_sklearn(train_data: np.ndarray, k: int, seed: int, niter: int) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=1024,
        max_iter=niter,
        n_init=3,
        verbose=1,
    )
    kmeans.fit(train_data)
    return kmeans.cluster_centers_


def nearest_centroids_faiss(centroids: np.ndarray, data: np.ndarray, topk: int) -> np.ndarray:
    import faiss  # type: ignore
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids.astype("float32"))
    _, idx = index.search(data.astype("float32"), topk)
    return idx


def nearest_centroids_sklearn(centroids: np.ndarray, data: np.ndarray, topk: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=topk, metric="euclidean")
    nn.fit(centroids)
    _, idx = nn.kneighbors(data, return_distance=True)
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate semi-synthetic modal2 .ivecs for UQV.")
    parser.add_argument("--base-fvecs", required=True, help="Modal1 base fvecs path.")
    parser.add_argument("--query-fvecs", required=True, help="Modal1 query fvecs path.")
    parser.add_argument("--out-base", required=True, help="Output modal2 base ivecs path.")
    parser.add_argument("--out-query", required=True, help="Output modal2 query ivecs path.")
    parser.add_argument("--clusters", type=int, default=4096, help="Number of k-means clusters.")
    parser.add_argument("--attr-dim", type=int, default=9, help="Attributes per vector (top-k centers).")
    parser.add_argument("--train-max", type=int, default=200000, help="Max vectors to train k-means.")
    parser.add_argument("--batch-size", type=int, default=50000, help="Batch size for assignment.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--niter", type=int, default=30)
    parser.add_argument("--algo", choices=("faiss", "sklearn"), default="faiss")
    parser.add_argument("--id-offset", type=int, default=1, help="Offset for cluster IDs (1 avoids zero).")
    args = parser.parse_args()

    if args.attr_dim > args.clusters:
        raise ValueError("--attr-dim must be <= --clusters")

    base_data, dim, base_num = read_fvecs_memmap(args.base_fvecs)
    query_data, qdim, query_num = read_fvecs_memmap(args.query_fvecs)
    if dim != qdim:
        raise ValueError(f"dim mismatch: base {dim} vs query {qdim}")

    rng = np.random.default_rng(args.seed)
    train_count = min(args.train_max, base_num)
    train_idx = rng.choice(base_num, size=train_count, replace=False)
    train_data = np.asarray(base_data[train_idx], dtype="float32")

    if args.algo == "faiss":
        centroids = train_kmeans_faiss(train_data, args.clusters, args.seed, args.niter)
        assign_fn = nearest_centroids_faiss
    else:
        centroids = train_kmeans_sklearn(train_data, args.clusters, args.seed, args.niter)
        assign_fn = nearest_centroids_sklearn

    def assign_and_write(data: np.ndarray, out_path: str) -> None:
        def rows():
            for batch in batch_iter(data, args.batch_size):
                idx = assign_fn(centroids, batch, args.attr_dim)
                idx = idx.astype("uint32") + args.id_offset
                for row in idx:
                    yield row
        write_ivecs(out_path, rows())

    assign_and_write(base_data, args.out_base)
    assign_and_write(query_data, args.out_query)


if __name__ == "__main__":
    main()
