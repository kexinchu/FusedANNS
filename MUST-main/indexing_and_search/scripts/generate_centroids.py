#!/usr/bin/env python3
"""
Generate centroid candidates for CelebA base vectors.

Reads:
  - celeba_modal1_base.fvecs (visual)
  - celeba_modal2_base.ivecs (attribute)

Outputs:
  - celeba_centroids_visual.fvecs
  - celeba_centroids_attr.ivecs
  - centroid_ids.ivecs (nearest base id per centroid)
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Tuple

import numpy as np


def _read_vecs_fixed(path: Path) -> Tuple[np.ndarray, int]:
    data = np.fromfile(path, dtype=np.int32)
    if data.size == 0:
        return np.empty((0, 0), dtype=np.int32), 0
    dim = int(data[0])
    stride = dim + 1
    if dim <= 0 or data.size % stride != 0:
        return np.empty((0, 0), dtype=np.int32), -1
    return data.reshape(-1, stride)[:, 1:].copy(), dim


def read_fvecs(path: Path) -> np.ndarray:
    raw, dim = _read_vecs_fixed(path)
    if dim < 0:
        return read_fvecs_var(path)
    return raw.view(np.float32)


def read_ivecs(path: Path) -> np.ndarray:
    raw, dim = _read_vecs_fixed(path)
    if dim < 0:
        return read_ivecs_var(path)
    return raw.astype(np.int32, copy=False)


def read_fvecs_var(path: Path) -> np.ndarray:
    rows = []
    with path.open("rb") as f:
        while True:
            head = f.read(4)
            if not head:
                break
            dim = struct.unpack("I", head)[0]
            vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            rows.append(vec)
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(rows)


def read_ivecs_var(path: Path) -> np.ndarray:
    rows = []
    with path.open("rb") as f:
        while True:
            head = f.read(4)
            if not head:
                break
            dim = struct.unpack("I", head)[0]
            vec = np.frombuffer(f.read(dim * 4), dtype=np.int32)
            rows.append(vec)
    if not rows:
        return np.empty((0, 0), dtype=np.int32)
    return np.vstack(rows)


def write_fvecs(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dim = data.shape[1] if data.size else 0
    with path.open("wb") as f:
        for row in data.astype(np.float32, copy=False):
            f.write(struct.pack("I", dim))
            f.write(row.tobytes())


def write_ivecs(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    with path.open("wb") as f:
        for row in data:
            row = np.asarray(row, dtype=np.int32)
            f.write(struct.pack("I", row.size))
            f.write(row.tobytes())


def run_kmeans(
    data: np.ndarray,
    k: int,
    seed: int,
    max_iter: int,
    n_init: int,
    method: str,
    batch_size: int,
) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans, MiniBatchKMeans
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for KMeans. Install with: pip install scikit-learn") from exc
    if method == "minibatch":
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            max_iter=max_iter,
            n_init=n_init,
            batch_size=batch_size,
        )
    else:
        model = KMeans(
            n_clusters=k,
            random_state=seed,
            max_iter=max_iter,
            n_init=n_init,
        )
    model.fit(data)
    return model.cluster_centers_.astype(np.float32, copy=False)


def nearest_ids(
    base: np.ndarray,
    centroids: np.ndarray,
    block_size: int,
) -> np.ndarray:
    if base.size == 0 or centroids.size == 0:
        return np.empty((0,), dtype=np.int32)
    base = base.astype(np.float32, copy=False)
    centroids = centroids.astype(np.float32, copy=False)
    centroids_norm = (centroids ** 2).sum(axis=1)
    best_dist = np.full(centroids.shape[0], np.inf, dtype=np.float32)
    best_idx = np.full(centroids.shape[0], -1, dtype=np.int64)
    for start in range(0, base.shape[0], block_size):
        block = base[start:start + block_size]
        block_norm = (block ** 2).sum(axis=1)
        dists = block_norm[:, None] + centroids_norm[None, :] - 2.0 * block @ centroids.T
        block_min_idx = np.argmin(dists, axis=0)
        block_min_dist = dists[block_min_idx, np.arange(centroids.shape[0])]
        update = block_min_dist < best_dist
        best_dist[update] = block_min_dist[update]
        best_idx[update] = start + block_min_idx[update]
    return best_idx.astype(np.int32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate K-Means centroids for CelebA base vectors.")
    parser.add_argument("--modal1-base", type=Path, default=Path("celeba_modal1_base.fvecs"))
    parser.add_argument("--modal2-base", type=Path, default=Path("celeba_modal2_base.ivecs"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("-k", "--clusters", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--method", choices=("kmeans", "minibatch"), default="kmeans")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--centroid-ids-source", choices=("visual", "attr", "none"), default="visual")
    parser.add_argument("--id-block-size", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    visual_base = read_fvecs(args.modal1_base)
    attr_base = read_ivecs(args.modal2_base)
    if visual_base.size == 0 or attr_base.size == 0:
        raise RuntimeError("Input base vectors are empty.")
    if visual_base.shape[0] != attr_base.shape[0]:
        print(
            f"Warning: base counts differ (visual={visual_base.shape[0]}, attr={attr_base.shape[0]}).",
        )

    print(f"Clustering visual base vectors: n={visual_base.shape[0]} d={visual_base.shape[1]}")
    visual_centroids = run_kmeans(
        visual_base,
        args.clusters,
        args.seed,
        args.max_iter,
        args.n_init,
        args.method,
        args.batch_size,
    )

    print(f"Clustering attribute base vectors: n={attr_base.shape[0]} d={attr_base.shape[1]}")
    attr_centroids = run_kmeans(
        attr_base.astype(np.float32, copy=False),
        args.clusters,
        args.seed,
        args.max_iter,
        args.n_init,
        args.method,
        args.batch_size,
    )
    attr_centroids_int = np.rint(attr_centroids).astype(np.int32, copy=False)

    out_visual = args.out_dir / "celeba_centroids_visual.fvecs"
    out_attr = args.out_dir / "celeba_centroids_attr.ivecs"
    out_ids = args.out_dir / "centroid_ids.ivecs"

    write_fvecs(out_visual, visual_centroids)
    write_ivecs(out_attr, attr_centroids_int)
    print(f"Saved visual centroids -> {out_visual}")
    print(f"Saved attribute centroids -> {out_attr}")

    if args.centroid_ids_source != "none":
        if args.centroid_ids_source == "visual":
            base_for_ids = visual_base
            centroids_for_ids = visual_centroids
        else:
            base_for_ids = attr_base.astype(np.float32, copy=False)
            centroids_for_ids = attr_centroids
        ids = nearest_ids(base_for_ids, centroids_for_ids, args.id_block_size)
        write_ivecs(out_ids, ids)
        print(f"Saved centroid ids ({args.centroid_ids_source}) -> {out_ids}")


if __name__ == "__main__":
    main()
