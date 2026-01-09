#!/usr/bin/env python3
"""
Prepare CelebA data for indexing/search with minimal changes.

Outputs:
  - modal1 base/query fvecs
  - modal2 base/query ivecs (attribute labels)
  - groundtruth ivecs (top-k by attribute similarity or modal1 similarity)
  - delete_id ivecs (dim=0 placeholder)
"""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def read_attr_file(path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 3:
        raise ValueError("Attribute file is too short.")
    _total = int(lines[0])
    attr_names = lines[1].split()
    names = []
    attrs = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        vals = [int(x) for x in parts[1:]]
        if len(vals) != len(attr_names):
            continue
        # Map -1/1 to 0/1 for attributes.
        vals = [1 if v > 0 else 0 for v in vals]
        names.append(name)
        attrs.append(vals)
    return attr_names, names, np.asarray(attrs, dtype=np.int32)


def read_partitions(path: Path) -> dict[str, int]:
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, part = line.split()
            mapping[name] = int(part)
    return mapping


def write_fvecs(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dim = data.shape[1] if data.size else 0
    with path.open("wb") as f:
        for row in data.astype(np.float32, copy=False):
            f.write(struct.pack("I", dim))
            f.write(row.tobytes())


def write_ivecs(path: Path, rows: Iterable[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for row in rows:
            row = np.asarray(row, dtype=np.int32)
            f.write(struct.pack("I", row.size))
            f.write(row.tobytes())


def select_split(
    names: List[str],
    attrs: np.ndarray,
    partitions: dict[str, int],
    split_id: int,
) -> Tuple[List[str], np.ndarray]:
    keep_names = []
    keep_attrs = []
    for name, attr in zip(names, attrs):
        if partitions.get(name) == split_id:
            keep_names.append(name)
            keep_attrs.append(attr)
    return keep_names, np.asarray(keep_attrs, dtype=np.int32)


def sample_subset(
    names: List[str],
    attrs: np.ndarray,
    max_count: int,
    seed: int,
) -> Tuple[List[str], np.ndarray]:
    if max_count <= 0 or len(names) <= max_count:
        return names, attrs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(names), size=max_count, replace=False)
    idx = np.sort(idx)
    return [names[i] for i in idx], attrs[idx]


def build_modal1_from_attrs(attrs: np.ndarray) -> np.ndarray:
    return attrs.astype(np.float32, copy=False)


def build_modal1_from_images(
    img_root: Path,
    names: List[str],
    image_size: int,
    gray: bool,
) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image features. Install with: pip install pillow") from exc
    mode = "L" if gray else "RGB"
    channels = 1 if gray else 3
    dim = image_size * image_size * channels
    data = np.zeros((len(names), dim), dtype=np.float32)
    for i, name in enumerate(names):
        img_path = img_root / name
        with Image.open(img_path) as im:
            im = im.convert(mode)
            im = im.resize((image_size, image_size))
            arr = np.asarray(im, dtype=np.float32)
            if gray:
                arr = arr.reshape(-1)
            else:
                arr = arr.reshape(-1, channels).reshape(-1)
            data[i] = arr / 255.0
    return data


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.size:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, k)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx


def compute_gt_modal2(
    base_attrs: np.ndarray,
    query_attrs: np.ndarray,
    topk: int,
) -> List[np.ndarray]:
    results = []
    for q in query_attrs:
        scores = (base_attrs == q).sum(axis=1).astype(np.int32)
        results.append(topk_indices(scores, topk).astype(np.int32))
    return results


def compute_gt_modal1(
    base_vecs: np.ndarray,
    query_vecs: np.ndarray,
    topk: int,
) -> List[np.ndarray]:
    results = []
    for q in query_vecs:
        scores = base_vecs @ q
        results.append(topk_indices(scores, topk).astype(np.int32))
    return results


def write_delete_id(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("I", 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CelebA for indexing/search.")
    parser.add_argument("--celeba-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--type", choices=("train", "test"), default="test")
    parser.add_argument("--dataset-suffix", default="raw32")
    parser.add_argument("--modal1-source", choices=("attrs", "images"), default="attrs")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--gray", action="store_true")
    parser.add_argument("--base-split", type=int, default=None)
    parser.add_argument("--query-split", type=int, default=None)
    parser.add_argument("--max-base", type=int, default=0)
    parser.add_argument("--max-query", type=int, default=0)
    parser.add_argument("--query-count", type=int, default=0)
    parser.add_argument("--gt-topk", type=int, default=10)
    parser.add_argument("--gt-mode", choices=("modal1", "modal2"), default="modal2")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    attr_path = args.celeba_root / "list_attr_celeba.txt"
    part_path = args.celeba_root / "list_eval_partition.txt"
    img_root = args.celeba_root / "img_align_celeba"

    if not attr_path.exists() or not part_path.exists():
        raise FileNotFoundError("CelebA attribute/partition files not found.")

    _, names, attrs = read_attr_file(attr_path)
    partitions = read_partitions(part_path)

    base_split = args.base_split
    query_split = args.query_split
    if base_split is None:
        base_split = 0 if args.type == "train" else 2
    if query_split is None:
        query_split = base_split

    base_names, base_attrs = select_split(names, attrs, partitions, base_split)
    query_names, query_attrs = select_split(names, attrs, partitions, query_split)

    base_names, base_attrs = sample_subset(base_names, base_attrs, args.max_base, args.seed)
    query_names, query_attrs = sample_subset(query_names, query_attrs, args.max_query, args.seed + 1)

    if args.query_count > 0 and args.query_count < len(query_names):
        query_names, query_attrs = sample_subset(query_names, query_attrs, args.query_count, args.seed + 2)

    if args.modal1_source == "images":
        base_modal1 = build_modal1_from_images(img_root, base_names, args.image_size, args.gray)
        query_modal1 = build_modal1_from_images(img_root, query_names, args.image_size, args.gray)
    else:
        base_modal1 = build_modal1_from_attrs(base_attrs)
        query_modal1 = build_modal1_from_attrs(query_attrs)

    out_type_dir = args.out_root / args.type
    modal1_dir = out_type_dir / args.dataset_suffix
    modal1_base_path = modal1_dir / "celeba_modal1_base.fvecs"
    modal1_query_path = modal1_dir / "celeba_modal1_query.fvecs"
    modal2_base_path = out_type_dir / "celeba_modal2_base.ivecs"
    modal2_query_path = out_type_dir / "celeba_modal2_query.ivecs"
    gt_path = out_type_dir / "celeba_gt.ivecs"
    delete_id_path = out_type_dir / "celeba_delete_id.ivecs"

    write_fvecs(modal1_base_path, base_modal1)
    write_fvecs(modal1_query_path, query_modal1)
    write_ivecs(modal2_base_path, base_attrs)
    write_ivecs(modal2_query_path, query_attrs)

    if args.gt_mode == "modal1":
        gt = compute_gt_modal1(base_modal1, query_modal1, args.gt_topk)
    else:
        gt = compute_gt_modal2(base_attrs, query_attrs, args.gt_topk)
    write_ivecs(gt_path, gt)
    write_delete_id(delete_id_path)

    print("[OK] CelebA prepared:")
    print("  modal1 base:", modal1_base_path)
    print("  modal1 query:", modal1_query_path)
    print("  modal2 base:", modal2_base_path)
    print("  modal2 query:", modal2_query_path)
    print("  groundtruth:", gt_path)
    print("  delete_id:", delete_id_path)


if __name__ == "__main__":
    main()
