#!/usr/bin/env python3
"""
Prepare CelebA data with ResNet50 embeddings for indexing/search.

Outputs:
  - modal1 base/query fvecs (ResNet50 embeddings)
  - modal2 base/query ivecs (attribute labels)
  - groundtruth ivecs (by modal2 similarity by default)
  - delete_id ivecs (dim=0 placeholder)
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def read_attr_file(path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 3:
        raise ValueError("Attribute file is too short.")
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


def write_fvecs(path: Path, rows: Iterable[np.ndarray], dim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for row in rows:
            f.write(struct.pack("I", dim))
            f.write(row.astype(np.float32, copy=False).tobytes())


def write_ivecs(path: Path, rows: Iterable[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for row in rows:
            row = np.asarray(row, dtype=np.int32)
            f.write(struct.pack("I", row.size))
            f.write(row.tobytes())


def write_delete_id(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("I", 0))


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


def read_fvecs(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header = f.read(4)
        if not header:
            return np.empty((0, 0), dtype=np.float32)
        dim = struct.unpack("I", header)[0]
        f.seek(0, 2)
        size = f.tell()
        num = size // ((dim + 1) * 4)
        f.seek(0)
        data = np.empty((num, dim), dtype=np.float32)
        for i in range(num):
            _dim = struct.unpack("I", f.read(4))[0]
            if _dim != dim:
                raise ValueError("Inconsistent fvecs dim.")
            vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            data[i] = vec
    return data


def compute_gt_modal1(
    base_vecs: np.ndarray,
    query_vecs: np.ndarray,
    topk: int,
    block_size: int,
) -> List[np.ndarray]:
    results: List[np.ndarray] = []
    if base_vecs.size == 0 or query_vecs.size == 0:
        return results
    n_base = base_vecs.shape[0]
    for start in range(0, query_vecs.shape[0], block_size):
        q = query_vecs[start:start + block_size]
        scores = q @ base_vecs.T
        if topk >= n_base:
            idx = np.argsort(-scores, axis=1)
        else:
            idx = np.argpartition(-scores, topk, axis=1)[:, :topk]
            row_scores = np.take_along_axis(scores, idx, axis=1)
            order = np.argsort(-row_scores, axis=1)
            idx = np.take_along_axis(idx, order, axis=1)
        for row in idx:
            results.append(row.astype(np.int32))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CelebA ResNet50 embeddings.")
    parser.add_argument("--celeba-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--type", choices=("train", "test"), default="test")
    parser.add_argument("--dataset-suffix", default="resnet50_encode")
    parser.add_argument("--base-split", type=int, default=None)
    parser.add_argument("--query-split", type=int, default=None)
    parser.add_argument("--max-base", type=int, default=0)
    parser.add_argument("--max-query", type=int, default=0)
    parser.add_argument("--query-count", type=int, default=0)
    parser.add_argument("--gt-topk", type=int, default=10)
    parser.add_argument("--gt-mode", choices=("modal1", "modal2"), default="modal2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gt-block-size", type=int, default=256)
    parser.add_argument("--reuse-modal1", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_resnet50(device: str):
    try:
        import torch
        import torchvision
    except ImportError as exc:
        raise RuntimeError("torch/torchvision are required. Install with: pip install torch torchvision") from exc

    try:
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        model = torchvision.models.resnet50(weights=weights)
        preprocess = weights.transforms()
    except AttributeError:
        model = torchvision.models.resnet50(pretrained=True)
        preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model, preprocess, torch


def iter_image_batches(
    img_root: Path,
    names: List[str],
    preprocess,
    batch_size: int,
):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required. Install with: pip install pillow") from exc
    batch = []
    for name in names:
        img_path = img_root / name
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            batch.append(preprocess(im))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def embed_and_write(
    img_root: Path,
    names: List[str],
    out_path: Path,
    model,
    preprocess,
    torch,
    device: str,
    batch_size: int,
) -> int:
    dim = None
    rows = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        for batch in iter_image_batches(img_root, names, preprocess, batch_size):
            batch_tensor = torch.stack(batch).to(device)
            with torch.no_grad():
                feats = model(batch_tensor).detach().cpu().numpy().astype(np.float32)
            if dim is None:
                dim = feats.shape[1]
            for vec in feats:
                f.write(struct.pack("I", dim))
                f.write(vec.tobytes())
                rows.append(vec)
    return dim


def main() -> None:
    args = parse_args()

    attr_path = args.celeba_root / "list_attr_celeba.txt"
    part_path = args.celeba_root / "list_eval_partition.txt"
    img_root = args.celeba_root / "img_align_celeba"
    if not attr_path.exists() or not part_path.exists() or not img_root.exists():
        raise FileNotFoundError("CelebA files not found under --celeba-root.")

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

    if args.device:
        device = args.device
    else:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    model, preprocess, torch = load_resnet50(device)

    out_type_dir = args.out_root / args.type
    modal1_dir = out_type_dir / args.dataset_suffix
    modal1_base_path = modal1_dir / "celeba_modal1_base.fvecs"
    modal1_query_path = modal1_dir / "celeba_modal1_query.fvecs"
    modal2_base_path = out_type_dir / "celeba_modal2_base.ivecs"
    modal2_query_path = out_type_dir / "celeba_modal2_query.ivecs"
    gt_path = out_type_dir / "celeba_gt.ivecs"
    delete_id_path = out_type_dir / "celeba_delete_id.ivecs"

    if args.reuse_modal1 and modal1_base_path.exists() and modal1_query_path.exists():
        pass
    else:
        embed_and_write(img_root, base_names, modal1_base_path, model, preprocess, torch, device, args.batch_size)
        embed_and_write(img_root, query_names, modal1_query_path, model, preprocess, torch, device, args.batch_size)

    write_ivecs(modal2_base_path, base_attrs)
    write_ivecs(modal2_query_path, query_attrs)

    if args.gt_mode == "modal1":
        base_vecs = read_fvecs(modal1_base_path)
        query_vecs = read_fvecs(modal1_query_path)
        gt = compute_gt_modal1(base_vecs, query_vecs, args.gt_topk, args.gt_block_size)
    else:
        gt = compute_gt_modal2(base_attrs, query_attrs, args.gt_topk)
    write_ivecs(gt_path, gt)
    write_delete_id(delete_id_path)

    print("[OK] CelebA ResNet50 prepared:")
    print("  modal1 base:", modal1_base_path)
    print("  modal1 query:", modal1_query_path)
    print("  modal2 base:", modal2_base_path)
    print("  modal2 query:", modal2_query_path)
    print("  groundtruth:", gt_path)
    print("  delete_id:", delete_id_path)


if __name__ == "__main__":
    main()
