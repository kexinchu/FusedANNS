#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path

import numpy as np


def read_fvecs(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int32)
    if data.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    dim = int(data[0])
    data = data.reshape(-1, dim + 1)
    return data[:, 1:].view(np.float32).copy()


def read_ivecs(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int32)
    if data.size == 0:
        return np.empty((0, 0), dtype=np.int32)
    dim = int(data[0])
    return data.reshape(-1, dim + 1)[:, 1:].copy()


def write_fvecs(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dim = data.shape[1] if data.size else 0
    with open(path, "wb") as fp:
        for row in data.astype(np.float32, copy=False):
            fp.write(struct.pack("I", dim))
            fp.write(row.tobytes())


def write_ivecs(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dim = data.shape[1] if data.size else 0
    with open(path, "wb") as fp:
        for row in data.astype(np.int32, copy=False):
            fp.write(struct.pack("I", dim))
            fp.write(row.tobytes())


def load_vecs(path: Path) -> tuple[np.ndarray, str]:
    if path.suffix.lower() == ".fvecs":
        return read_fvecs(path), "float"
    if path.suffix.lower() == ".ivecs":
        return read_ivecs(path), "int"
    raise ValueError(f"Unsupported vector file extension: {path}")


def mean_cosine(q: np.ndarray, base: np.ndarray, base_norms: np.ndarray, q_norm: float) -> float:
    if q_norm <= 0:
        return 0.0
    denom = base_norms * q_norm
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean((base @ q) / denom))


def mean_attribute_similarity(q: np.ndarray, base: np.ndarray, skip_value: int | None) -> float:
    if base.size == 0:
        return 0.0
    if skip_value is None:
        return float(np.mean(base == q, axis=1).mean())
    mask = q != skip_value
    if not np.any(mask):
        return 0.0
    return float(np.mean(base[:, mask] == q[mask], axis=1).mean())


def zscore(values: np.ndarray) -> np.ndarray:
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-12:
        return values * 0
    return (values - mean) / std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split queries into visual-dominant and text-dominant groups."
    )
    parser.add_argument("--base1", required=True, type=Path)
    parser.add_argument("--base2", required=True, type=Path)
    parser.add_argument("--query1", required=True, type=Path)
    parser.add_argument("--query2", required=True, type=Path)
    parser.add_argument("--groundtruth", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--gt-k", type=int, default=0)
    parser.add_argument("--skip-value", type=int, default=None)
    parser.add_argument("--normalize-modal1", action="store_true")
    parser.add_argument("--normalize-modal2", action="store_true")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base1, base1_kind = load_vecs(args.base1)
    base2, base2_kind = load_vecs(args.base2)
    query1, query1_kind = load_vecs(args.query1)
    query2, query2_kind = load_vecs(args.query2)
    gt = read_ivecs(args.groundtruth)

    if base1_kind != "float" or query1_kind != "float":
        raise ValueError("Modal1 must be fvecs for this grouping script.")
    if base2_kind != query2_kind:
        raise ValueError("Modal2 base/query must share the same format.")

    base_count = min(len(base1), len(base2))
    if len(base1) != len(base2):
        base1 = base1[:base_count]
        base2 = base2[:base_count]

    query_count = min(len(query1), len(query2), len(gt))
    if len(query1) != query_count:
        query1 = query1[:query_count]
    if len(query2) != query_count:
        query2 = query2[:query_count]
    if len(gt) != query_count:
        gt = gt[:query_count]

    gt_k = args.gt_k if args.gt_k > 0 else None
    valid_ids = []
    for qid in range(query_count):
        row = gt[qid]
        if gt_k is not None:
            row = row[:gt_k]
        row = row[(row >= 0) & (row < base_count)]
        if row.size:
            valid_ids.append(qid)

    if not valid_ids:
        raise RuntimeError("No valid queries found after filtering groundtruth indices.")

    candidate_ids = valid_ids
    if args.max_queries and args.max_queries < len(candidate_ids):
        rng = np.random.default_rng(args.seed)
        candidate_ids = rng.choice(candidate_ids, size=args.max_queries, replace=False)
        candidate_ids = np.sort(candidate_ids).tolist()

    base1_norms = np.linalg.norm(base1, axis=1) if args.normalize_modal1 else None
    query1_norms = np.linalg.norm(query1, axis=1) if args.normalize_modal1 else None

    base2_norms = None
    query2_norms = None
    if base2_kind == "float" and args.normalize_modal2:
        base2_norms = np.linalg.norm(base2, axis=1)
        query2_norms = np.linalg.norm(query2, axis=1)

    v_scores = []
    t_scores = []
    kept_ids = []
    for qid in candidate_ids:
        row = gt[qid]
        if gt_k is not None:
            row = row[:gt_k]
        row = row[(row >= 0) & (row < base_count)]
        if row.size == 0:
            continue
        kept_ids.append(qid)

        base1_sel = base1[row]
        q1 = query1[qid]
        if args.normalize_modal1:
            v_scores.append(mean_cosine(q1, base1_sel, base1_norms[row], query1_norms[qid]))
        else:
            v_scores.append(float(np.mean(base1_sel @ q1)))

        if base2_kind == "float":
            base2_sel = base2[row]
            q2 = query2[qid]
            if args.normalize_modal2:
                t_scores.append(mean_cosine(q2, base2_sel, base2_norms[row], query2_norms[qid]))
            else:
                t_scores.append(float(np.mean(base2_sel @ q2)))
        else:
            base2_sel = base2[row]
            q2 = query2[qid]
            t_scores.append(mean_attribute_similarity(q2, base2_sel, args.skip_value))

    if not kept_ids:
        raise RuntimeError("No queries left after scoring.")

    v_scores = np.asarray(v_scores, dtype=np.float64)
    t_scores = np.asarray(t_scores, dtype=np.float64)

    dominance = zscore(v_scores) - zscore(t_scores)
    order = np.argsort(dominance)
    group_size = max(1, int(len(order) * args.quantile))

    group_b_idx = order[:group_size]
    group_a_idx = order[-group_size:]

    group_a_ids = [kept_ids[i] for i in group_a_idx]
    group_b_ids = [kept_ids[i] for i in group_b_idx]

    out_dir = args.out_dir
    out_a = out_dir / "group_a"
    out_b = out_dir / "group_b"

    q1_a = query1[group_a_ids]
    q2_a = query2[group_a_ids]
    gt_a = gt[group_a_ids]
    q1_b = query1[group_b_ids]
    q2_b = query2[group_b_ids]
    gt_b = gt[group_b_ids]

    write_fvecs(out_a / "query_modal1.fvecs", q1_a)
    write_fvecs(out_b / "query_modal1.fvecs", q1_b)

    if query2_kind == "float":
        write_fvecs(out_a / "query_modal2.fvecs", q2_a)
        write_fvecs(out_b / "query_modal2.fvecs", q2_b)
    else:
        write_ivecs(out_a / "query_modal2.ivecs", q2_a)
        write_ivecs(out_b / "query_modal2.ivecs", q2_b)

    write_ivecs(out_a / "groundtruth.ivecs", gt_a)
    write_ivecs(out_b / "groundtruth.ivecs", gt_b)

    np.savetxt(out_a / "query_ids.txt", np.array(group_a_ids, dtype=np.int32), fmt="%d")
    np.savetxt(out_b / "query_ids.txt", np.array(group_b_ids, dtype=np.int32), fmt="%d")

    stats = np.vstack([
        np.array(kept_ids, dtype=np.int32),
        v_scores,
        t_scores,
        dominance,
    ]).T
    np.savetxt(out_dir / "scores.tsv", stats, fmt=["%d", "%.6f", "%.6f", "%.6f"], delimiter="\t")

    print(f"group_a size: {len(group_a_ids)}")
    print(f"group_b size: {len(group_b_ids)}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
