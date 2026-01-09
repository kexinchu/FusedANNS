#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot uqv_syn M1 switch results.")
    parser.add_argument("--input", required=True, type=Path, help="TSV file from uqv_syn_m1_switch.sh")
    parser.add_argument("--output-prefix", required=True, type=Path, help="Output path prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[PLOT] Matplotlib not available: {exc}")
        return

    rows = []
    with open(args.input, "r", encoding="utf-8") as fp:
        header = fp.readline()
        for line in fp:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            group, lamb, topk, recall = parts
            rows.append((group, float(lamb), int(topk), float(recall)))

    if not rows:
        print("[PLOT] No rows to plot.")
        return

    rows_by_topk = {}
    for row in rows:
        rows_by_topk.setdefault(row[2], []).append(row)

    for topk, topk_rows in rows_by_topk.items():
        groups = {}
        for group, lamb, _, recall in topk_rows:
            groups.setdefault(group, []).append((lamb, recall))

        plt.figure(figsize=(6, 4))
        for group, values in groups.items():
            values.sort(key=lambda x: x[0])
            xs = [v[0] for v in values]
            ys = [v[1] for v in values]
            plt.plot(xs, ys, marker="o", label=group)

        plt.title(f"UQV-SYN Fixed Graph: R@{topk}")
        plt.xlabel("lambda")
        plt.ylabel("recall")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path = args.output_prefix.with_suffix(f".topk{topk}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"[PLOT] Saved {out_path}")


if __name__ == "__main__":
    main()
