#!/usr/bin/env python3

import argparse
import sys


def load_success_set(path):
    success = set()
    total = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                raise ValueError(f"Invalid line in {path}: {line}")
            try:
                qid = int(parts[0])
                is_correct = int(parts[1])
            except ValueError as exc:
                raise ValueError(f"Invalid line in {path}: {line}") from exc
            total += 1
            if is_correct == 1:
                success.add(qid)
    return success, total


def main():
    parser = argparse.ArgumentParser(
        description="Analyze M1 weight sensitivity using per-query result files."
    )
    parser.add_argument("--target", default="res_target.txt", help="Setting A results file")
    parser.add_argument("--aux", default="res_aux.txt", help="Setting B results file")
    parser.add_argument("--balance", default="res_balance.txt", help="Setting C results file")
    args = parser.parse_args()

    a_success, a_total = load_success_set(args.target)
    b_success, b_total = load_success_set(args.aux)
    c_success, c_total = load_success_set(args.balance)

    if len({a_total, b_total, c_total}) != 1:
        print(
            f"[WARN] Totals differ: A={a_total}, B={b_total}, C={c_total}",
            file=sys.stderr,
        )

    a_only = a_success - b_success
    b_only = b_success - a_success
    a_and_b = a_success & b_success
    a_or_b = a_success | b_success
    c_fail_but_a_or_b = a_or_b - c_success

    print(f"[SUMMARY] A success: {len(a_success)} / {a_total}")
    print(f"[SUMMARY] B success: {len(b_success)} / {b_total}")
    print(f"[SUMMARY] C success: {len(c_success)} / {c_total}")
    print(f"[M1] A\\B (only A): {len(a_only)}")
    print(f"[M1] B\\A (only B): {len(b_only)}")
    print(f"[M1] A_and_B: {len(a_and_b)}")
    print(f"[M1] A_or_B: {len(a_or_b)}")
    print(f"[M1] C fail but A or B success: {len(c_fail_but_a_or_b)}")


if __name__ == "__main__":
    main()
