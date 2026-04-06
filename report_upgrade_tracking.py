#!/usr/bin/env python3
"""
Print upgrade-stage metric history and deltas from checkpoints/upgrade_tracking.json.
"""

import argparse
import json
import os


def format_row(cols, widths):
    return " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))


def main():
    parser = argparse.ArgumentParser(description="Report staged steganalysis upgrade deltas.")
    parser.add_argument(
        "--path",
        type=str,
        default="checkpoints/upgrade_tracking.json",
        help="Path to upgrade tracking JSON file.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise SystemExit(f"Tracking file not found: {args.path}")

    with open(args.path, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if not runs:
        print("No tracked runs found.")
        return

    headers = [
        "Stage",
        "Test Acc",
        "Test AUC",
        "Test F1",
        "dAcc",
        "dAUC",
        "dF1",
    ]

    rows = []
    for r in runs:
        rows.append(
            [
                r.get("stage_name", "unknown"),
                f'{float(r.get("test_acc", 0.0)):.4f}',
                f'{float(r.get("test_auc", 0.0)):.4f}',
                f'{float(r.get("test_f1", 0.0)):.4f}',
                f'{float(r.get("delta_test_acc", 0.0)):+.4f}',
                f'{float(r.get("delta_test_auc", 0.0)):+.4f}',
                f'{float(r.get("delta_test_f1", 0.0)):+.4f}',
            ]
        )

    widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]
    print(format_row(headers, widths))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(format_row(row, widths))


if __name__ == "__main__":
    main()

