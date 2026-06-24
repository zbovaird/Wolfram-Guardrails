#!/usr/bin/env python3
"""Pull fine-tuned parser merged_hf from Hugging Face Hub."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

from colab_runtime_helpers import pull_merged_hf_from_hub  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face model repo, e.g. your-user/wolfram-guardrails-qwen25-3b-parser-v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/finetune/colab_qwen25_3b/merged_hf",
        help="Local directory for merged_hf weights",
    )
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pull_merged_hf_from_hub(
        repo_id=args.repo_id,
        merged_dir=args.output_dir,
        revision=args.revision,
    )
    print("Ready for compare scripts with --candidate-model", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
