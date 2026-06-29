"""Small Torch/CUDA visibility check for Athena jobs."""

from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether PyTorch can use CUDA.")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit non-zero if torch.cuda.is_available() is false.",
    )
    args = parser.parse_args()

    print("Torch GPU check")
    print(f"  torch.__version__      = {torch.__version__}")
    print(f"  torch.version.cuda     = {torch.version.cuda}")
    print(f"  cuda.is_available      = {torch.cuda.is_available()}")
    print(f"  cuda.device_count      = {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        if args.require_cuda:
            print("ERROR: PyTorch cannot see CUDA, but CUDA is required.", file=sys.stderr)
            return 1
        return 0

    current = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(current)
    print(f"  cuda.current_device    = {current}")
    print(f"  cuda.device_name       = {torch.cuda.get_device_name(current)}")
    print(f"  cuda.capability        = {props.major}.{props.minor}")
    print(f"  cuda.total_memory_gb   = {props.total_memory / 1024**3:.2f}")

    x = torch.ones((8, 8), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"  test_tensor.device     = {y.device}")
    print(f"  test_tensor.sum        = {float(y.sum().cpu()):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
