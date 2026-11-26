#!/usr/bin/env python3
"""Regenerate the 45,000 / 5,000 stratified split from the roll number, independently of the notebooks,
and compare it with the saved split (results/q1_outputs/split_indices.npz).

Usage: python scripts/make_split.py --roll 21CS10045 [--data-root data] [--saved results/q1_outputs/split_indices.npz]
"""
import argparse
from pathlib import Path

import numpy as np
import torchvision
from sklearn.model_selection import train_test_split


def seed_from_roll(roll: str) -> int:
    digits = "".join(ch for ch in roll if ch.isdigit())
    if not digits:
        raise SystemExit("The roll number contains no digits.")
    return int(digits)


def derive_seed(seed: int, *tags) -> int:
    """Same as derive_seed() in the notebooks."""
    words = [seed] + [int.from_bytes(str(t).encode(), "little") for t in tags]
    return int(np.random.SeedSequence(words).generate_state(1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--saved", default="results/q1_outputs/split_indices.npz")
    ap.add_argument("--out", default="results/split_regenerated.npz")
    args = ap.parse_args()

    seed = seed_from_roll(args.roll)
    state = seed if seed < 2**32 else derive_seed(seed, "split")
    y = np.asarray(torchvision.datasets.CIFAR10(args.data_root, train=True, download=True).targets)
    train_idx, val_idx = train_test_split(np.arange(len(y)), train_size=45_000, test_size=5_000,
                                          stratify=y, random_state=state)
    print(f"seed {seed}, random_state {state}")
    print("per-class train:", np.bincount(y[train_idx], minlength=10).tolist())
    print("per-class val:  ", np.bincount(y[val_idx], minlength=10).tolist())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, train_idx=train_idx, val_idx=val_idx, seed=seed)
    print("wrote", args.out)
    if Path(args.saved).exists():
        s = np.load(args.saved)
        same = np.array_equal(s["train_idx"], train_idx) and np.array_equal(s["val_idx"], val_idx)
        print("identical to saved split:", same)
        raise SystemExit(0 if same else 1)


if __name__ == "__main__":
    main()
