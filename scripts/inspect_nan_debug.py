"""Small helper to summarize nan_debug .pt files saved by training.

Usage:
    python scripts/inspect_nan_debug.py [path_prefix]

If no path prefix given, searches output/*/nan_debug/*.pt
"""
import sys
from pathlib import Path
import torch


def summarize_file(p: Path):
    try:
        d = torch.load(p)
    except Exception as e:
        print(f"Failed to load {p}: {e}")
        return
    print(f"\nFile: {p}")
    for k, v in d.items():
        if v is None:
            print(f"  {k}: None")
            continue
        try:
            t = v
            if not torch.is_tensor(t):
                import numpy as np
                arr = np.asarray(t)
                try:
                    amin = arr.min()
                    amax = arr.max()
                    has_nan = np.isnan(arr).any()
                except Exception:
                    amin = amax = 'NA'
                    has_nan = 'NA'
                print(f"  {k}: numpy shape={arr.shape}, dtype={arr.dtype}, min={amin}, max={amax}, has_nan={has_nan}")
            else:
                try:
                    amin = t.min().item() if t.numel() > 0 else 'NA'
                    amax = t.max().item() if t.numel() > 0 else 'NA'
                    has_nan = t.isnan().any().item()
                except Exception:
                    amin = amax = has_nan = 'NA'
                print(f"  {k}: tensor shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, min={amin}, max={amax}, has_nan={has_nan}")
        except Exception as e:
            print(f"  {k}: failed to summarize ({e})")


if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else 'output'
    p = Path(base)
    files = list(p.glob('**/nan_debug/*.pt'))
    if not files:
        print('No nan_debug files found under', base)
        sys.exit(0)
    for f in files:
        summarize_file(f)
