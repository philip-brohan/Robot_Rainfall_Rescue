#!/usr/bin/env python3
"""Print the nth line of each regular file in a directory tree (default: current dir)."""
import sys
from pathlib import Path
import os


def nth_line(path: Path, n: int):
    if n < 1:
        return None
    try:
        with path.open("r", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                if i == n:
                    return line.rstrip("\n")
    except Exception:
        return None
    return None


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Print the Nth line of each regular file in a directory"
    )
    p.add_argument("dir", nargs="?", default=".", help="Directory to scan")
    p.add_argument(
        "-n", "--line", type=int, default=10, help="Line number to print (1-based)"
    )
    args = p.parse_args()

    dirpath = Path(args.dir)
    n = args.line
    if not dirpath.exists():
        print(f"Directory not found: {dirpath}", file=sys.stderr)
        raise SystemExit(2)

    # Walk directory tree and process every regular file; collect files first
    files = [p for p in sorted(dirpath.rglob("*")) if p.is_file()]
    # Compute relative paths (as strings) and column width
    rels = []
    for p in files:
        try:
            rels.append(p.relative_to(dirpath).as_posix())
        except Exception:
            rels.append(os.path.relpath(str(p), str(dirpath)))

    width = max((len(r) for r in rels), default=0)

    for p, rel in zip(files, rels):
        content = nth_line(p, n)
        if content:
            print(f"{rel.ljust(width)} : {content}")
        else:
            print(f"{rel.ljust(width)} : [no {n}th line]")


if __name__ == "__main__":
    main()
