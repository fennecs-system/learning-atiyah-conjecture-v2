import argparse
import glob
import os
import re

import matplotlib.pyplot as plt

from tokenizer import decode
from utils import compute_max_dot


def load_generation(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    lines = [[int(x) for x in line.split(",") if x.strip()] for line in lines]

    values = []
    for line in lines:
        v, p, _k = decode(line)
        dots, k_eval = compute_max_dot(v, p)
        values.append(dots[k_eval].abs().item())
    return values


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Plot the max |dot| distribution for each pattern-boost generation"
    )
    argparser.add_argument(
        "--dir", type=str, default="out", help="directory containing data_generation-*.txt files"
    )
    argparser.add_argument("--bins", type=int, default=50)
    argparser.add_argument(
        "--out", type=str, default=None, help="save the plot to this path instead of/as well as showing it"
    )
    args = argparser.parse_args()

    paths = glob.glob(os.path.join(args.dir, "data_generation-*.txt"))
    paths = sorted(paths, key=lambda p: int(re.search(r"data_generation-(\d+)\.txt$", p).group(1)))

    if not paths:
        raise SystemExit(f"no data_generation-*.txt files found under {args.dir}")

    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap("viridis")(
        [i / max(len(paths) - 1, 1) for i in range(len(paths))]
    )

    for color, path in zip(colors, paths):
        generation = int(re.search(r"data_generation-(\d+)\.txt$", path).group(1))
        values = load_generation(path)
        plt.hist(
            values,
            bins=args.bins,
            histtype="step",
            linewidth=2,
            color=color,
            label=f"generation {generation} (n={len(values)})",
        )

    plt.title(f"max |dot| distribution by generation ({args.dir})")
    plt.xlabel("max |dot|")
    plt.ylabel("count")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out)
        print(f"saved to {args.out}")
    else:
        plt.show()
