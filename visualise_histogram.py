import os
import sys
import glob
import matplotlib.pyplot as plt
from utils import decode, compute_max_dot
import torch
import numpy as np
from scipy import stats


def main():
    if len(sys.argv) != 2:
        print("Usage: python visualise_histogram.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    # Find all data_generation-*.txt files
    pattern = os.path.join(directory, "data_generation-*.txt")
    files = glob.glob(pattern)

    if not files:
        print(f"No data_generation-*.txt files found in {directory}")
        sys.exit(1)

    print(f"Found {len(files)} files to process.")
    plt.figure(figsize=(10, 6))

    colors = plt.get_cmap("plasma")(np.linspace(0.2, 0.9, len(files)))

    # sort files by generation number
    sorted_files = sorted(
        files, key=lambda x: int(os.path.basename(x).split("-")[-1].split(".")[0])
    )

    for i, file_path in enumerate(sorted_files):
        max_dots = []
        filename = os.path.basename(file_path)

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    line = [int(x) for x in line.split(",") if x.strip()]
                    v, p, _k = decode(line)
                    dots, _k = compute_max_dot(v, p)
                    max_dots.append(torch.max(dots.abs()).item())

        if max_dots:
            bins = np.linspace(0, 3.0, 200)
            counts, bins, _ = plt.hist(
                max_dots,
                alpha=0.1 + 0.01 * i,
                bins=bins,
                label=f"{filename} (hist)",
                color=colors[i],
            )

            kde = stats.gaussian_kde(max_dots)
            x_smooth = np.linspace(0, 3.0, 200)
            y_smooth = (
                kde(x_smooth) * len(max_dots) * (bins[1] - bins[0])
            )  # Scale to match histogram
            plt.plot(
                x_smooth,
                y_smooth,
                linewidth=2,
                label=f"{filename} (fit)",
                color=colors[i],
            )
            # Fit and plot a smooth curve

    plt.xlabel("Max Dot Product")
    plt.ylabel("Count")
    plt.title("Distribution of Max Dot Products by Generation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
