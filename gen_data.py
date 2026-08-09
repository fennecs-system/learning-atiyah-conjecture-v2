import argparse

import torch
from tqdm import tqdm

from tokenizer import encode
from utils import gen_rand_sample_2d_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for the Atiyah pattern-boost model"
    )
    parser.add_argument("--num-samples", "-n", type=int, default=100_000)
    parser.add_argument("--n-points", type=int, default=4)
    parser.add_argument("--round-factor", type=int, default=2)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="how many samples to generate per vectorized batch",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", "-o", type=str, default="data.txt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    with open(args.out, "w") as f, tqdm(total=args.num_samples) as pbar:
        remaining = args.num_samples
        while remaining > 0:
            batch = min(args.batch_size, remaining)
            v, p, dots = gen_rand_sample_2d_data(args.n_points, batch, args.round_factor)
            k = dots.abs().argmax(dim=-1)

            for i in range(batch):
                tokens = encode(v[i], p[i], int(k[i]))
                f.write(",".join(str(x) for x in tokens) + "\n")

            remaining -= batch
            pbar.update(batch)


if __name__ == "__main__":
    main()
