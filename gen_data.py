from utils import gen_rand_sample_2d_data, encode
import torch

n = 4

# open file to write
with open("data.txt", "a") as f:
    # write lines of data to file
    for i in range(500000):
        data = gen_rand_sample_2d_data(4, 2)
        v, p, dots = data
        k = torch.argmax(dots.abs()).item()
        tokens = encode(v, p, k)
        f.write(",".join([str(x) for x in tokens]) + "\n")
