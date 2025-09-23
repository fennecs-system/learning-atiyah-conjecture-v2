from .utils import gen_rand_sample_2d_data, encode
import torch

n = 4

# open file to write
with open("data.txt", "w") as f:
    # write lines of data to file
    for i in range(100000):
        data = gen_rand_sample_2d_data(4)
        p, v, k = data
        k = torch.argmax(k.abs()).item()
        tokens = encode(p, v, k)
        f.write(",".join([str(x) for x in tokens]) + "\n")
