from poly import PolyM
from functools import reduce
from random import choice

import torch
from torch import argmax, cat, dot, empty, rand, stack


# rounds a tensor to n decimal places
def round_by(x, n):
    x = (x * 10**n).round() / (10**n)
    return x


def gen_rand_sample_2d_data(n_points: int, round_by: int):
    # uniform in [0.00,0.99]
    p = (0.00 - 0.99) * rand(n_points, 2) + 0.99

    # a linear dependence in [-0.99, 0.99]
    v = (-0.99 - 0.99) * rand(n_points) + 0.99

    # normalize to (0,1)

    # round p and v to 2 decimal places
    # gives quantised numbers of 0-99
    # fits in a quantised 2^10-vector

    p = round_by(p, round_by)
    v = round_by(v, round_by)

    dots = empty(n_points)

    # differences
    # ps[j,k] = p[j] - p[k]
    ps = p.unsqueeze(1) - p.unsqueeze(0)
    # sum of x_ij^2 + y_ij^2
    M = ps.square().sum(2).sqrt()
    xs = ps[:, :, 0]

    Xi = stack(((M + xs).sqrt(), (M - xs).sqrt()))

    # coeff_tensors = []

    for j in range(n_points):
        poly_j = []
        for k in range(n_points):
            if j == k:
                continue
            else:
                y_jk = ps[j, k][1]
                if y_jk < 0:
                    poly_j.append(PolyM([-Xi[1][j, k], Xi[0][j, k]]))
                elif y_jk > 0:
                    poly_j.append(PolyM([Xi[0][j, k], Xi[1][j, k]]))
                else:  # y_jk =0
                    x_jk = ps[j, k][0]
                    if x_jk < 0:
                        poly_j.append(PolyM([Xi[0][j, k], Xi[1][j, k]]))
                    else:
                        poly_j.append(PolyM([Xi[0][j, k], Xi[1][j, k]]))

        prod_poly_j = reduce((lambda x, y: x * y), poly_j).values()
        coeffs = stack(prod_poly_j)
        dots[j] = dot(coeffs, v)

    # max_index = argmax(dots.abs())
    # coeff_tensors.append(coeffs)
    return p, v, dots


# second experiment
# range 0.00 -> 0.99
# map to 1 -> 100
# 102 for sign
# 103 to denote end of p / v / k
# 104 -> 127 (0 indexed)


SIGN_TOKEN = 102
END_BLOCK_TOKEN = 103
CLASS_START = 104


def encode(p, v, k):
    encoded = []
    n, _ = p.shape
    for j in range(n):
        x, y = p[j]
        if x < 0:
            x *= -1
            encoded.append(SIGN_TOKEN)
        encoded.append(int(x * 100) + 1)
        if y < 0:
            y *= -1
            encoded.append(SIGN_TOKEN)
        encoded.append(int(y * 100) + 1)
    encoded.append(END_BLOCK_TOKEN)
    for j in range(n):
        vk = v[j]
        if vk < 0:
            vk *= -1
            encoded.append(SIGN_TOKEN)
        encoded.append(int(vk * 100) + 1)
    encoded.append(END_BLOCK_TOKEN)
    encoded.append(k + CLASS_START)
    return encoded


def decode(encoded):
    p = []
    v = []
    k = None
    is_negative = False  # Flag to track if the current number is negative

    # Split the encoded list at END_BLOCK_TOKENs
    blocks = []
    temp_block = []
    for num in encoded:
        if num == END_BLOCK_TOKEN:
            blocks.append(temp_block)
            temp_block = []
        else:
            temp_block.append(num)
    blocks.append(temp_block)  # For the last block, which will be `k`

    # Decode `p` block
    p_block = blocks[0]
    i = 0
    while i < len(p_block):
        if p_block[i] == SIGN_TOKEN:
            is_negative = True
            i += 1
            continue

        val = (p_block[i] - 1) / 100.0
        if is_negative:
            val *= -1
            is_negative = False
        if i % 2 == 0:
            temp_tuple = (val,)
        else:
            p.append(temp_tuple + (val,))
        i += 1

    # Decode `v` block
    v_block = blocks[1]
    for num in v_block:
        if num == SIGN_TOKEN:
            is_negative = True
            continue

        val = (num - 1) / 100.0
        if is_negative:
            val *= -1
            is_negative = False
        v.append(val)

    # Decode `k`
    k = blocks[2][0] - CLASS_START

    p_tensor = torch.tensor(p, dtype=torch.float32)
    v_tensor = torch.tensor(v, dtype=torch.float32)
    # k_tensor = torch.tensor([k], dtype=torch.int64)  # k is a single value, so wrap in a list

    return p_tensor, v_tensor, k


def compute_max_dot(p, v):
    # in general n points
    n_points = 4
    dots = empty(n_points)

    # differences
    # ps[j,k] = p[j] - p[k]
    ps = p.unsqueeze(1) - p.unsqueeze(0)
    # sum of x_ij^2 + y_ij^2
    M = ps.square().sum(2).sqrt()
    xs = ps[:, :, 0]

    Xi = stack(((M + xs).sqrt(), (M - xs).sqrt()))

    # coeff_tensors = []

    for j in range(n_points):
        poly_j = []
        for k in range(n_points):
            if j == k:
                continue
            else:
                y_jk = ps[j, k][1]
                if y_jk < 0:
                    poly_j.append(PolyM([-Xi[1][j, k], Xi[0][j, k]]))
                elif y_jk > 0:
                    poly_j.append(PolyM([Xi[0][j, k], Xi[1][j, k]]))
                else:  # y_jk =0
                    x_jk = ps[j, k][0]
                    if x_jk < 0:
                        poly_j.append(PolyM([Xi[0][j, k], Xi[1][j, k]]))
                    else:
                        poly_j.append(PolyM([Xi[0][j, k], Xi[1][j, k]]))

        prod_poly_j = reduce((lambda x, y: x * y), poly_j).values()
        coeffs = stack(prod_poly_j)
        dots[j] = dot(coeffs, v)

    max_index = argmax(dots.abs())

    # coeff_tensors.append(coeffs)
    return dots, max_index.item()


def decode_and_check(input):
    # there should be p, v and a k
    # p is 2 n ints, v is n ints, k is 1 int
    p, v, k_out = decode(input)
    dots, k_eval = compute_max_dot(p, v)
    print(f"Predicted {k_out}, found {int(k_eval)} from computing dot of predicted")
    if abs(int(k_eval) - int(k_out)) < 0.001:
        return True
    else:
        return False
    #
