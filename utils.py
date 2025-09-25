from poly import PolyM
from functools import reduce
from random import choice

import torch
from torch import argmax, dot, empty, rand, stack

from torch import Tensor


# rounds a tensor to n decimal places
def round_by(x, n):
    x = (x * 10**n).round() / (10**n)
    return x


def gen_rand_sample_2d_data(n_points: int, round_factor: int = 2):
    # uniform in [0.00,0.99]
    p = (0.00 - 0.99) * rand(n_points, 2) + 0.99

    # a linear dependence in [-0.99, 0.99]
    v = (-0.99 - 0.99) * rand(n_points) + 0.99
    # normalize to (0,1)

    # truncate p and v to 2 decimal places
    # gives quantised numbers of 0-99
    # fits in a quantised 2^10-vector
    p = torch.clamp(p, 0.00, 0.99)
    v = torch.clamp(v, -0.99, 0.99)

    p = torch.trunc(p * 100) / 100.0
    v = torch.trunc(v * 100) / 100.0

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
    return v, p, dots


# second experiment
# range 0.00 -> 0.99
# map to 1 -> 100
# 102 for sign
# 103 to denote end of p / v / k
# 104 -> 127 (0 indexed)


SIGN_TOKEN = 102
END_BLOCK_TOKEN = 103
CLASS_START = 104


def encode(v, p, k):
    encoded = []
    n, _ = p.shape
    for j in range(n):
        vk = v[j]
        if vk < 0:
            vk *= -1
            encoded.append(SIGN_TOKEN)
        encoded.append(int(vk * 100) + 1)
    encoded.append(END_BLOCK_TOKEN)

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

    encoded.append(k + CLASS_START)
    return encoded


def decode(encoded):
    v = []
    p = []
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

    # Decode `v` block
    v_block = blocks[0]
    for num in v_block:
        if num == SIGN_TOKEN:
            is_negative = True
            continue

        val = (num - 1) / 100.0
        if is_negative:
            val *= -1
            is_negative = False
        v.append(val)

    # Decode `p` block
    p_block = blocks[1]
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

    # Decode `k`
    k = blocks[2][0] - CLASS_START

    p_tensor = torch.tensor(p, dtype=torch.float32)
    v_tensor = torch.tensor(v, dtype=torch.float32)
    # k_tensor = torch.tensor([k], dtype=torch.int64)  # k is a single value, so wrap in a list

    return v_tensor, p_tensor, k


def compute_max_dot(v: Tensor, p: Tensor):
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
    try:
        v, p, k_out = decode(input)
        _dots, k_eval = compute_max_dot(v, p)
        if abs(int(k_eval) - int(k_out)) < 0.001:
            return v, p, k_out
        else:
            return None
    except:
        return None


# make sure p doesnt become 0
# or doesnt become too close
def repulsion_loss(p, min_distance=0.01):
    n_points = p.shape[0]
    eps = 1e-5
    diff = p.unsqueeze(1) - p.unsqueeze(0)
    distances = torch.linalg.vector_norm(diff, dim=2)

    # Adaptive epsilon based on minimum desired distance
    eps = min_distance / 10.0

    distances = torch.clamp(distances, min=eps)

    mask = torch.triu(torch.ones(n_points, n_points), diagonal=1).bool()
    forces = 1.0 / (distances.pow(2) + eps)
    total_force = forces[mask].sum()

    return 0.0001 * total_force


# p is a list of tuples of coords
# v is a list of values
# v, p, k are python tensors
def local_search(v, p, k, max_num_steps=20):
    # given a v, p and k,
    # fix a v, and then try to find a better p
    # by local search.
    # k determines the pot product,

    best_p = p.clone()
    best_dots, k_eval = compute_max_dot(v, best_p)
    assert k_eval == k

    best_dot_squared = best_dots[k_eval].abs() ** 2

    # nelder mead coefficients
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5  # contraction
    sigma = 0.5  # shrink

    step = 0

    vertices = [best_p.clone()]
    best_p_seen = [best_p.clone()]

    for _ in range(8):
        perturbation = 0.01 * (2 * rand(best_p.shape) - 1)
        new_vertex = best_p + perturbation
        # clamp to [0.00,0.99]
        new_vertex = torch.clamp(new_vertex, 0.00, 0.99)

        vertices.append(new_vertex)

    # add an electostatic repulsion loss to keep points apart
    def objective_function(p_candidate):
        dots_found, k_found = compute_max_dot(v, p_candidate)
        return dots_found[k_found].abs() ** 2 + repulsion_loss(p_candidate)

    def clamp(p_candidate):
        return torch.clamp(p_candidate, 0.00, 0.99)

    while True:
        step += 1

        # order (ascending for minimization)
        vertex_dot_values = [objective_function(vtx) for vtx in vertices]

        sorted_indices = sorted(range(9), key=lambda i: vertex_dot_values[i])
        vertices = [vertices[i] for i in sorted_indices]
        vertex_dot_values = [vertex_dot_values[i] for i in sorted_indices]

        # check if best is better than global best
        if vertex_dot_values[0] < best_dot_squared:
            best_p = vertices[0].clone()
            best_dot_squared = vertex_dot_values[0]
            best_p_seen.append(best_p.clone())

        # terminate after max number
        if step >= max_num_steps:
            break

        centroid = sum(vertices[:-1]) / 8.0

        reflected = clamp(centroid + alpha * (centroid - vertices[-1]))
        reflected_value = objective_function(reflected)

        # if reflected point is better than second worst but not better than best
        if vertex_dot_values[0] <= reflected_value < vertex_dot_values[-2]:
            vertices[-1] = reflected
            continue

        # expand
        if reflected_value < vertex_dot_values[0]:
            expanded = clamp(centroid + gamma * (centroid - vertices[-1]))
            expanded_value = objective_function(expanded)

            if expanded_value < reflected_value:
                vertices[-1] = expanded
            else:
                vertices[-1] = reflected
            continue

        # contract
        if reflected_value >= vertex_dot_values[-2]:
            if reflected_value < vertex_dot_values[-1]:
                # outside contraction
                contracted = clamp(centroid + rho * (reflected - centroid))
                contracted_value = objective_function(contracted)

                if contracted_value < reflected_value:
                    vertices[-1] = contracted
                    continue
            else:
                # inside contraction
                contracted = clamp(centroid + rho * (vertices[-1] - centroid))
                contracted_value = objective_function(contracted)

                if contracted_value < vertex_dot_values[-1]:
                    vertices[-1] = contracted
                    continue

        # shrink
        best_vertex = vertices[0]
        for i in range(1, 9):
            vertices[i] = clamp(best_vertex + sigma * (vertices[i] - best_vertex))

    # compute the dots of the best seen p's
    best = [
        (objective_function(p_candidate), p_candidate) for p_candidate in best_p_seen
    ]

    # choose the smallest dot
    # this may just be the original best p
    best_sorted = sorted(best, key=lambda pair: pair[0])

    found_better = best_sorted[0][0] < best_dots[k].abs() ** 2

    return found_better, best_sorted


def test_local_search():
    p = torch.rand(4, 2)
    v = torch.rand(4)
    dots, k_eval = compute_max_dot(v, p)
    print(
        f"Initial max dot at k={k_eval} with value {dots[k_eval].abs()} and p={p} and v={v}"
    )

    found_better, candidates = local_search(v, p, k_eval, 10)

    new_dot = candidates[0][0]
    improved_p = candidates[0][1]

    new_dots, new_k_eval = compute_max_dot(v, improved_p)
    print(
        f"After local search max dot at k={new_k_eval} with value {new_dots[new_k_eval].abs()} and p={improved_p} and v={v}, and found better is {found_better}"
    )

    assert new_dots[new_k_eval].abs() <= dots[k_eval].abs()

    # plot the new points on the same plot
    import matplotlib.pyplot as plt

    plt.scatter(p[:, 0].numpy(), p[:, 1].numpy(), color="blue", label="Original Points")
    plt.scatter(
        improved_p[:, 0].numpy(),
        improved_p[:, 1].numpy(),
        color="red",
        label="Improved Points",
    )
    plt.title("Local Search Improvement of Points")
    plt.show()
