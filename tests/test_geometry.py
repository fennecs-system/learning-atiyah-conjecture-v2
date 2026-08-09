import math

import torch

from utils import _has_duplicate_points, compute_dots, gen_rand_sample_2d_data


def _poly_mul(a, b):
    result = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] += ai * bj
    return result


def _reference_dots(p, v):
    """Direct, unvectorized port of the original per-point construction
    (products of the pairwise stereographic factors, see main.tex's "cell
    classification" section) -- kept only here, to cross-check the
    vectorized utils.compute_dots that replaced it."""
    n = p.shape[0]
    dots = torch.zeros(n, dtype=v.dtype)

    for j in range(n):
        poly = [1.0]
        for k in range(n):
            if j == k:
                continue
            xjk = (p[j, 0] - p[k, 0]).item()
            yjk = (p[j, 1] - p[k, 1]).item()
            m = math.sqrt(xjk**2 + yjk**2)
            xi0 = math.sqrt(max(m + xjk, 0.0))
            xi1 = math.sqrt(max(m - xjk, 0.0))
            factor = [-xi1, xi0] if yjk < 0 else [xi0, xi1]
            poly = _poly_mul(poly, factor)
        coeffs = torch.tensor(poly, dtype=v.dtype)
        dots[j] = torch.dot(coeffs, v)
    return dots


def test_compute_dots_matches_reference_unbatched():
    torch.manual_seed(0)
    for _ in range(20):
        v, p, _ = gen_rand_sample_2d_data(4, batch_size=1)
        v, p = v[0], p[0]
        expected = _reference_dots(p, v)
        actual = compute_dots(p, v)
        assert torch.allclose(actual, expected, atol=1e-4)


def test_compute_dots_batches_match_unbatched():
    torch.manual_seed(0)
    n_points = 4
    v, p, _ = gen_rand_sample_2d_data(n_points, batch_size=8)
    batched = compute_dots(p, v)
    for i in range(8):
        single = compute_dots(p[i], v[i])
        assert torch.allclose(batched[i], single, atol=1e-6)


def test_generalizes_to_other_n_points():
    torch.manual_seed(0)
    for n_points in (3, 5, 6):
        v, p, _ = gen_rand_sample_2d_data(n_points, batch_size=4)
        expected = torch.stack([_reference_dots(p[i], v[i]) for i in range(4)])
        actual = compute_dots(p, v)
        assert torch.allclose(actual, expected, atol=1e-4)


def test_duplicate_points_are_resampled_not_left_degenerate():
    torch.manual_seed(0)
    n_points = 4
    _, p, _ = gen_rand_sample_2d_data(n_points, batch_size=500)
    assert not _has_duplicate_points(p).any()
