import torch

from utils import (
    _configuration_scale,
    _search_objective,
    compute_max_dot,
    gen_rand_sample_2d_data,
    local_search,
)


def test_local_search_never_gets_worse():
    torch.manual_seed(0)
    for _ in range(5):
        v, p, _dots = gen_rand_sample_2d_data(4, batch_size=1)
        v, p = v[0], p[0]
        _, k = compute_max_dot(v, p)

        original_objective = _search_objective(v, p)

        found_better, results = local_search(v, p, k, max_num_steps=10)
        best_objective, _best_p = results[0]

        assert best_objective <= original_objective + 1e-6
        if found_better:
            assert best_objective < original_objective


def test_local_search_generalizes_to_other_n_points():
    torch.manual_seed(0)
    for n_points in (3, 5):
        v, p, _dots = gen_rand_sample_2d_data(n_points, batch_size=1)
        v, p = v[0], p[0]
        _, k = compute_max_dot(v, p)

        _found_better, results = local_search(v, p, k, max_num_steps=5)

        assert results[0][1].shape == (n_points, 2)


def test_objective_is_scale_invariant():
    """Uniformly rescaling a configuration must not change
    _search_objective -- otherwise local_search could "win" by shrinking the
    whole configuration rather than finding a genuinely different shape."""
    torch.manual_seed(0)
    v, p, _dots = gen_rand_sample_2d_data(4, batch_size=1)
    v, p = v[0], p[0]

    baseline = _search_objective(v, p)
    for scale_factor in (0.1, 2.0, 10.0):
        scaled = p * scale_factor
        assert torch.allclose(_search_objective(v, scaled), baseline, rtol=1e-4)


def test_configuration_scale_is_homogeneous_degree_one():
    torch.manual_seed(0)
    _, p, _dots = gen_rand_sample_2d_data(4, batch_size=1)
    p = p[0]

    base_scale = _configuration_scale(p)
    for scale_factor in (0.1, 2.0, 10.0):
        assert torch.allclose(
            _configuration_scale(p * scale_factor), base_scale * scale_factor, rtol=1e-4
        )
