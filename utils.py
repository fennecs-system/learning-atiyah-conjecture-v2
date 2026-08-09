import torch
import torch.nn.functional as F
from torch import Tensor

from tokenizer import decode


def _poly_mul_step(coeffs: Tensor, factor_a: Tensor, factor_b: Tensor) -> Tensor:
    """Multiply a batch of polynomials (increasing-degree coefficients, shape
    [..., d+1]) by the linear factor (factor_a + factor_b * z), where
    factor_a/factor_b have shape [...] (one scalar per leading dim). Returns
    the new coefficients, shape [..., d+2]."""
    a = factor_a.unsqueeze(-1)
    b = factor_b.unsqueeze(-1)
    return F.pad(coeffs, (0, 1)) * a + F.pad(coeffs, (1, 0)) * b


def compute_dots(p: Tensor, v: Tensor) -> Tensor:
    """For a configuration of n points p (shape [..., n, 2]) and weights v
    (shape [..., n]), build the degree-(n-1) polynomial rho_j(z) for each
    point j from the pairwise stereographic factors (see main.tex, "cell
    classification"), and return dots[..., j] = <coeffs(rho_j), v>.

    This is the batched replacement for the old per-point Python loop over
    poly.py's PolyM: multiplying the n-1 linear factors of rho_j is just an
    iterative convolution, which vectorizes over every leading dim (batch,
    and the "which point" index j) at once.
    """
    n = p.shape[-2]

    ps = p.unsqueeze(-2) - p.unsqueeze(-3)  # ps[..., j, k, :] = p_j - p_k
    xs = ps[..., 0]
    ys = ps[..., 1]
    m = ps.square().sum(-1).sqrt()

    xi0 = (m + xs).clamp_min(0).sqrt()
    xi1 = (m - xs).clamp_min(0).sqrt()

    negative_y = ys < 0
    factor_a = torch.where(negative_y, -xi1, xi0)
    factor_b = torch.where(negative_y, xi0, xi1)

    # the k == j entry must not contribute to rho_j's product, so make it the
    # identity factor (1 + 0*z)
    diag = torch.eye(n, dtype=torch.bool, device=p.device)
    factor_a = factor_a.masked_fill(diag, 1.0)
    factor_b = factor_b.masked_fill(diag, 0.0)

    coeffs = torch.ones(p.shape[:-2] + (n, 1), dtype=p.dtype, device=p.device)
    for k in range(n):
        coeffs = _poly_mul_step(coeffs, factor_a[..., :, k], factor_b[..., :, k])
    # each rho_j's k==j step is an identity factor (1 + 0*z), which still
    # pads the array by one column despite contributing nothing, so the
    # loop leaves a spurious always-zero top coefficient; drop it.
    coeffs = coeffs[..., :n]
    # coeffs: [..., n (index j), n (coefficient of z^c)]

    return torch.einsum("...jc,...c->...j", coeffs, v)


def _has_duplicate_points(p: Tensor) -> Tensor:
    """True for batch rows where two of the n points coincide exactly (which
    would otherwise make rho_j/rho_k a degenerate zero polynomial)."""
    n = p.shape[-2]
    diff = p.unsqueeze(-2) - p.unsqueeze(-3)
    dist = diff.square().sum(-1).sqrt()
    diag = torch.eye(n, dtype=torch.bool, device=p.device)
    dist = dist.masked_fill(diag, float("inf"))
    return (dist == 0).any(dim=-1).any(dim=-1)


def _configuration_scale(p: Tensor) -> Tensor:
    """A homogeneous-degree-1 measure of a configuration's size (the mean
    pairwise distance): scaling p by lambda scales this by lambda too. Used
    to make the local-search objective and repulsion_loss scale invariant,
    since the underlying conjecture is about a configuration up to overall
    scale, not its absolute size."""
    n_points = p.shape[-2]
    diff = p.unsqueeze(-2) - p.unsqueeze(-3)
    distances = torch.linalg.vector_norm(diff, dim=-1)
    mask = torch.triu(torch.ones(n_points, n_points, device=p.device), diagonal=1).bool()
    num_pairs = n_points * (n_points - 1) / 2
    return (distances * mask).sum(dim=(-2, -1)).clamp_min(1e-6) / num_pairs


def gen_rand_sample_2d_data(n_points: int, batch_size: int = 1, round_factor: int = 2):
    """Generate `batch_size` random configurations of n_points in the plane
    (coordinates quantised to `round_factor` decimal places) plus a random
    weight vector v, and dots = compute_dots(p, v). Returns v, p, dots, each
    with a leading batch dimension.
    """
    if round_factor != 2:
        # tokenizer.encode()/decode() hardcode a *100 quantisation scale, so
        # this can't be changed independently without also updating those.
        raise NotImplementedError(
            "tokenizer.encode assumes 2-decimal quantisation; round_factor must be 2"
        )

    def sample(n):
        p = torch.clamp((0.00 - 0.99) * torch.rand(n, n_points, 2) + 0.99, 0.00, 0.99)
        v = torch.clamp((-0.99 - 0.99) * torch.rand(n, n_points) + 0.99, -0.99, 0.99)
        p = torch.trunc(p * 100) / 100.0
        v = torch.trunc(v * 100) / 100.0
        return v, p

    v, p = sample(batch_size)

    # points sit on a coarse 100x100 grid, so exact duplicates are rare but
    # possible; resample just those rows instead of silently producing a
    # degenerate (all-zero) polynomial for them.
    for _ in range(10):
        bad = _has_duplicate_points(p)
        if not bad.any():
            break
        v_new, p_new = sample(int(bad.sum()))
        v[bad] = v_new
        p[bad] = p_new

    dots = compute_dots(p, v)
    return v, p, dots


def compute_max_dot(v: Tensor, p: Tensor):
    dots = compute_dots(p, v)
    return dots, dots.abs().argmax().item()


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
    except Exception:
        return None


# Penalizes points that are close *relative to the configuration's own
# scale* (see _configuration_scale) -- this is about discouraging a
# degenerate sub-configuration (two points collapsing together), which is
# a different failure mode from the configuration overall shrinking, and
# needs to stay scale invariant just like the rest of the search objective.
def repulsion_loss(p: Tensor, min_distance: float = 0.01) -> Tensor:
    n_points = p.shape[-2]
    scale = _configuration_scale(p).unsqueeze(-1).unsqueeze(-1)
    diff = p.unsqueeze(-2) - p.unsqueeze(-3)
    distances = torch.linalg.vector_norm(diff, dim=-1) / scale

    eps = min_distance / 10.0
    distances = torch.clamp(distances, min=eps)

    mask = torch.triu(torch.ones(n_points, n_points, device=p.device), diagonal=1).bool()
    forces = 1.0 / (distances.pow(2) + eps)
    total_force = (forces * mask).sum(dim=(-2, -1))

    return 0.0001 * total_force


def _search_objective(v: Tensor, p: Tensor) -> Tensor:
    """The quantity local_search minimizes: the squared max |dot|, made
    scale invariant (by dividing by the configuration's own scale, raised
    to the power that cancels compute_dots' scaling -- see
    _configuration_scale) plus a scale-invariant repulsion term. Without
    this normalization, uniformly shrinking p toward a single point trivially
    drives the raw squared max |dot| to zero, which is a cheat rather than a
    genuine improvement: the conjecture is about a configuration up to
    overall scale. Works for a single configuration (p: [n, 2]) or a batch
    of them (p: [..., n, 2])."""
    n_points = p.shape[-2]
    v = v.expand(p.shape[:-2] + v.shape[-1:])
    dots = compute_dots(p, v)
    max_dot_sq = dots.abs().amax(dim=-1) ** 2
    scale = _configuration_scale(p)
    return max_dot_sq / scale.pow(n_points - 1) + repulsion_loss(p)


# p is a tensor of points (n_points, dim)
# v is a vector of weights (n_points,)
# k is the argmax index of |dots|, fixed throughout the search
def local_search(v: Tensor, p: Tensor, k: int, max_num_steps: int = 20):
    # given a v, p and k, fix v, and try to find a better p by local search
    # (Nelder-Mead over the point coordinates), minimising _search_objective.
    best_p = p.clone()
    _, k_eval = compute_max_dot(v, best_p)
    assert k_eval == k

    original_objective = _search_objective(v, best_p)
    best_objective = original_objective

    # nelder mead coefficients
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5  # contraction
    sigma = 0.5  # shrink

    n_vertices = p.numel() + 1  # simplex in R^(n_points * dim)

    def clamp(p_candidate):
        return torch.clamp(p_candidate, 0.00, 0.99)

    vertices = [best_p.clone()]
    best_p_seen = [best_p.clone()]

    for _ in range(n_vertices - 1):
        perturbation = 0.01 * (2 * torch.rand(best_p.shape) - 1)
        vertices.append(clamp(best_p + perturbation))

    step = 0
    while True:
        step += 1

        # order (ascending for minimization), evaluated as one batched call
        vertex_values = _search_objective(v, torch.stack(vertices)).tolist()

        sorted_indices = sorted(range(n_vertices), key=lambda i: vertex_values[i])
        vertices = [vertices[i] for i in sorted_indices]
        vertex_values = [vertex_values[i] for i in sorted_indices]

        # check if best is better than global best
        if vertex_values[0] < best_objective:
            best_p = vertices[0].clone()
            best_objective = vertex_values[0]
            best_p_seen.append(best_p.clone())

        # terminate after max number
        if step >= max_num_steps:
            break

        centroid = sum(vertices[:-1]) / (n_vertices - 1)

        reflected = clamp(centroid + alpha * (centroid - vertices[-1]))
        reflected_value = _search_objective(v, reflected)

        # if reflected point is better than second worst but not better than best
        if vertex_values[0] <= reflected_value < vertex_values[-2]:
            vertices[-1] = reflected
            continue

        # expand
        if reflected_value < vertex_values[0]:
            expanded = clamp(centroid + gamma * (centroid - vertices[-1]))
            expanded_value = _search_objective(v, expanded)

            if expanded_value < reflected_value:
                vertices[-1] = expanded
            else:
                vertices[-1] = reflected
            continue

        # contract
        if reflected_value >= vertex_values[-2]:
            if reflected_value < vertex_values[-1]:
                # outside contraction
                contracted = clamp(centroid + rho * (reflected - centroid))
                contracted_value = _search_objective(v, contracted)

                if contracted_value < reflected_value:
                    vertices[-1] = contracted
                    continue
            else:
                # inside contraction
                contracted = clamp(centroid + rho * (vertices[-1] - centroid))
                contracted_value = _search_objective(v, contracted)

                if contracted_value < vertex_values[-1]:
                    vertices[-1] = contracted
                    continue

        # shrink
        best_vertex = vertices[0]
        for i in range(1, n_vertices):
            vertices[i] = clamp(best_vertex + sigma * (vertices[i] - best_vertex))

    # compute the objective of the best seen p's
    best = [(_search_objective(v, p_candidate), p_candidate) for p_candidate in best_p_seen]

    # choose the smallest objective value; this may just be the original best p
    best_sorted = sorted(best, key=lambda pair: pair[0])

    found_better = best_sorted[0][0] < original_objective

    return found_better, best_sorted


def test_local_search():
    p = torch.rand(4, 2)
    v = torch.rand(4)
    dots, k_eval = compute_max_dot(v, p)
    original_objective = _search_objective(v, p)
    print(
        f"Initial max dot at k={k_eval} with value {dots[k_eval].abs()} "
        f"(objective {original_objective}) and p={p} and v={v}"
    )

    found_better, candidates = local_search(v, p, k_eval, 10)

    improved_p = candidates[0][1]

    new_dots, new_k_eval = compute_max_dot(v, improved_p)
    new_objective = _search_objective(v, improved_p)
    print(
        f"After local search max dot at k={new_k_eval} with value {new_dots[new_k_eval].abs()} "
        f"(objective {new_objective}) and p={improved_p} and v={v}, and found better is {found_better}"
    )

    # note: the scale-invariant objective can decrease even if the raw max
    # |dot| doesn't, since shrinking the whole configuration alone is not
    # rewarded -- so we compare on the objective, not the raw dots.
    assert new_objective <= original_objective

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
