import torch

import tokenizer
from utils import gen_rand_sample_2d_data


def test_encode_decode_roundtrip():
    torch.manual_seed(0)
    n_points = 4
    v, p, _dots = gen_rand_sample_2d_data(n_points, batch_size=20)
    for i in range(v.shape[0]):
        k = i % n_points
        tokens = tokenizer.encode(v[i], p[i], k)
        v2, p2, k2 = tokenizer.decode(tokens)

        assert torch.allclose(v[i], v2, atol=1e-6)
        assert torch.allclose(p[i], p2, atol=1e-6)
        assert k2 == k


def test_token_ids_within_vocab_size():
    torch.manual_seed(0)
    n_points = 4
    v, p, _dots = gen_rand_sample_2d_data(n_points, batch_size=20)
    for i in range(v.shape[0]):
        for k in range(n_points):
            tokens = tokenizer.encode(v[i], p[i], k)
            assert all(1 <= t < tokenizer.vocab_size(n_points) for t in tokens)


def test_max_encoded_length_is_an_upper_bound():
    torch.manual_seed(0)
    n_points = 4
    v, p, _dots = gen_rand_sample_2d_data(n_points, batch_size=200)
    bound = tokenizer.max_encoded_length(n_points)
    for i in range(v.shape[0]):
        for k in range(n_points):
            tokens = tokenizer.encode(v[i], p[i], k)
            assert len(tokens) <= bound
