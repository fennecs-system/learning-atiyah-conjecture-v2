import torch

import dataset_utils
import tokenizer


def _make_line(n_points, k):
    v = torch.zeros(n_points)
    p = torch.zeros(n_points, 2)
    tokens = tokenizer.encode(v, p, k)
    return ",".join(str(t) for t in tokens)


def _write_lines(tmp_path, n_points, count=120):
    lines = [_make_line(n_points, k % n_points) for k in range(count)]
    data_file = tmp_path / "data.txt"
    data_file.write_text("\n".join(lines) + "\n")
    return str(data_file), lines


def test_token_dataset_reports_fixed_vocab_and_block_size(tmp_path):
    n_points = 4
    data_file, lines = _write_lines(tmp_path, n_points)

    train_dataset, test_dataset = dataset_utils.create_datasets(
        data_file, n_points=n_points, seed=0
    )

    assert train_dataset.get_vocab_size() == tokenizer.vocab_size(n_points) + 1
    expected_block_size = tokenizer.max_encoded_length(n_points) + 1
    assert train_dataset.get_output_length() == expected_block_size
    assert test_dataset.get_output_length() == expected_block_size
    assert len(train_dataset) + len(test_dataset) == len(lines)


def test_token_dataset_getitem_shapes_and_start_token(tmp_path):
    n_points = 4
    data_file, _lines = _write_lines(tmp_path, n_points)

    train_dataset, _test_dataset = dataset_utils.create_datasets(
        data_file, n_points=n_points, seed=0
    )

    x, y = train_dataset[0]
    block_size = train_dataset.get_output_length()
    assert x.shape == (block_size,)
    assert y.shape == (block_size,)
    assert x[0].item() == 0  # leading <START>
    assert x[1].item() != 0  # first real token is never the reserved id


def test_split_is_reproducible_given_seed(tmp_path):
    n_points = 4
    data_file, _lines = _write_lines(tmp_path, n_points)

    train_a, test_a = dataset_utils.create_datasets(data_file, n_points=n_points, seed=42)
    train_b, test_b = dataset_utils.create_datasets(data_file, n_points=n_points, seed=42)

    assert train_a.lines == train_b.lines
    assert test_a.lines == test_b.lines
