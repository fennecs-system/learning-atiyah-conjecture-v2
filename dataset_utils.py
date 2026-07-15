import torch
from torch.utils.data import Dataset, DataLoader

import tokenizer


def _parse_tokens(line: str) -> list[int]:
    return [int(x) for x in line.split(",") if x.strip()]


def _load_lines(path: str) -> list[str]:
    with open(path, "r") as f:
        data = f.read()
    lines = [line.strip() for line in data.splitlines()]
    return [line for line in lines if line]


class TokenDataset(Dataset):
    """Wraps lines of comma-separated integer tokens, as produced by
    tokenizer.encode(), as a next-token-prediction dataset. Each token id is
    used directly (no char-level re-tokenisation): id 0 is reserved for the
    leading <START> token / the <STOP> padding that follows a sequence."""

    def __init__(self, lines: list[str], n_points: int, block_size: int):
        self.lines = lines
        self.n_points = n_points
        self.block_size = block_size  # includes the leading <START> token

    def __len__(self):
        return len(self.lines)

    def contains(self, line: str) -> bool:
        return line in self.lines

    def get_vocab_size(self) -> int:
        return tokenizer.vocab_size(self.n_points) + 1  # +1 for the reserved 0

    def get_output_length(self) -> int:
        return self.block_size

    def encode(self, line: str) -> torch.Tensor:
        return torch.tensor(_parse_tokens(line), dtype=torch.long)

    def decode(self, ids) -> str:
        return ",".join(str(int(i)) for i in ids)

    def __getitem__(self, idx):
        ix = self.encode(self.lines[idx])
        x = torch.zeros(self.block_size, dtype=torch.long)
        y = torch.zeros(self.block_size, dtype=torch.long)
        x[1 : 1 + len(ix)] = ix
        y[: len(ix)] = ix
        y[len(ix) + 1 :] = -1  # index -1 masks the loss at the inactive locations
        return x, y


class StreamDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(
            dataset,
            replacement=False,
        )
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:  # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch


def _split_train_test(lines: list[str], seed: int):
    generator = torch.Generator().manual_seed(seed)
    test_set_size = min(1000, int(len(lines) * 0.1))
    rp = torch.randperm(len(lines), generator=generator).tolist()
    train_lines = [lines[i] for i in rp[:-test_set_size]]
    test_lines = [lines[i] for i in rp[-test_set_size:]]
    print(
        f"split up the dataset into {len(train_lines)} training examples and {len(test_lines)} test examples"
    )
    return train_lines, test_lines


def _build_datasets(lines: list[str], n_points: int, seed: int):
    print(f"number of examples in the dataset: {len(lines)}")

    train_lines, test_lines = _split_train_test(lines, seed)

    # fixed from n_points alone (see tokenizer.max_encoded_length), so it
    # stays the same across pattern-boost generations regardless of which
    # lines happen to be in this particular dataset
    block_size = tokenizer.max_encoded_length(n_points) + 1
    print(f"vocab size: {tokenizer.vocab_size(n_points) + 1}, block size: {block_size}")

    train_dataset = TokenDataset(train_lines, n_points, block_size)
    test_dataset = TokenDataset(test_lines, n_points, block_size)
    return train_dataset, test_dataset


def create_datasets(input_file: str, n_points: int = 4, seed: int = 0):
    lines = _load_lines(input_file)
    return _build_datasets(lines, n_points, seed)


def create_fused_datasets(
    seed_dataset_file: str,
    additional_dataset_files: list[str],
    n_points: int = 4,
    seed: int = 0,
):
    lines = _load_lines(seed_dataset_file)
    for path in additional_dataset_files:
        lines += _load_lines(path)
    return _build_datasets(lines, n_points, seed)
