import torch
from torch.utils.data import Dataset, DataLoader
import random


class CharDataset(Dataset):
    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}  # inverse mapping

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1  # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_word_length + 1  # <START> token followed by words

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = "".join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1 : 1 + len(ix)] = ix
        y[: len(ix)] = ix
        y[len(ix) + 1 :] = -1  # index -1 will mask the loss at the inactive locations
        return x, y


class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(
            dataset, replacement=True, num_samples=int(1e10)
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


def create_datasets(input_file):
    # preprocessing of the input text file
    with open(input_file, "r") as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words]  # get rid of any leading or trailing white space
    words = [w for w in words if w]  # get rid of any empty strings
    chars = sorted(list(set("".join(words))))  # all the possible characters
    max_word_length = max(len(w) for w in words)
    print(f"number of examples in the dataset: {len(words)}")
    print(f"max word length: {max_word_length}")
    print(f"number of unique characters in the vocabulary: {len(chars)}")
    print("vocabulary:")
    print("".join(chars))

    # partition the input data into a training and the test set
    test_set_size = min(
        1000, int(len(words) * 0.1)
    )  # 10% of the training set, or up to 1000 examples
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    print(
        f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples"
    )

    # wrap in dataset objects
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset


def create_fused_datasets(seed_dataset_file, additional_dataset_files):
    # create the base dataset
    with open(seed_dataset_file, "r") as f:
        data = f.read()

    words = data.splitlines()
    words = [w.strip() for w in words]  # get rid of any leading or trailing white space
    words = [w for w in words if w]  # get rid of any empty strings
    chars = sorted(list(set("".join(words))))  # all the possible characters
    max_word_length = max(len(w) for w in words)

    total_words = words

    for file in additional_dataset_files:
        with open(file, "r") as f:
            data = f.read()

        words = data.splitlines()
        words = [w.strip() for w in words]  # get rid of any leading
        words = [w for w in words if w]  # get rid of any empty strings

        total_words += words

    # randomly shuffle the total words

    total_words = random.shuffle(total_words)

    test_set_size = min(
        1000, int(len(total_words) * 0.1)
    )  # 10% of the training set, or up to 1000 examples

    rp = torch.randperm(len(total_words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    print(
        f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples"
    )

    # wrap in dataset objects
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset
