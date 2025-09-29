import torch
from torch.utils.data import Dataset, DataLoader
import random


# represents each int uniquely,
class IntDataset(Dataset):
    def __init__(self, sequences, max_sequence_length, vocab_size=130):
        self.sequences = sequences
        self.max_sequence_length = max_sequence_length

        # Create vocab: map all integers directly, reserve -1 as special token
        self.stoi = {i: i for i in range(vocab_size)}  # 0-109 direct mapping
        
        self.stoi[128] = 128  # START token
        self.stoi[129] = 129  # STOP token
        self.start_token = self.stoi[128]
        self.stop_token = self.stoi[129]

        self.itos = {i: val for val, i in self.stoi.items()}

        for seq in sequences:
            for val in seq:
                if val not in self.stoi:
                    raise ValueError(
                        f"Value {val} not in vocab range [0-109] or special token -1"
                    )

        self.vocab_ints = list(self.stoi.keys())

        print(f"Vocab size: {len(self.vocab_ints)} integers")
        print(f"Integer range: {min(self.vocab_ints)} to {max(self.vocab_ints)}")

    def __len__(self):
        return len(self.sequences)

    def contains(self, sequence):
        return sequence in self.sequences

    def get_vocab_size(self):
        return max(self.vocab_ints) + 2  # +1 for highest int, +1 for special token

    def get_output_length(self):
        return self.max_sequence_length + 1

    def encode(self, sequence):
        return torch.tensor([self.stoi[val] for val in sequence], dtype=torch.long)

    def decode(self, ix):
        return [self.itos[i] for i in ix if i != self.stop_token]  # skip STOP tokens
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        ix = self.encode(sequence)
       
        x = torch.full((self.max_sequence_length + 1,), self.stop_token, dtype=torch.long)
        y = torch.full(
            (self.max_sequence_length + 1,), -1, dtype=torch.long
        )  # -1 for loss masking
        
        x[0] = self.start_token  # START token
        x[1 : 1 + len(ix)] = ix
        y[: len(ix)] = ix
        y[len(ix)] = self.stop_token

        return x, y


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


class StreamDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(
            dataset,
            replacement=False,  # num_samples=int(1e10)
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

    sequences = []
    lines = data.splitlines()
    for line in lines:
        seq = [int(x.strip()) for x in line.split(",")]
        sequences.append(seq)

    max_sequence_length = max(len(seq) for seq in sequences)
    print(f"Number of sequences: {len(sequences)}")
    print(f"Max sequence length: {max_sequence_length}")

    # up

    # [8 tokens for v]
    # [1 pad token]
    # [16 tokens for p - values and sign]
    # [1 pad token]
    # when there are dots 
    # [8 tokens for the dot values]
    # [1 pad token]
    # [1 token for k]
    #
    # total = 16+8 = 27 + 9 = 36
    # otherwise
    max_sequence_length = 27

    test_size = min(1000, int(len(sequences) * 0.1))
    rp = torch.randperm(len(sequences)).tolist()
    train_sequences = [sequences[i] for i in rp[:-test_size]]
    test_sequences = [sequences[i] for i in rp[-test_size:]]

    train_dataset = IntDataset(train_sequences, max_sequence_length)
    test_dataset = IntDataset(test_sequences, max_sequence_length)
    return train_dataset, test_dataset


def create_fused_int_datasets(seed_dataset_file, additional_dataset_files):
    all_sequences = []

    # Load seed dataset
    with open(seed_dataset_file, "r") as f:
        data = f.read()

    lines = data.splitlines()
    lines = [line.strip() for line in lines if line.strip()]

    for line in lines:
        seq = [int(x.strip()) for x in line.split(",")]
        all_sequences.append(seq)

    # Load additional datasets
    for file in additional_dataset_files:
        with open(file, "r") as f:
            data = f.read()

        lines = data.splitlines()
        lines = [line.strip() for line in lines if line.strip()]

        for line in lines:
            seq = [int(x.strip()) for x in line.split(",")]
            all_sequences.append(seq)

    # Shuffle all sequences
    random.shuffle(all_sequences)

    # max_sequence_length = 36
    # when there is dots the max sequence length adds another seperator
    # +1
    # +4 for the 4 dot values
    # +4 for up to 4 minus signs
    # when there is no dots the max sequence length is 27
    max_sequence_length = 27

    print(f"Number of sequences in fused dataset: {len(all_sequences)}")
    print(f"Max sequence length: {max_sequence_length}")

    # Train/test split
    test_size = min(1000, int(len(all_sequences) * 0.1))
    rp = torch.randperm(len(all_sequences)).tolist()
    train_sequences = [all_sequences[i] for i in rp[:-test_size]]
    test_sequences = [all_sequences[i] for i in rp[-test_size:]]

    print(
        f"Split into {len(train_sequences)} training and {len(test_sequences)} test examples"
    )

    # Create datasets
    train_dataset = IntDataset(train_sequences, max_sequence_length)
    test_dataset = IntDataset(test_sequences, max_sequence_length)

    return train_dataset, test_dataset
