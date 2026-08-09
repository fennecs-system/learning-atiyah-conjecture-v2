import torch
from torch import Tensor

# token layout:
# 1..100        -> a quantised value in [0.00, 0.99] (encoded as int(x*100)+1)
# SIGN_TOKEN    -> next value token is negated
# END_BLOCK_TOKEN -> separates the v block / p block / k
# CLASS_START.. -> k (the argmax index), one token per point
SIGN_TOKEN = 102
END_BLOCK_TOKEN = 103
CLASS_START = 104


def vocab_size(n_points: int) -> int:
    """Number of distinct token ids produced by encode() for a given n_points,
    i.e. one past the largest id (CLASS_START + n_points - 1). Token id 0 is
    never produced here; it's reserved by the dataset for START/STOP."""
    return CLASS_START + n_points


def max_encoded_length(n_points: int) -> int:
    """Upper bound on len(encode(v, p, k)) for n_points in R^2: the worst
    case is every value/coordinate being negative, costing a sign token
    each. This only depends on n_points, not on any particular dataset, so
    it stays fixed across pattern-boost generations (and therefore across
    a saved model's position embedding size)."""
    dim = 2
    v_block = 2 * n_points  # sign + value, per point
    p_block = 2 * dim * n_points  # sign + value, per coordinate
    return v_block + 1 + p_block + 1 + 1  # + END_BLOCK_TOKEN x2 + k


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

    return v_tensor, p_tensor, k
