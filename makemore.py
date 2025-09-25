"""
Forked from https://github.com/karpathy/makemore

you give this script some words (one per line) and it will generate more things like it.
uses super state of the art Transformer AI tech
this code is intended to be super hackable. tune it to your needs.

Changes from minGPT:
- I removed the from_pretrained function where we init with GPT2 weights
- I removed dropout layers because the models we train here are small,
  it's not necessary to understand at this stage and at this scale.
- I removed weight decay and all of the complexity around what parameters are
  and are not weight decayed. I don't believe this should make a massive
  difference at the scale that we operate on here.
"""

import os
import sys
import time
import argparse
from dataclasses import dataclass
from itertools import takewhile
import gc

from utils import decode_and_check, local_search, compute_max_dot, encode
import time

from typing import Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Transformer
from dataset_utils import create_datasets, create_fused_datasets, StreamDataLoader

# -----------------------------------------------------------------------------


def warmup_lambda(step):
    warmup_steps = 500
    if step < warmup_steps:
        return step / warmup_steps

    return 1.0


@dataclass
class ModelConfig:
    block_size: int | None = None  # length of the input sequences of integers
    vocab_size: int | None = (
        None  # the input integers are in range [0 .. vocab_size -1]
    )
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 512  # refers to the total for the multi-head attention so must be divisible by n_head
    n_head: int = 4


def atomic_torch_save(dict, filename):
    tempname = filename + ".tmp"
    torch.save(dict, tempname)
    os.replace(tempname, filename)


# -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def print_samples(num=10):
    """samples from the model and pretty prints the decoded samples"""
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = (
        train_dataset.get_output_length() - 1
    )  # -1 because we already start with <START> token (index 0)
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to("cpu")
    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[
            i, 1:
        ].tolist()  # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        # separately track samples that we have and have not seen before
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    print("-" * 80)
    num_new_correct = 0
    for lst, desc in [
        (train_samples, "in train"),
        (test_samples, "in test"),
        (new_samples, "new"),
    ]:
        print(f"{len(lst)} samples that are {desc}:")

        num_correct = 0
        for word in lst:
            # strip out the commas
            # check if the word is a valid list of integers
            try:
                word = [int(x.strip()) for x in word.split(",") if x.strip()]
                if decode_and_check(word):
                    num_correct += 1
            except Exception as e:
                pass
                # print(f"Could not convert {word} to list of integers: {e}")
                # print the word
                # print(word)

        print(f"Found {num_correct} out of {len(lst)} correct")
        # print(word)
        if desc == "new":
            num_new_correct = num_correct
    print("-" * 80)
    return num_new_correct, len(new_samples)


# a sample must be grammatically correct
def check_sample_valid(word):
    # 4 points, in R^2
    n = 4
    dim = 2

    try:
        ints = [int(x.strip()) for x in word.split(",") if x.strip()]

        # assert the first four tokens before the stop token 103 - eg the v
        # is less than 102  -- allowing for sign
        it = iter(ints)
        v_ints = list(takewhile(lambda x: x < 103, it))
        assert all(x < 103 for x in v_ints)

        # assert at most 8 tokens for v (one token for sign, one for value)
        assert len(v_ints) <= 2 * n

        # check that the next block of tokens before the stop token 103
        # take everything after the head
        it = list(it)[1:]
        p_ints = list(takewhile(lambda x: x < 103, it))

        assert all(x < 103 for x in p_ints)
        # assert not all zeros for p
        assert not all(x == 0 for x in p_ints)

        # assert that at least 50% are non zero
        assert sum(1 for x in p_ints if x != 0) >= 0.5 * len(p_ints)

        # assert that the p ints cant be all the same, possibly removing 102
        assert len(set(p_ints) - {102}) > 1

        # assert at most 16 tokens for p (one token for sign, one for value)
        assert len(p_ints) <= 2 * n * dim

        v, p, k = decode_and_check(ints)

        # k should be a valid index
        assert k >= 0 and k < len(v)

        # assert k should be 1 x 4
        assert v.shape == (n,)
        # p should be 4 x 2
        # two points in 2D for each of the 4 vertices
        assert p.shape == (n, dim)

        return v, p, k
    except Exception as e:
        return None


def generate_n_improved_samples(num=1000, generation=1):
    def process_sample(i: int, X_samp) -> Optional[Tuple[int, str]]:
        """Process a single sample and return result if valid improvement found."""
        try:
            # get the i'th row of sampled integers, as python list
            row = X_samp[
                i, 1:
            ].tolist()  # note: we need to crop out the first <START> token
            # token 0 is the <STOP> token, so we crop the output sequence at that point
            crop_index = row.index(0) if 0 in row else len(row)
            row = row[:crop_index]
            word_samp = train_dataset.decode(row)

            # separately track samples that we have and have not seen before
            if train_dataset.contains(word_samp):
                return None
            elif test_dataset.contains(word_samp):
                return None
            else:
                # its not in the dataset
                try:
                    v, p, k = check_sample_valid(word_samp)
                    found_better, candidates = local_search(v, p, k, 10)
                    improved_p = candidates[0][1]
                    _, new_k_eval = compute_max_dot(v, improved_p)

                    if found_better:
                        tokens = encode(v, improved_p, new_k_eval)
                        tokens_str = ",".join([str(x) for x in tokens]) + "\n"

                        # Thread-safe file writing
                        print(f"Found improved sample {tokens}")

                        return (i, tokens_str)

                except Exception as e:
                    return None

        except Exception as e:
            return None

        return None

    # generate 100 samples at a time
    # keep the ones that are valid and can be improved by local search
    # and are not already in the train or test set
    # repeat until we have num such samples
    num_found = 0
    out_path = os.path.join(run_dir, f"data_generation-{generation}.txt")

    write_batch_num = 10
    current_write_batch = 0

    with open(out_path, "w") as f:
        while num_found < num:
            write_batch_str = ""

            # seed 100 random samples
            X_init = torch.zeros(100, 1, dtype=torch.long).to(args.device)
            top_k = args.top_k if args.top_k != -1 else None
            steps = (
                train_dataset.get_output_length() - 1
            )  # -1 because we already start with <START> token (index 0)
            X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to(
                "cpu"
            )

            # Process samples in parallel with 10 workers
            batch_found = 0
            max_workers = 100

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks for this batch
                futures = {
                    executor.submit(process_sample, i, X_samp): i
                    for i in range(X_samp.size(0))
                }

                # Process completed tasks
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        i, tokens_str = result

                        write_batch_str += tokens_str
                        current_write_batch += 1

                        if current_write_batch >= write_batch_num:
                            f.write(write_batch_str)
                            f.flush()
                            write_batch_str = ""
                            current_write_batch = 0

                        batch_found += 1
                        num_found += 1

                        # Stop early if we've found enough samples
                        if num_found >= num:
                            # Cancel remaining futures to avoid unnecessary work
                            for remaining_future in futures:
                                if not remaining_future.done():
                                    remaining_future.cancel()

                            # flush any remaining write batch
                            if write_batch_str != "":
                                f.write(write_batch_str)
                                f.flush()
                            break

            print(
                f"Batch complete: found {batch_found} new samples. Total: {num_found}/{num}"
            )

            # Break if we've found enough samples
            if num_found >= num:
                break

    print(
        f"Generation {generation} complete: {num_found} improved samples saved to {out_path}"
    )


def train_one_batch(
    model,
    optimizer,
    scheduler,
    batch_loader,
    out_path,
    sample_step,
    generation,
    args,
    total_batches,
    best_loss,
    step,
):
    t0 = time.time()
    # get the next batch, ship to device, and unpack it to input and target
    batch = batch_loader.next()
    batch = [t.to(args.device) for t in batch]
    X, Y = batch

    # feed into the model
    logits, loss = model(X, Y)

    # calculate the gradient, update the weights
    model.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

    # wait for all CUDA work on the GPU to finish then calculate iteration time taken
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.time()

    # logging
    if step % 10 == 0:
        print(
            f"step {step} | loss {loss.item():.4f} | step time {(t1 - t0) * 1000:.2f}ms"
        )

    # evaluate the model
    if step > 0 and step % 500 == 0:
        train_loss, train_acc = evaluate(
            model, train_dataset, batch_size=100, max_batches=10
        )
        test_loss, train_acc = evaluate(
            model, test_dataset, batch_size=100, max_batches=10
        )
        writer.add_scalar("Loss/train", train_loss, step + generation * total_batches)
        writer.add_scalar("Loss/test", test_loss, step + generation * total_batches)

        writer.add_scalar(
            "Accuracy/train", train_acc, step + generation * total_batches
        )
        writer.add_scalar("Accuracy/test", train_acc, step + generation * total_batches)

        # accuracy

        writer.flush()
        print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
        # save the model to disk if it has improved
        if best_loss is None or test_loss < best_loss:
            print(
                f"test loss {test_loss} is the best so far, saving model to {out_path}"
            )

            # save the step count too
            # make it atomic
            state_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "best_loss": best_loss,
            }

            # first generation, dont save generation number
            if generation > 0:
                state_dict["generation"] = generation

            atomic_torch_save(
                state_dict,
                out_path,
            )

            best_loss = test_loss

    # sample from the model
    if step > 0 and step % sample_step == 0:
        num_correct, num_samples = print_samples(num=10)
        writer.add_scalar(
            "Sampling/new_correct",
            num_correct / num_samples,
            step + generation * total_batches,
        )


def train_one_epoch(
    model, optimizer, scheduler, batch_loader, out_path, sample_step, generation, args
):
    best_loss = None
    step = 0
    total_batches = batch_loader.train_loader.__len__()
    print(f"Starting generation {generation} with {total_batches} batches")

    while True:
        try:
            train_one_batch(
                model,
                optimizer,
                scheduler,
                batch_loader,
                out_path,
                sample_step,
                generation,
                args,
                total_batches,
                best_loss,
                step,
            )
            step += 1
        except Exception as e:
            print(e)

        if step >= total_batches:
            print("End of epoch")
            break

        # termination conditions


@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()

    accuracies = []

    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)

        # accuracy
        preds = torch.argmax(logits, dim=-1)
        num_correct = torch.sum((preds == Y) * (Y != -1)).item()
        num_total = torch.sum(Y != -1).item()
        accuracy = num_correct / num_total

        accuracies.append(accuracy)

        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    total_accuracy = sum(accuracies) / len(accuracies)
    model.train()  # reset model back to training mode
    return mean_loss, total_accuracy


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # number of times to
    # train the model to max iters
    # get 10000 samples from the model that
    # 1 - are valid sequences
    # 2 - correctly computes index k
    # 3 - can be improved by local search
    # we collect 10000 such samples every boost iteration

    max_pattern_boost_steps = 30

    # parse command line args
    parser = argparse.ArgumentParser(description="Learning Atiyah Conjecture")
    # system/input/output
    parser.add_argument(
        "--input-file",
        "-i",
        type=str,
        default="names.txt",
        help="input file with things one per line",
    )
    parser.add_argument(
        "--work-dir", "-o", type=str, default="out", help="output working directory"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="when this flag is used, we will resume optimization from existing model in the workdir",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="just sample from the model and quit, don't train",
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=4,
        help="number of data workers for both train/test",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to use for compute, examples: cpu|cuda|cuda:2|mps",
    )
    parser.add_argument("--seed", type=int, default=3407, help="seed")

    # sampling
    parser.add_argument(
        "--top-k", type=int, default=-1, help="top-k for sampling, -1 means no top-k"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="number of samples to generate when using --sample-only",
    )

    parser.add_argument("--vocab-size", type=int, help="vocab size of training data")
    parser.add_argument("--block-size", type=int, help="block size of training data")

    # model
    parser.add_argument("--n-layer", type=int, default=4, help="number of layers")
    parser.add_argument(
        "--n-head", type=int, default=4, help="number of heads (in a transformer)"
    )
    parser.add_argument(
        "--n-embd", type=int, default=64, help="number of feature channels in the model"
    )
    parser.add_argument(
        "--n-embd2",
        type=int,
        default=64,
        help="number of feature channels elsewhere in the model",
    )

    # optimization
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=64,
        help="batch size during optimization",
    )
    parser.add_argument(
        "--learning-rate", "-l", type=float, default=5e-4, help="learning rate"
    )
    parser.add_argument(
        "--weight-decay", "-w", type=float, default=0.01, help="weight decay"
    )
    args = parser.parse_args()
    print(vars(args))

    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if args.resume or args.sample_only:
        # dont need to append a new timestamp
        run_dir = args.work_dir
    else:
        run_dir = os.path.join(args.work_dir, f"run-{timestamp}")
        os.makedirs(run_dir, exist_ok=False)  # raise error if it already exists

    out_path = os.path.join(run_dir, "model.pt")

    loaded = None
    starting_generation = 0

    # first determine the generation
    if (
        args.resume or args.sample_only
    ):  # note: if we sample-only then we also assume we are resuming
        assert os.path.exists(out_path), (
            f"could not find model file {out_path}, expected a model.pt in the workdir"
        )
        print(f"resuming from existing model in the workdir {out_path}")
        loaded = torch.load(out_path, map_location=args.device)
        starting_generation = loaded.get("generation", 0)

    if args.sample_only:
        print_samples(num=args.num_samples)
        sys.exit()

    writer = SummaryWriter(log_dir=run_dir, comment=f"makemore run-{timestamp}")

    # only load the datateset in train mode
    if args.sample_only:
        block_size = args.block_size
        vocab_size = args.vocab_size
    else:
        # init datasets
        if starting_generation == 0:
            print("loading initial dataset")
            train_dataset, test_dataset = create_datasets(args.input_file)
        else:
            # load all the data_generation-*.txt files up to and including starting_generation
            generation_zero = args.input_file
            all_other_data_files = []
            for gen in range(1, starting_generation + 1):
                gen_file = os.path.join(run_dir, f"data_generation-{gen}.txt")
                assert os.path.exists(gen_file), (
                    f"could not find expected dataset file {gen_file}"
                )
                all_other_data_files.append(gen_file)
            print(
                f"loading fused dataset from {len(all_other_data_files)} files, up to generation {starting_generation}"
            )
            # load the fused dataset
            train_dataset, test_dataset = create_fused_datasets(
                generation_zero, all_other_data_files
            )

        vocab_size = train_dataset.get_vocab_size()
        block_size = train_dataset.get_output_length()
        print(f"dataset determined that: {vocab_size=}, {block_size=}")

    # init model
    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )

    model = Transformer(config)
    # model = torch.compile(model)
    model.to(args.device)

    # compile model

    batch_size = args.batch_size

    # init optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

    # init dataloader
    batch_loader = StreamDataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    print(f"model #params: {sum(p.numel() for p in model.parameters())}")

    # if there is a loaded model, load it
    if loaded is not None:
        print(f"resuming from existing model in the workdir {out_path}")
        print("loaded keys:", loaded.keys())

        model.load_state_dict(loaded["model_state_dict"])
        optimizer.load_state_dict(loaded["optimizer_state_dict"])
        step = loaded["step"]
        best_loss = loaded["best_loss"]

    if args.sample_only:
        print_samples(num=args.num_samples)
        sys.exit()

    # training loop

    for generation in range(starting_generation, max_pattern_boost_steps):
        train_one_epoch(
            model,
            optimizer,
            scheduler,
            batch_loader,
            out_path,
            sample_step=500,
            generation=generation,
            args=args,
        )

        print(f"finished generation {generation}, generating new samples")

        # set model to
        model.eval()
        # save in generation+1, since we take initial dataset as generation 0
        generate_n_improved_samples(num=1000, generation=generation + 1)

        # rebuild the dataloaders with the new data
        all_other_data_files = []

        for gen in range(1, generation + 2):
            gen_file = os.path.join(run_dir, f"data_generation-{gen}.txt")
            if os.path.exists(gen_file):
                all_other_data_files.append(gen_file)

        print(
            f"loading fused dataset from {len(all_other_data_files)} files, up to generation {generation + 1}"
        )
        generation_zero = args.input_file

        train_dataset, test_dataset = create_fused_datasets(
            generation_zero, all_other_data_files
        )
        batch_loader = StreamDataLoader(
            train_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
        )

        # set model back to train mode
        model.train()
