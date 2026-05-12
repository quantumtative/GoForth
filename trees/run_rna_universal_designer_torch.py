#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from trees.rna_models import RnaMaskedConfig, RnaMaskedTransformer, count_parameters
from trees.rna_secondary import (
    BASES,
    SIDE_TOKENS,
    bases_to_ids,
    bases_to_text,
    compute_log_partition,
    generate_conditional_dataset,
    pair_is_allowed,
    pair_log_bonus,
    parse_based_position_order,
    side_matching_pairs,
    structure_to_side,
)


UNKNOWN_SIDE_TOKEN = "?"
CONDITION_TOKENS = (*SIDE_TOKENS, UNKNOWN_SIDE_TOKEN)
CONDITION_TO_ID = {token: index for index, token in enumerate(CONDITION_TOKENS)}
ID_TO_CONDITION = {index: token for token, index in CONDITION_TO_ID.items()}
NEG_INF = -1.0e9


@dataclass(frozen=True)
class UniversalStep:
    condition_ids: tuple[int, ...]
    partial_base_ids: tuple[int, ...]
    masked_positions: tuple[bool, ...]
    target_position: int
    target_base_id: int


class StepDataset(Dataset[UniversalStep]):
    def __init__(self, steps: list[UniversalStep]) -> None:
        self.steps = steps

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int) -> UniversalStep:
        return self.steps[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a universal RNA motif-to-sequence designer.")
    parser.add_argument("--output-dir", default="outputs/rna_universal_designer_finite_temp_12_28_10k_4layer")
    parser.add_argument("--train-min-length", type=int, default=12)
    parser.add_argument("--train-max-length", type=int, default=28)
    parser.add_argument("--val-lengths", default="12,20,28,36")
    parser.add_argument("--train-examples", type=int, default=10000)
    parser.add_argument("--val-examples", type=int, default=1500)
    parser.add_argument("--mask-views-per-example", type=int, default=1)
    parser.add_argument("--temperature-celsius", type=float, default=37.0)
    parser.add_argument("--au-pair-bonus-kcal", type=float, default=1.0)
    parser.add_argument("--cg-pair-bonus-kcal", type=float, default=2.0)
    parser.add_argument("--min-loop-unpaired", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--hold-fraction", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--position-encoding", choices=("learned", "sinusoidal", "fractional", "alibi", "none"), default="fractional")
    parser.add_argument("--sample-count", type=int, default=300)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=451)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_logger(log_file: Any) -> Any:
    def log(message: str) -> None:
        line = f"{utc_timestamp()} {message}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    return log


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pair_log_weights_from_kcal(*, au_pair_bonus_kcal: float, cg_pair_bonus_kcal: float, temperature_celsius: float) -> tuple[float, float]:
    gas_constant = 0.00198720425864083
    beta = 1.0 / (gas_constant * (273.15 + temperature_celsius))
    return beta * au_pair_bonus_kcal, beta * cg_pair_bonus_kcal


def parse_val_lengths(args: argparse.Namespace) -> list[int]:
    return [int(item) for item in args.val_lengths.split(",") if item.strip()]


def generate_examples(args: argparse.Namespace, *, au_log_weight: float, cg_log_weight: float) -> tuple[list[Any], dict[str, list[Any]], int]:
    train_rng = random.Random(args.seed)
    train: list[Any] = []
    for index in range(args.train_examples):
        length = train_rng.randint(args.train_min_length, args.train_max_length)
        train.extend(
            generate_conditional_dataset(
                count=1,
                length=length,
                seed=args.seed * 1_000_003 + index,
                pair_log_weight=0.0,
                au_pair_log_weight=au_log_weight,
                cg_pair_log_weight=cg_log_weight,
                min_loop_unpaired=args.min_loop_unpaired,
            )
        )
    val_sets: dict[str, list[Any]] = {}
    for offset, length in enumerate(parse_val_lengths(args)):
        val_sets[str(length)] = generate_conditional_dataset(
            count=args.val_examples,
            length=length,
            seed=args.seed + 10_000 + offset,
            pair_log_weight=0.0,
            au_pair_log_weight=au_log_weight,
            cg_pair_log_weight=cg_log_weight,
            min_loop_unpaired=args.min_loop_unpaired,
        )
    max_len = max([args.train_max_length, *parse_val_lengths(args)])
    return train, val_sets, max_len


def random_mask_condition(side: tuple[str, ...], rng: random.Random) -> tuple[int, ...]:
    length = len(side)
    reveal_count = rng.randint(0, length)
    reveal = set(rng.sample(range(length), reveal_count))
    return tuple(CONDITION_TO_ID[side[index] if index in reveal else UNKNOWN_SIDE_TOKEN] for index in range(length))


def build_steps(examples: list[Any], *, views_per_example: int, seed: int, mask_id: int) -> list[UniversalStep]:
    rng = random.Random(seed)
    steps: list[UniversalStep] = []
    for example in examples:
        base_ids = bases_to_ids(example.bases)
        order = parse_based_position_order(example.structure)
        side = structure_to_side(example.structure)
        for _ in range(views_per_example):
            condition_ids = random_mask_condition(side, rng)
            partial = [mask_id] * len(base_ids)
            masked = [True] * len(base_ids)
            for position in order:
                steps.append(
                    UniversalStep(
                        condition_ids=condition_ids,
                        partial_base_ids=tuple(partial),
                        masked_positions=tuple(masked),
                        target_position=position,
                        target_base_id=base_ids[position],
                    )
                )
                partial[position] = base_ids[position]
                masked[position] = False
    return steps


def collate_steps(batch: list[UniversalStep], *, target_pad_id: int, condition_pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(step.condition_ids) for step in batch)
    condition_ids = torch.full((len(batch), max_len), condition_pad_id, dtype=torch.long)
    partial_ids = torch.full((len(batch), max_len), target_pad_id, dtype=torch.long)
    masked_positions = torch.zeros((len(batch), max_len), dtype=torch.bool)
    valid_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    target_position = torch.empty((len(batch),), dtype=torch.long)
    target_base_id = torch.empty((len(batch),), dtype=torch.long)
    for row, step in enumerate(batch):
        length = len(step.condition_ids)
        condition_ids[row, :length] = torch.tensor(step.condition_ids, dtype=torch.long)
        partial_ids[row, :length] = torch.tensor(step.partial_base_ids, dtype=torch.long)
        masked_positions[row, :length] = torch.tensor(step.masked_positions, dtype=torch.bool)
        valid_mask[row, :length] = True
        target_position[row] = step.target_position
        target_base_id[row] = step.target_base_id
    return {
        "condition_ids": condition_ids,
        "partial_ids": partial_ids,
        "masked_positions": masked_positions,
        "valid_mask": valid_mask,
        "target_position": target_position,
        "target_base_id": target_base_id,
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def loss_fn(model: RnaMaskedTransformer, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    pos_logits, token_logits = model(batch["partial_ids"], batch["condition_ids"], batch["valid_mask"])
    masked_pos_logits = pos_logits.masked_fill(~batch["masked_positions"], NEG_INF)
    pos_loss = F.cross_entropy(masked_pos_logits, batch["target_position"])
    row = torch.arange(batch["condition_ids"].shape[0], device=batch["condition_ids"].device)
    token_at_target = token_logits[row, batch["target_position"]]
    token_loss = F.cross_entropy(token_at_target, batch["target_base_id"])
    loss = pos_loss + token_loss
    with torch.no_grad():
        pos_acc = (masked_pos_logits.argmax(dim=-1) == batch["target_position"]).float().mean().item()
        tok_acc = (token_at_target.argmax(dim=-1) == batch["target_base_id"]).float().mean().item()
    return loss, {
        "loss": float(loss.item()),
        "position_loss": float(pos_loss.item()),
        "token_loss": float(token_loss.item()),
        "position_accuracy": pos_acc,
        "token_accuracy": tok_acc,
        "step_count": float(batch["condition_ids"].shape[0]),
    }


def evaluate(model: RnaMaskedTransformer, loader: DataLoader[Any], *, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    total_count = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            _, metrics = loss_fn(model, batch)
            count = metrics["step_count"]
            total_count += count
            for key, value in metrics.items():
                if key == "step_count":
                    continue
                totals[key] = totals.get(key, 0.0) + value * count
    return {key: value / max(total_count, 1.0) for key, value in totals.items()}


def scheduled_learning_rate(*, step_index: int, total_steps: int, min_lr: float, peak_lr: float, warmup_steps: int, hold_steps: int) -> float:
    if total_steps <= 1:
        return peak_lr
    if warmup_steps > 0 and step_index < warmup_steps:
        progress = (step_index + 1) / warmup_steps
        return min_lr + progress * (peak_lr - min_lr)
    if step_index < warmup_steps + hold_steps:
        return peak_lr
    decay_steps = max(total_steps - warmup_steps - hold_steps, 1)
    decay_index = min(step_index - warmup_steps - hold_steps, decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_index / decay_steps))
    return min_lr + cosine * (peak_lr - min_lr)


def train_epoch(model: RnaMaskedTransformer, loader: DataLoader[Any], optimizer: torch.optim.Optimizer, *, device: torch.device, global_step: int, total_steps: int, args: argparse.Namespace) -> tuple[dict[str, float], int, float]:
    model.train()
    totals: dict[str, float] = {}
    total_count = 0.0
    last_lr = args.min_learning_rate
    warmup_steps = int(round(args.warmup_fraction * total_steps))
    hold_steps = int(round(args.hold_fraction * total_steps))
    if warmup_steps + hold_steps >= total_steps:
        hold_steps = max(total_steps - warmup_steps - 1, 0)
    for batch in loader:
        batch = move_batch(batch, device)
        last_lr = scheduled_learning_rate(
            step_index=global_step,
            total_steps=total_steps,
            min_lr=args.min_learning_rate,
            peak_lr=args.learning_rate,
            warmup_steps=warmup_steps,
            hold_steps=hold_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = last_lr
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = loss_fn(model, batch)
        loss.backward()
        if args.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        count = metrics["step_count"]
        total_count += count
        for key, value in metrics.items():
            if key == "step_count":
                continue
            totals[key] = totals.get(key, 0.0) + value * count
        global_step += 1
    return {key: value / max(total_count, 1.0) for key, value in totals.items()}, global_step, last_lr


def sample_bases(model: RnaMaskedTransformer, condition_ids: tuple[int, ...], *, temperature: float, count: int, seed: int, device: torch.device) -> list[tuple[str, ...]]:
    rng = random.Random(seed)
    samples: list[tuple[str, ...]] = []
    with torch.no_grad():
        for _ in range(count):
            partial = [model.mask_id] * len(condition_ids)
            for _step in range(len(condition_ids)):
                condition = torch.tensor([condition_ids], dtype=torch.long, device=device)
                partial_ids = torch.tensor([partial], dtype=torch.long, device=device)
                valid = torch.ones_like(condition, dtype=torch.bool)
                masked = partial_ids.eq(model.mask_id)
                pos_logits, token_logits = model(partial_ids, condition, valid)
                pos = int(
                    torch.distributions.Categorical(
                        logits=pos_logits.masked_fill(~masked, NEG_INF).squeeze(0) / temperature
                    )
                    .sample()
                    .item()
                )
                token = int(torch.distributions.Categorical(logits=token_logits[0, pos] / temperature).sample().item())
                partial[pos] = token
            samples.append(tuple(BASES[index] for index in partial))
    return samples


def constrained_log_partition(
    bases: tuple[str, ...] | str,
    condition_side_ids: tuple[int, ...],
    *,
    au_log_weight: float,
    cg_log_weight: float,
    min_loop_unpaired: int,
) -> float:
    base_tuple = tuple(bases)
    length = len(base_tuple)
    full = compute_log_partition(
        base_tuple,
        pair_log_weight=0.0,
        au_pair_log_weight=au_log_weight,
        cg_pair_log_weight=cg_log_weight,
        min_loop_unpaired=min_loop_unpaired,
    )
    log_z = [[-math.inf for _ in range(length + 1)] for _ in range(length + 1)]
    for i in range(length + 1):
        log_z[i][i] = 0.0

    def allows(index: int, token: str) -> bool:
        observed = CONDITION_TOKENS[condition_side_ids[index]]
        return observed == UNKNOWN_SIDE_TOKEN or observed == token

    def logsumexp(values: list[float]) -> float:
        finite = [value for value in values if value != -math.inf]
        if not finite:
            return -math.inf
        top = max(finite)
        return top + math.log(sum(math.exp(value - top) for value in finite))

    for span in range(1, length + 1):
        for i in range(0, length - span + 1):
            j = i + span
            choices: list[float] = []
            if allows(i, "x"):
                choices.append(log_z[i + 1][j])
            if allows(i, "L"):
                for k in range(i + 1, j):
                    if not allows(k, "R"):
                        continue
                    if not pair_is_allowed(base_tuple, i, k, min_loop_unpaired=min_loop_unpaired):
                        continue
                    choices.append(
                        pair_log_bonus(
                            base_tuple[i],
                            base_tuple[k],
                            pair_log_weight=0.0,
                            au_pair_log_weight=au_log_weight,
                            cg_pair_log_weight=cg_log_weight,
                        )
                        + log_z[i + 1][k]
                        + log_z[k + 1][j]
                    )
            log_z[i][j] = logsumexp(choices)
    return log_z[0][length] - full[0][length]


def motif_condition(length: int) -> tuple[int, ...]:
    condition = [CONDITION_TO_ID[UNKNOWN_SIDE_TOKEN]] * length
    start = length // 2 - 1
    for index in range(start, start + 3):
        condition[index] = CONDITION_TO_ID["x"]
    return tuple(condition)


def sample_motif_metrics(
    model: RnaMaskedTransformer,
    *,
    length: int,
    count: int,
    temperature: float,
    seed: int,
    au_log_weight: float,
    cg_log_weight: float,
    min_loop_unpaired: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[str]]:
    condition = motif_condition(length)
    samples = sample_bases(model, condition, temperature=temperature, count=count, seed=seed, device=device)
    log_probs = [
        constrained_log_partition(
            bases,
            condition,
            au_log_weight=au_log_weight,
            cg_log_weight=cg_log_weight,
            min_loop_unpaired=min_loop_unpaired,
        )
        for bases in samples
    ]
    probs = [math.exp(value) if value != -math.inf else 0.0 for value in log_probs]
    pair_counts = []
    for bases in samples:
        # Estimate pair count scale from exact full distribution with a small DP sample surrogate omitted;
        # motif probability is the primary exact metric here.
        pair_counts.append(0)
    metrics = {
        "motif": "".join(CONDITION_TOKENS[idx] for idx in condition),
        "motif_start": length // 2 - 1,
        "avg_exact_motif_probability": sum(probs) / max(len(probs), 1),
        "max_exact_motif_probability": max(probs) if probs else 0.0,
        "median_exact_motif_probability": sorted(probs)[len(probs) // 2] if probs else 0.0,
        "distinct_fraction": len(set(samples)) / max(len(samples), 1),
    }
    sample_lines = [
        f"{bases_to_text(bases)}\texact_motif_prob={prob:.6g}" for bases, prob in sorted(zip(samples, probs), key=lambda item: item[1], reverse=True)[:20]
    ]
    return metrics, sample_lines


def main() -> None:
    args = parse_args()
    au_log_weight, cg_log_weight = pair_log_weights_from_kcal(
        au_pair_bonus_kcal=args.au_pair_bonus_kcal,
        cg_pair_bonus_kcal=args.cg_pair_bonus_kcal,
        temperature_celsius=args.temperature_celsius,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "train.log").open("w", encoding="utf-8") as log_file:
        log = make_logger(log_file)
        log(f"starting universal designer output_dir={output_dir}")
        log(f"pair_log_weights AU={au_log_weight:.4f} CG={cg_log_weight:.4f}")
        device = choose_device(args.device)
        log(f"using device={device}")
        train_examples, val_sets, max_len = generate_examples(args, au_log_weight=au_log_weight, cg_log_weight=cg_log_weight)
        log(f"datasets train_examples={len(train_examples)} val_sets={list(val_sets)} max_len={max_len}")
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        config = RnaMaskedConfig(
            base_vocab_size=len(CONDITION_TOKENS),
            struct_vocab_size=len(BASES),
            max_len=max_len,
            d_model=args.d_model,
            n_head=args.heads,
            n_layer=args.layers,
            dim_feedforward=args.ffn_size,
            dropout=args.dropout,
            position_encoding=args.position_encoding,
        )
        model = RnaMaskedTransformer(config).to(device)
        train_steps = build_steps(
            train_examples,
            views_per_example=args.mask_views_per_example,
            seed=args.seed + 100,
            mask_id=model.mask_id,
        )
        val_steps = {
            key: build_steps(value, views_per_example=1, seed=args.seed + 200 + int(key), mask_id=model.mask_id)
            for key, value in val_sets.items()
        }
        log(
            f"built steps train={len(train_steps)} "
            f"val={{{', '.join(f'{key}:{len(value)}' for key, value in val_steps.items())}}}"
        )
        train_loader = DataLoader(
            StepDataset(train_steps),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda batch: collate_steps(batch, target_pad_id=model.pad_id, condition_pad_id=model.base_pad_id),
        )
        val_loaders = {
            key: DataLoader(
                StepDataset(value),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=lambda batch: collate_steps(batch, target_pad_id=model.pad_id, condition_pad_id=model.base_pad_id),
            )
            for key, value in val_steps.items()
        }
        log(f"model parameters={count_parameters(model)}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
        total_steps = args.epochs * len(train_loader)
        log(f"optimizer_steps={total_steps}")
        history: list[dict[str, Any]] = []
        global_step = 0
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            train_metrics, global_step, lr = train_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                global_step=global_step,
                total_steps=total_steps,
                args=args,
            )
            val_metrics = {key: evaluate(model, loader, device=device) for key, loader in val_loaders.items()}
            record: dict[str, Any] = {"epoch": epoch, "learning_rate": lr, "train": train_metrics, "val": val_metrics}
            if epoch % args.sample_every == 0 or epoch == args.epochs:
                motif_metrics: dict[str, Any] = {}
                for length in parse_val_lengths(args):
                    metrics, samples = sample_motif_metrics(
                        model,
                        length=length,
                        count=args.sample_count,
                        temperature=args.sample_temperature,
                        seed=args.seed + 10_000 * epoch + length,
                        au_log_weight=au_log_weight,
                        cg_log_weight=cg_log_weight,
                        min_loop_unpaired=args.min_loop_unpaired,
                        device=device,
                    )
                    motif_metrics[str(length)] = metrics
                    (output_dir / f"motif_samples_len_{length}_epoch_{epoch:03d}.txt").write_text(
                        "\n".join(samples) + "\n",
                        encoding="utf-8",
                    )
                record["motif_sample"] = motif_metrics
            history.append(record)
            sample_msg = ""
            if "motif_sample" in record:
                sample_msg = " " + " ".join(
                    f"motif_len_{key}_p={value['avg_exact_motif_probability']:.3f}"
                    for key, value in record["motif_sample"].items()
                )
            first_key = next(iter(val_metrics))
            log(
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                f"val[{first_key}]_loss={val_metrics[first_key]['loss']:.4f} lr={lr:.2e}{sample_msg}"
            )
        torch.save(model.state_dict(), output_dir / "model_final.pt")
        summary = {
            "args": vars(args),
            "resolved_pair_log_weights": {
                "au_pair_log_weight": au_log_weight,
                "cg_pair_log_weight": cg_log_weight,
            },
            "condition_tokens": list(CONDITION_TOKENS),
            "target_base_tokens": list(BASES),
            "config": asdict(config),
            "history": history,
            "elapsed_sec": time.time() - start,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(f"wrote summary path={summary_path}")


if __name__ == "__main__":
    main()
