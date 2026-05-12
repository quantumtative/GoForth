#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from trees.rna_models import RnaMaskedConfig, RnaMaskedTransformer, count_parameters
from trees.rna_secondary import BASES, bases_to_ids, parse_based_position_order_side
from trees.run_rna_universal_designer_torch import (
    CONDITION_TOKENS,
    CONDITION_TO_ID,
    ID_TO_CONDITION,
    NEG_INF,
    UNKNOWN_SIDE_TOKEN,
    motif_condition,
    sample_bases,
)


@dataclass(frozen=True)
class CachedSequence:
    source: str
    entry_id: str
    base_ids: tuple[int, ...]
    epoch_structures: tuple[str, ...] = ()
    epoch_structures_packed: str = ""
    epoch_structure_count: int = 0

    @property
    def length(self) -> int:
        return len(self.base_ids)

    def structure_at(self, index: int) -> str:
        if self.epoch_structures_packed:
            count = self.epoch_structure_count
            if count <= 0:
                raise ValueError(f"packed cache row {self.entry_id} has no structure count")
            start = (index % count) * self.length
            return self.epoch_structures_packed[start : start + self.length]
        if not self.epoch_structures:
            raise ValueError(f"cache row {self.entry_id} has no epoch structures")
        return self.epoch_structures[index % len(self.epoch_structures)]


@dataclass(frozen=True)
class UniversalOneStep:
    condition_ids: tuple[int, ...]
    partial_base_ids: tuple[int, ...]
    masked_positions: tuple[bool, ...]
    target_position: int
    target_base_id: int
    length: int
    source: str


class StepDataset(Dataset[UniversalOneStep]):
    def __init__(self, steps: list[UniversalOneStep]) -> None:
        self.steps = steps

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int) -> UniversalOneStep:
        return self.steps[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train p(x|masked t) designer on matrixmodel sampled structures.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--cache",
        action="append",
        required=True,
        help="Source-tagged cache path, as SOURCE:PATH. May be repeated.",
    )
    parser.add_argument(
        "--validation-cache",
        action="append",
        default=[],
        help="Optional source-tagged validation cache path, as SOURCE:PATH. May be repeated.",
    )
    parser.add_argument(
        "--cache-split",
        default=None,
        help="Optional split filter for --cache rows, e.g. train. Rows without this split are skipped.",
    )
    parser.add_argument(
        "--validation-split",
        default=None,
        help="Optional split filter for --validation-cache rows, e.g. holdout.",
    )
    parser.add_argument(
        "--max-structures-per-row",
        type=int,
        default=0,
        help=(
            "If positive, keep only this many sampled structures per row in RAM. "
            "Useful when training for fewer epochs than a large teacher cache contains."
        ),
    )
    parser.add_argument(
        "--reveal-order",
        choices=("parse", "random", "random_pair_follow"),
        default="parse",
        help=(
            "Base reveal order for x during training: parse-derived order, a fresh random permutation, "
            "or a random permutation that immediately follows a paired base with its partner."
        ),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bucket-width", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--hold-fraction", type=float, default=0.55)
    parser.add_argument(
        "--lr-schedule-epochs",
        type=int,
        default=0,
        help="Number of epochs used to size the LR schedule; defaults to --epochs.",
    )
    parser.add_argument(
        "--reset-lr-schedule-on-resume",
        action="store_true",
        help="Start a fresh LR schedule from the resumed global step.",
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--position-encoding", choices=("learned", "sinusoidal", "fractional", "alibi", "none"), default="fractional")
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--validation-every", type=int, default=0)
    parser.add_argument("--validation-batch-size", type=int, default=0)
    parser.add_argument("--feasibility-every", type=int, default=0)
    parser.add_argument("--feasibility-count", type=int, default=256)
    parser.add_argument("--feasibility-batch-size", type=int, default=128)
    parser.add_argument("--feasibility-temperature", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=901)
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
            raise RuntimeError("CUDA requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_cache_arg(spec: str) -> tuple[str, Path]:
    if ":" not in spec:
        raise ValueError("--cache must be SOURCE:PATH")
    source, path = spec.split(":", 1)
    if not source:
        raise ValueError("cache source label cannot be empty")
    return source, Path(path)


def expand_cache_path(path: Path) -> list[Path]:
    if path.is_dir():
        candidate = path / "cache_files.json"
        if not candidate.is_file():
            raise FileNotFoundError(f"cache directory has no cache_files.json: {path}")
        path = candidate
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [Path(item) for item in payload]
        if isinstance(payload, dict) and isinstance(payload.get("cache_files"), list):
            return [Path(item) for item in payload["cache_files"]]
        raise ValueError(f"JSON cache path must contain a list of files or cache_files field: {path}")
    return [path]


def trim_structures(
    *,
    sequence: str,
    structures: tuple[str, ...],
    packed: str,
    structure_count: int,
    max_structures_per_row: int,
) -> tuple[tuple[str, ...], str, int]:
    if max_structures_per_row <= 0:
        return structures, packed, structure_count
    kept = min(structure_count, max_structures_per_row)
    if packed:
        packed = packed[: len(sequence) * kept]
        return (), packed, kept
    return structures[:kept], "", kept


def load_caches(
    cache_specs: list[str],
    *,
    split_filter: str | None = None,
    max_structures_per_row: int = 0,
    progress_log: Any | None = None,
    progress_label: str = "cache",
    progress_every: int = 25,
) -> list[CachedSequence]:
    rows: list[CachedSequence] = []
    for spec in cache_specs:
        source, path = read_cache_arg(spec)
        cache_paths = expand_cache_path(path)
        for file_index, cache_path in enumerate(cache_paths, start=1):
            if progress_log is not None and (
                file_index == 1 or file_index == len(cache_paths) or file_index % max(progress_every, 1) == 0
            ):
                progress_log(
                    f"loading_{progress_label} source={source} "
                    f"file={file_index}/{len(cache_paths)} path={cache_path} rows_so_far={len(rows)}"
                )
            opener = gzip.open if cache_path.suffix == ".gz" else open
            with opener(cache_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if split_filter is not None and record.get("split") != split_filter:
                        continue
                    sequence = record["sequence"]
                    structures = tuple(record.get("epoch_structures", ()))
                    packed = record.get("epoch_structures_packed", "")
                    structure_count = int(record.get("epoch_structure_count", len(structures)))
                    structures, packed, structure_count = trim_structures(
                        sequence=sequence,
                        structures=structures,
                        packed=packed,
                        structure_count=structure_count,
                        max_structures_per_row=max_structures_per_row,
                    )
                    rows.append(
                        CachedSequence(
                            source=source,
                            entry_id=record["id"],
                            base_ids=bases_to_ids(sequence),
                            epoch_structures=structures,
                            epoch_structures_packed=packed,
                            epoch_structure_count=structure_count,
                        )
                    )
    return rows


def random_mask_condition(side: str, rng: random.Random) -> tuple[int, ...]:
    length = len(side)
    reveal_count = rng.randint(0, length)
    reveal = set(rng.sample(range(length), reveal_count))
    return tuple(CONDITION_TO_ID[side[index] if index in reveal else UNKNOWN_SIDE_TOKEN] for index in range(length))


def side_partner_map(side: str) -> dict[int, int]:
    stack: list[int] = []
    pairs: dict[int, int] = {}
    for index, token in enumerate(side):
        if token == "L":
            stack.append(index)
        elif token == "R":
            if not stack:
                raise ValueError("unbalanced side string: extra R")
            left = stack.pop()
            pairs[left] = index
            pairs[index] = left
        elif token != "x":
            raise ValueError(f"unsupported side token {token!r}")
    if stack:
        raise ValueError("unbalanced side string: extra L")
    return pairs


def random_pair_follow_order(side: str, rng: random.Random) -> list[int]:
    pairs = side_partner_map(side)
    remaining = set(range(len(side)))
    order: list[int] = []
    while remaining:
        options = sorted(remaining)
        position = options[rng.randrange(len(options))]
        remaining.remove(position)
        order.append(position)
        partner = pairs.get(position)
        if partner is not None and partner in remaining:
            remaining.remove(partner)
            order.append(partner)
    return order


def reveal_order_for_side(side: str, *, rng: random.Random, reveal_order: str) -> list[int]:
    if reveal_order == "parse":
        return parse_based_position_order_side(side)
    if reveal_order == "random":
        order = list(range(len(side)))
        rng.shuffle(order)
        return order
    if reveal_order == "random_pair_follow":
        return random_pair_follow_order(side, rng)
    raise ValueError(f"unsupported reveal_order={reveal_order!r}")


def build_epoch_steps(
    rows: list[CachedSequence],
    *,
    epoch_index: int,
    mask_id: int,
    seed: int,
    reveal_order: str = "parse",
) -> list[UniversalOneStep]:
    rng = random.Random(seed + 100_003 * epoch_index)
    steps: list[UniversalOneStep] = []
    for row_index, row in enumerate(rows):
        side = row.structure_at(epoch_index)
        order = reveal_order_for_side(side, rng=rng, reveal_order=reveal_order)
        prefix_len = rng.randrange(len(order))
        partial = [mask_id] * row.length
        masked = [True] * row.length
        for position in order[:prefix_len]:
            partial[position] = row.base_ids[position]
            masked[position] = False
        target_position = order[prefix_len]
        condition_ids = random_mask_condition(side, rng)
        steps.append(
            UniversalOneStep(
                condition_ids=condition_ids,
                partial_base_ids=tuple(partial),
                masked_positions=tuple(masked),
                target_position=target_position,
                target_base_id=row.base_ids[target_position],
                length=row.length,
                source=row.source,
            )
        )
    return steps


def make_bucket_batches(steps: list[UniversalOneStep], *, batch_size: int, bucket_width: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    buckets: dict[int, list[int]] = {}
    for index, step in enumerate(steps):
        bucket = step.length // bucket_width
        buckets.setdefault(bucket, []).append(index)
    batches: list[list[int]] = []
    for indices in buckets.values():
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batches.append(indices[start : start + batch_size])
    rng.shuffle(batches)
    return batches


def collate_steps(batch: list[UniversalOneStep], *, target_pad_id: int, condition_pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(step.length for step in batch)
    condition_ids = torch.full((len(batch), max_len), condition_pad_id, dtype=torch.long)
    partial_ids = torch.full((len(batch), max_len), target_pad_id, dtype=torch.long)
    masked_positions = torch.zeros((len(batch), max_len), dtype=torch.bool)
    valid_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    target_position = torch.empty((len(batch),), dtype=torch.long)
    target_base_id = torch.empty((len(batch),), dtype=torch.long)
    for row, step in enumerate(batch):
        length = step.length
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
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


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


def evaluate(model: RnaMaskedTransformer, steps: list[UniversalOneStep], *, batch_size: int, bucket_width: int, device: torch.device, model_pad_id: int, condition_pad_id: int) -> dict[str, float]:
    model.eval()
    batches = make_bucket_batches(steps, batch_size=batch_size, bucket_width=bucket_width, seed=7)
    loader = DataLoader(
        StepDataset(steps),
        batch_sampler=batches,
        num_workers=0,
        collate_fn=lambda batch: collate_steps(batch, target_pad_id=model_pad_id, condition_pad_id=condition_pad_id),
    )
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


def sample_motif_sequences(model: RnaMaskedTransformer, *, length: int, count: int, temperature: float, seed: int, device: torch.device) -> dict[str, Any]:
    condition = motif_condition(length)
    samples = sample_bases(model, condition, temperature=temperature, count=count, seed=seed, device=device)
    text = ["".join(sample) for sample in samples]
    return {
        "motif": "".join(ID_TO_CONDITION[idx] for idx in condition),
        "length": length,
        "distinct_fraction": len(set(text)) / max(len(text), 1),
        "examples": text[:20],
    }


def pair_table_side(side: str) -> dict[int, int]:
    stack: list[int] = []
    pairs: dict[int, int] = {}
    for index, token in enumerate(side):
        if token == "L":
            stack.append(index)
        elif token == "R":
            if not stack:
                raise ValueError("unbalanced side string: extra R")
            left = stack.pop()
            pairs[left] = index
            pairs[index] = left
        elif token != "x":
            raise ValueError(f"unsupported side token {token!r}")
    if stack:
        raise ValueError("unbalanced side string: extra L")
    return pairs


def is_sequence_feasible_for_side(sequence: str, side: str) -> bool:
    pair_ok = {"AU", "UA", "CG", "GC", "GU", "UG"}
    pairs = pair_table_side(side)
    for left, right in pairs.items():
        if left < right and sequence[left] + sequence[right] not in pair_ok:
            return False
    return True


def sample_sequences_for_conditions(
    model: RnaMaskedTransformer,
    condition_rows: list[tuple[int, ...]],
    *,
    temperature: float,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> list[str]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    results: list[str] = []
    row_seed_offset = 0
    with torch.no_grad():
        for start in range(0, len(condition_rows), batch_size):
            rows = condition_rows[start : start + batch_size]
            max_len = max(len(row) for row in rows)
            condition = torch.full((len(rows), max_len), model.base_pad_id, dtype=torch.long, device=device)
            partial = torch.full((len(rows), max_len), model.pad_id, dtype=torch.long, device=device)
            valid = torch.zeros((len(rows), max_len), dtype=torch.bool, device=device)
            for row_index, row in enumerate(rows):
                length = len(row)
                condition[row_index, :length] = torch.tensor(row, dtype=torch.long, device=device)
                partial[row_index, :length] = model.mask_id
                valid[row_index, :length] = True
            arange = torch.arange(len(rows), device=device)
            for _step in range(max_len):
                masked = partial.eq(model.mask_id) & valid
                active = masked.any(dim=1)
                if not bool(active.any().item()):
                    break
                active_rows = arange[active]
                pos_logits, token_logits = model(partial, condition, valid)
                active_pos_logits = pos_logits[active].masked_fill(~masked[active], NEG_INF)
                positions = torch.distributions.Categorical(logits=active_pos_logits / temperature).sample()
                selected_token_logits = token_logits[active_rows, positions]
                tokens = torch.distributions.Categorical(logits=selected_token_logits / temperature).sample()
                partial[active_rows, positions] = tokens
            for row_index, row in enumerate(rows):
                token_ids = partial[row_index, : len(row)].detach().cpu().tolist()
                results.append("".join(BASES[token_id] for token_id in token_ids))
            row_seed_offset += len(rows)
    return results


def evaluate_feasibility(
    model: RnaMaskedTransformer,
    rows: list[CachedSequence],
    *,
    epoch_index: int,
    count: int,
    temperature: float,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    if not rows or count <= 0:
        return {}
    rng = random.Random(seed + 1_000_003 * epoch_index)
    chosen = rng.sample(rows, min(count, len(rows)))
    sides = [row.structure_at(epoch_index) for row in chosen]
    conditions = [tuple(CONDITION_TO_ID[token] for token in side) for side in sides]
    sequences = sample_sequences_for_conditions(
        model,
        conditions,
        temperature=temperature,
        seed=seed + 2_000_003 * epoch_index,
        device=device,
        batch_size=batch_size,
    )
    feasible = sum(is_sequence_feasible_for_side(sequence, side) for sequence, side in zip(sequences, sides))
    lengths = [len(side) for side in sides]
    pair_counts = [side.count("L") for side in sides]
    return {
        "count": float(len(sides)),
        "feasible_count": float(feasible),
        "feasibility_rate": feasible / max(len(sides), 1),
        "mean_length": sum(lengths) / max(len(lengths), 1),
        "mean_pair_count": sum(pair_counts) / max(len(pair_counts), 1),
        "temperature": temperature,
    }


def source_counts(rows: list[CachedSequence]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.source] = counts.get(row.source, 0) + 1
    return counts


def length_summary(rows: list[CachedSequence]) -> dict[str, float | int]:
    lengths = [row.length for row in rows]
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
        "total": sum(lengths),
    }


def json_sanitize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(item) for item in value]
    return value


def save_checkpoint(
    output_dir: Path,
    *,
    epoch: int,
    global_step: int,
    model: RnaMaskedTransformer,
    optimizer: torch.optim.Optimizer,
    history: list[dict[str, Any]],
    args: argparse.Namespace,
    config: RnaMaskedConfig,
) -> Path:
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "args": vars(args),
        "config": asdict(config),
    }
    tmp_path = output_dir / "checkpoint_latest.tmp.pt"
    final_path = output_dir / "checkpoint_latest.pt"
    torch.save(payload, tmp_path)
    tmp_path.replace(final_path)
    return final_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if args.resume_from is not None else "w"
    with (output_dir / "train.log").open(log_mode, encoding="utf-8") as log_file:
        log = make_logger(log_file)
        rows = load_caches(
            args.cache,
            split_filter=args.cache_split,
            max_structures_per_row=args.max_structures_per_row,
            progress_log=log,
            progress_label="train_cache",
        )
        validation_rows = (
            load_caches(
                args.validation_cache,
                split_filter=args.validation_split,
                max_structures_per_row=args.max_structures_per_row,
                progress_log=log,
                progress_label="validation_cache",
            )
            if args.validation_cache
            else []
        )
        if not rows:
            raise ValueError("no training rows loaded; check --cache and --cache-split")
        if args.validation_cache and not validation_rows:
            raise ValueError("no validation rows loaded; check --validation-cache and --validation-split")
        max_len = max([row.length for row in rows] + [row.length for row in validation_rows])
        device = choose_device(args.device)
        log(f"starting output_dir={output_dir}")
        log(f"loaded rows={len(rows)} source_counts={source_counts(rows)} length_summary={length_summary(rows)}")
        if validation_rows:
            log(
                f"loaded validation_rows={len(validation_rows)} "
                f"validation_source_counts={source_counts(validation_rows)} "
                f"validation_length_summary={length_summary(validation_rows)}"
            )
        log(f"device={device}")
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
            torch.set_float32_matmul_precision("high")
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
        log(f"model_parameters={count_parameters(model)} config={asdict(config)}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
        start_epoch = 1
        global_step = 0
        history: list[dict[str, Any]] = []
        if args.resume_from is not None:
            checkpoint = torch.load(args.resume_from, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint.get("global_step", 0))
            history = list(checkpoint.get("history", []))
            log(f"resumed checkpoint={args.resume_from} start_epoch={start_epoch} global_step={global_step}")
        schedule_start_step = global_step if args.reset_lr_schedule_on_resume else 0
        initial_steps_for_schedule = build_epoch_steps(
            rows,
            epoch_index=0,
            mask_id=model.mask_id,
            seed=args.seed,
            reveal_order=args.reveal_order,
        )
        steps_per_epoch = len(
            make_bucket_batches(
                initial_steps_for_schedule,
                batch_size=args.batch_size,
                bucket_width=args.bucket_width,
                seed=args.seed + 1,
            )
        )
        schedule_epochs = args.lr_schedule_epochs if args.lr_schedule_epochs > 0 else args.epochs
        total_steps = schedule_epochs * steps_per_epoch
        warmup_steps = int(round(args.warmup_fraction * total_steps))
        hold_steps = int(round(args.hold_fraction * total_steps))
        if warmup_steps + hold_steps >= total_steps:
            hold_steps = max(total_steps - warmup_steps - 1, 0)
        log(
            "lr_schedule_steps="
            f"{total_steps} schedule_epochs={schedule_epochs} schedule_start_step={schedule_start_step} "
            f"warmup_steps={warmup_steps} hold_steps={hold_steps}"
        )
        start = time.time()
        validation_batch_size = args.validation_batch_size if args.validation_batch_size > 0 else args.batch_size
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_steps = build_epoch_steps(
                rows,
                epoch_index=epoch - 1,
                mask_id=model.mask_id,
                seed=args.seed,
                reveal_order=args.reveal_order,
            )
            batches = make_bucket_batches(
                epoch_steps,
                batch_size=args.batch_size,
                bucket_width=args.bucket_width,
                seed=args.seed + epoch,
            )
            loader = DataLoader(
                StepDataset(epoch_steps),
                batch_sampler=batches,
                num_workers=args.num_workers,
                collate_fn=lambda batch: collate_steps(batch, target_pad_id=model.pad_id, condition_pad_id=model.base_pad_id),
                pin_memory=device.type == "cuda",
            )
            model.train()
            totals: dict[str, float] = {}
            total_count = 0.0
            last_lr = args.min_learning_rate
            for batch in loader:
                batch = move_batch(batch, device)
                last_lr = scheduled_learning_rate(
                    step_index=max(global_step - schedule_start_step, 0),
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
            train_metrics = {key: value / max(total_count, 1.0) for key, value in totals.items()}
            record: dict[str, Any] = {"epoch": epoch, "learning_rate": last_lr, "train": train_metrics, "batch_count": len(batches)}
            if validation_rows and args.validation_every > 0 and (epoch % args.validation_every == 0 or epoch == args.epochs):
                validation_steps = build_epoch_steps(
                    validation_rows,
                    epoch_index=epoch - 1,
                    mask_id=model.mask_id,
                    seed=args.seed + 7_000_001,
                    reveal_order=args.reveal_order,
                )
                record["validation"] = evaluate(
                    model,
                    validation_steps,
                    batch_size=validation_batch_size,
                    bucket_width=args.bucket_width,
                    device=device,
                    model_pad_id=model.pad_id,
                    condition_pad_id=model.base_pad_id,
                )
            if validation_rows and args.feasibility_every > 0 and (epoch % args.feasibility_every == 0 or epoch == args.epochs):
                record["validation_feasibility"] = evaluate_feasibility(
                    model,
                    validation_rows,
                    epoch_index=epoch - 1,
                    count=args.feasibility_count,
                    temperature=args.feasibility_temperature,
                    seed=args.seed + 8_000_003,
                    device=device,
                    batch_size=args.feasibility_batch_size,
                )
            if epoch % args.sample_every == 0 or epoch == args.epochs:
                record["motif_sample"] = {
                    str(length): sample_motif_sequences(
                        model,
                        length=length,
                        count=args.sample_count,
                        temperature=args.sample_temperature,
                        seed=args.seed + epoch * 1000 + length,
                        device=device,
                    )
                    for length in (80, 120, 160)
                    if length <= max_len
                }
            history.append(record)
            if args.checkpoint_every > 0 and (epoch % args.checkpoint_every == 0 or epoch == args.epochs):
                checkpoint_path = save_checkpoint(
                    output_dir,
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    history=history,
                    args=args,
                    config=config,
                )
                log(f"checkpoint epoch={epoch} path={checkpoint_path}")
            motif_msg = ""
            if "motif_sample" in record:
                motif_msg = " " + " ".join(
                    f"motif_len_{length}_distinct={metrics['distinct_fraction']:.3f}"
                    for length, metrics in record["motif_sample"].items()
                )
            validation_msg = ""
            if "validation" in record:
                validation = record["validation"]
                validation_msg = (
                    f" val_loss={validation['loss']:.4f}"
                    f" val_pos_acc={validation['position_accuracy']:.3f}"
                    f" val_tok_acc={validation['token_accuracy']:.3f}"
                )
            feasibility_msg = ""
            if "validation_feasibility" in record:
                feasibility = record["validation_feasibility"]
                feasibility_msg = (
                    f" val_feasible={feasibility['feasibility_rate']:.3f}"
                    f"({int(feasibility['feasible_count'])}/{int(feasibility['count'])})"
                )
            log(
                f"epoch={epoch} batches={len(batches)} train_loss={train_metrics['loss']:.4f} "
                f"pos_acc={train_metrics['position_accuracy']:.3f} tok_acc={train_metrics['token_accuracy']:.3f} "
                f"lr={last_lr:.2e}{validation_msg}{feasibility_msg}{motif_msg}"
            )
        torch.save(model.state_dict(), output_dir / "model_final.pt")
        final_checkpoint = save_checkpoint(
            output_dir,
            epoch=args.epochs,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            history=history,
            args=args,
            config=config,
        )
        summary = {
            "args": json_sanitize(vars(args)),
            "source_counts": source_counts(rows),
            "length_summary": length_summary(rows),
            "validation_source_counts": source_counts(validation_rows) if validation_rows else {},
            "validation_length_summary": length_summary(validation_rows) if validation_rows else {},
            "condition_tokens": list(CONDITION_TOKENS),
            "target_base_tokens": list(BASES),
            "config": asdict(config),
            "history": json_sanitize(history),
            "elapsed_sec": time.time() - start,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(f"wrote summary path={output_dir / 'summary.json'}")
        log(f"wrote final checkpoint path={final_checkpoint}")


if __name__ == "__main__":
    main()
