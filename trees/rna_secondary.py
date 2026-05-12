from __future__ import annotations

import math
import random
from dataclasses import dataclass


BASES: tuple[str, ...] = ("A", "U", "C", "G")
STRUCT_TOKENS: tuple[str, ...] = ("x", "(", ")", "[", "]", "{", "}", "<", ">")

BASE_TO_ID = {base: index for index, base in enumerate(BASES)}
ID_TO_BASE = {index: base for base, index in BASE_TO_ID.items()}
STRUCT_TO_ID = {token: index for index, token in enumerate(STRUCT_TOKENS)}
ID_TO_STRUCT = {index: token for token, index in STRUCT_TO_ID.items()}
SIDE_TOKENS: tuple[str, ...] = ("x", "L", "R")
SIDE_TO_ID = {token: index for index, token in enumerate(SIDE_TOKENS)}
ID_TO_SIDE = {index: token for token, index in SIDE_TO_ID.items()}

OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}", "<": ">"}
CLOSE_TO_OPEN = {close: open_ for open_, close in OPEN_TO_CLOSE.items()}
PAIR_TO_BRACKETS = {
    ("A", "U"): ("(", ")"),
    ("U", "A"): ("[", "]"),
    ("C", "G"): ("{", "}"),
    ("G", "C"): ("<", ">"),
}
OPEN_BY_BASE = {"A": "(", "U": "[", "C": "{", "G": "<"}
CLOSE_BY_BASE = {"A": "]", "U": ")", "C": ">", "G": "}"}
GAS_CONSTANT_KCAL_PER_MOL_K = 0.00198720425864083


@dataclass(frozen=True)
class RnaStructureExample:
    bases: tuple[str, ...]
    structure: tuple[str, ...]

    @property
    def length(self) -> int:
        return len(self.bases)

    @property
    def pair_count(self) -> int:
        return sum(token in OPEN_TO_CLOSE for token in self.structure)


def bases_to_ids(bases: tuple[str, ...] | list[str] | str) -> tuple[int, ...]:
    return tuple(BASE_TO_ID[base] for base in bases)


def structure_to_ids(structure: tuple[str, ...] | list[str] | str) -> tuple[int, ...]:
    return tuple(STRUCT_TO_ID[token] for token in structure)


def ids_to_structure(ids: tuple[int, ...] | list[int]) -> tuple[str, ...]:
    return tuple(ID_TO_STRUCT[index] for index in ids)


def side_structure_to_ids(structure: tuple[str, ...] | list[str] | str) -> tuple[int, ...]:
    return tuple(SIDE_TO_ID[token] for token in structure)


def ids_to_side_structure(ids: tuple[int, ...] | list[int]) -> tuple[str, ...]:
    return tuple(ID_TO_SIDE[index] for index in ids)


def structure_to_side(structure: tuple[str, ...] | list[str] | str) -> tuple[str, ...]:
    side: list[str] = []
    for token in structure:
        if token == "x":
            side.append("x")
        elif token in OPEN_TO_CLOSE:
            side.append("L")
        elif token in CLOSE_TO_OPEN:
            side.append("R")
        else:
            raise ValueError(f"unknown structure token: {token}")
    return tuple(side)


def random_rna_sequence(length: int, rng: random.Random) -> tuple[str, ...]:
    return tuple(rng.choice(BASES) for _ in range(length))


def allowed_structure_tokens(base: str) -> tuple[str, ...]:
    return ("x", OPEN_BY_BASE[base], CLOSE_BY_BASE[base])


def allowed_structure_ids(base_id: int) -> tuple[int, ...]:
    return tuple(STRUCT_TO_ID[token] for token in allowed_structure_tokens(ID_TO_BASE[base_id]))


def are_complementary(left_base: str, right_base: str) -> bool:
    return (left_base, right_base) in PAIR_TO_BRACKETS


def pair_is_allowed(
    bases: tuple[str, ...] | list[str] | str,
    left: int,
    right: int,
    *,
    min_loop_unpaired: int = 0,
) -> bool:
    return right - left - 1 >= min_loop_unpaired and are_complementary(bases[left], bases[right])


def pair_family(left_base: str, right_base: str) -> str:
    if (left_base, right_base) in (("A", "U"), ("U", "A")):
        return "AU"
    if (left_base, right_base) in (("C", "G"), ("G", "C")):
        return "CG"
    raise ValueError(f"not a supported complementary pair: {left_base}{right_base}")


def pair_log_bonus(
    left_base: str,
    right_base: str,
    *,
    pair_log_weight: float = 0.0,
    au_pair_log_weight: float | None = None,
    cg_pair_log_weight: float | None = None,
) -> float:
    family = pair_family(left_base, right_base)
    if family == "AU":
        return pair_log_weight if au_pair_log_weight is None else au_pair_log_weight
    if family == "CG":
        return pair_log_weight if cg_pair_log_weight is None else cg_pair_log_weight
    raise AssertionError("unreachable pair family")


def pair_log_weights_from_kcal_bonuses(
    *,
    au_pair_bonus_kcal: float,
    cg_pair_bonus_kcal: float,
    temperature_celsius: float = 37.0,
) -> tuple[float, float]:
    temperature_kelvin = 273.15 + temperature_celsius
    beta = 1.0 / (GAS_CONSTANT_KCAL_PER_MOL_K * temperature_kelvin)
    return beta * au_pair_bonus_kcal, beta * cg_pair_bonus_kcal


def _logsumexp(values: list[float]) -> float:
    if not values:
        return -math.inf
    max_value = max(values)
    if max_value == -math.inf:
        return -math.inf
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def _sample_log_choice(log_weights: list[float], rng: random.Random) -> int:
    total = _logsumexp(log_weights)
    threshold = rng.random()
    cumulative = 0.0
    for index, log_weight in enumerate(log_weights):
        cumulative += math.exp(log_weight - total)
        if threshold <= cumulative:
            return index
    return len(log_weights) - 1


def compute_log_partition(
    bases: tuple[str, ...] | list[str] | str,
    *,
    pair_log_weight: float = 0.0,
    au_pair_log_weight: float | None = None,
    cg_pair_log_weight: float | None = None,
    min_loop_unpaired: int = 0,
) -> list[list[float]]:
    base_tuple = tuple(bases)
    length = len(base_tuple)
    log_z = [[-math.inf for _ in range(length + 1)] for _ in range(length + 1)]
    for i in range(length + 1):
        log_z[i][i] = 0.0
    for span in range(1, length + 1):
        for i in range(0, length - span + 1):
            j = i + span
            choices = [log_z[i + 1][j]]
            for k in range(i + 1, j):
                if pair_is_allowed(base_tuple, i, k, min_loop_unpaired=min_loop_unpaired):
                    choices.append(
                        pair_log_bonus(
                            base_tuple[i],
                            base_tuple[k],
                            pair_log_weight=pair_log_weight,
                            au_pair_log_weight=au_pair_log_weight,
                            cg_pair_log_weight=cg_pair_log_weight,
                        )
                        + log_z[i + 1][k]
                        + log_z[k + 1][j]
                    )
            log_z[i][j] = _logsumexp(choices)
    return log_z


def sample_secondary_structure(
    bases: tuple[str, ...] | list[str] | str,
    *,
    pair_log_weight: float = 0.0,
    au_pair_log_weight: float | None = None,
    cg_pair_log_weight: float | None = None,
    min_loop_unpaired: int = 0,
    rng: random.Random,
    log_z: list[list[float]] | None = None,
) -> tuple[str, ...]:
    base_tuple = tuple(bases)
    length = len(base_tuple)
    if log_z is None:
        log_z = compute_log_partition(
            base_tuple,
            pair_log_weight=pair_log_weight,
            au_pair_log_weight=au_pair_log_weight,
            cg_pair_log_weight=cg_pair_log_weight,
            min_loop_unpaired=min_loop_unpaired,
        )
    structure = ["?"] * length

    def sample_interval(i: int, j: int) -> None:
        if i >= j:
            return
        labels: list[tuple[str, int | None]] = [("unpaired", None)]
        log_weights = [log_z[i + 1][j]]
        for k in range(i + 1, j):
            if pair_is_allowed(base_tuple, i, k, min_loop_unpaired=min_loop_unpaired):
                labels.append(("pair", k))
                log_weights.append(
                    pair_log_bonus(
                        base_tuple[i],
                        base_tuple[k],
                        pair_log_weight=pair_log_weight,
                        au_pair_log_weight=au_pair_log_weight,
                        cg_pair_log_weight=cg_pair_log_weight,
                    )
                    + log_z[i + 1][k]
                    + log_z[k + 1][j]
                )
        choice = labels[_sample_log_choice(log_weights, rng)]
        if choice[0] == "unpaired":
            structure[i] = "x"
            sample_interval(i + 1, j)
            return
        k = choice[1]
        if k is None:
            raise AssertionError("pair choice missing partner")
        open_symbol, close_symbol = PAIR_TO_BRACKETS[(base_tuple[i], base_tuple[k])]
        structure[i] = open_symbol
        structure[k] = close_symbol
        sample_interval(i + 1, k)
        sample_interval(k + 1, j)

    sample_interval(0, length)
    if any(token == "?" for token in structure):
        raise AssertionError("incomplete sampled structure")
    return tuple(structure)


def compute_mfe_structure(
    bases: tuple[str, ...] | list[str] | str,
    *,
    pair_log_weight: float = 0.0,
    au_pair_log_weight: float | None = None,
    cg_pair_log_weight: float | None = None,
    min_loop_unpaired: int = 0,
) -> tuple[str, ...]:
    base_tuple = tuple(bases)
    length = len(base_tuple)
    score = [[0.0 for _ in range(length + 1)] for _ in range(length + 1)]
    choice: list[list[tuple[str, int | None]]] = [
        [("empty", None) for _ in range(length + 1)] for _ in range(length + 1)
    ]
    for span in range(1, length + 1):
        for i in range(0, length - span + 1):
            j = i + span
            best_score = score[i + 1][j]
            best_choice: tuple[str, int | None] = ("unpaired", None)
            for k in range(i + 1, j):
                if not pair_is_allowed(base_tuple, i, k, min_loop_unpaired=min_loop_unpaired):
                    continue
                candidate = (
                    pair_log_bonus(
                        base_tuple[i],
                        base_tuple[k],
                        pair_log_weight=pair_log_weight,
                        au_pair_log_weight=au_pair_log_weight,
                        cg_pair_log_weight=cg_pair_log_weight,
                    )
                    + score[i + 1][k]
                    + score[k + 1][j]
                )
                if candidate > best_score + 1.0e-12:
                    best_score = candidate
                    best_choice = ("pair", k)
            score[i][j] = best_score
            choice[i][j] = best_choice
    structure = ["x"] * length

    def backtrace(i: int, j: int) -> None:
        if i >= j:
            return
        action, k = choice[i][j]
        if action == "unpaired":
            structure[i] = "x"
            backtrace(i + 1, j)
            return
        if action == "pair":
            if k is None:
                raise AssertionError("pair choice missing partner")
            open_symbol, close_symbol = PAIR_TO_BRACKETS[(base_tuple[i], base_tuple[k])]
            structure[i] = open_symbol
            structure[k] = close_symbol
            backtrace(i + 1, k)
            backtrace(k + 1, j)
            return
        raise AssertionError(f"unknown MFE action {action}")

    backtrace(0, length)
    return tuple(structure)


def validate_secondary_structure(
    bases: tuple[str, ...] | list[str] | str,
    structure: tuple[str, ...] | list[str] | str,
    *,
    min_loop_unpaired: int = 0,
) -> bool:
    base_tuple = tuple(bases)
    structure_tuple = tuple(structure)
    if len(base_tuple) != len(structure_tuple):
        return False
    stack: list[tuple[int, str, str]] = []
    for index, (base, token) in enumerate(zip(base_tuple, structure_tuple)):
        if token not in allowed_structure_tokens(base):
            return False
        if token == "x":
            continue
        if token in OPEN_TO_CLOSE:
            stack.append((index, base, OPEN_TO_CLOSE[token]))
            continue
        if token in CLOSE_TO_OPEN:
            if not stack:
                return False
            left_index, left_base, expected_close = stack.pop()
            if token != expected_close:
                return False
            if not pair_is_allowed(base_tuple, left_index, index, min_loop_unpaired=min_loop_unpaired):
                return False
            continue
        return False
    return not stack


def matching_pairs(structure: tuple[str, ...] | list[str] | str) -> dict[int, int]:
    stack: list[tuple[str, int]] = []
    pairs: dict[int, int] = {}
    for index, token in enumerate(structure):
        if token in OPEN_TO_CLOSE:
            stack.append((token, index))
        elif token in CLOSE_TO_OPEN:
            if not stack:
                raise ValueError("invalid structure: close without open")
            open_symbol, open_index = stack.pop()
            if OPEN_TO_CLOSE[open_symbol] != token:
                raise ValueError("invalid structure: mismatched bracket")
            pairs[open_index] = index
            pairs[index] = open_index
        elif token != "x":
            raise ValueError(f"unknown structure token: {token}")
    if stack:
        raise ValueError("invalid structure: unclosed bracket")
    return pairs


def side_matching_pairs(structure: tuple[str, ...] | list[str] | str) -> dict[int, int]:
    stack: list[int] = []
    pairs: dict[int, int] = {}
    for index, token in enumerate(structure):
        if token == "x":
            continue
        if token == "L":
            stack.append(index)
            continue
        if token == "R":
            if not stack:
                raise ValueError("invalid side structure: close without open")
            open_index = stack.pop()
            pairs[open_index] = index
            pairs[index] = open_index
            continue
        raise ValueError(f"unknown side token: {token}")
    if stack:
        raise ValueError("invalid side structure: unclosed open")
    return pairs


def validate_side_structure(
    bases: tuple[str, ...] | list[str] | str,
    structure: tuple[str, ...] | list[str] | str,
    *,
    min_loop_unpaired: int = 0,
) -> bool:
    base_tuple = tuple(bases)
    structure_tuple = tuple(structure)
    if len(base_tuple) != len(structure_tuple):
        return False
    try:
        pairs = side_matching_pairs(structure_tuple)
    except ValueError:
        return False
    for left, right in pairs.items():
        if left < right and not pair_is_allowed(base_tuple, left, right, min_loop_unpaired=min_loop_unpaired):
            return False
    return True


def parse_based_position_order(structure: tuple[str, ...] | list[str] | str) -> list[int]:
    structure_tuple = tuple(structure)
    pairs = matching_pairs(structure_tuple)
    order: list[int] = []

    def visit(i: int, j: int) -> None:
        if i >= j:
            return
        token = structure_tuple[i]
        if token == "x":
            order.append(i)
            visit(i + 1, j)
            return
        if token in OPEN_TO_CLOSE:
            k = pairs[i]
            if not (i < k < j):
                raise ValueError("pair crosses interval boundary")
            order.extend([i, k])
            visit(i + 1, k)
            visit(k + 1, j)
            return
        raise ValueError("parse visit landed on a close bracket")

    visit(0, len(structure_tuple))
    if sorted(order) != list(range(len(structure_tuple))):
        raise AssertionError("parse order is not a permutation")
    return order


def parse_based_position_order_side(structure: tuple[str, ...] | list[str] | str) -> list[int]:
    structure_tuple = tuple(structure)
    pairs = side_matching_pairs(structure_tuple)
    order: list[int] = []

    def visit(i: int, j: int) -> None:
        if i >= j:
            return
        token = structure_tuple[i]
        if token == "x":
            order.append(i)
            visit(i + 1, j)
            return
        if token == "L":
            k = pairs[i]
            if not (i < k < j):
                raise ValueError("pair crosses interval boundary")
            order.extend([i, k])
            visit(i + 1, k)
            visit(k + 1, j)
            return
        raise ValueError("parse visit landed on a right-pair token")

    visit(0, len(structure_tuple))
    if sorted(order) != list(range(len(structure_tuple))):
        raise AssertionError("side parse order is not a permutation")
    return order


def structure_log_score(
    bases: tuple[str, ...] | list[str] | str,
    structure: tuple[str, ...] | list[str] | str,
    *,
    pair_log_weight: float = 0.0,
    au_pair_log_weight: float | None = None,
    cg_pair_log_weight: float | None = None,
    min_loop_unpaired: int = 0,
) -> float:
    base_tuple = tuple(bases)
    structure_tuple = tuple(structure)
    try:
        if all(token in SIDE_TO_ID for token in structure_tuple):
            pairs = side_matching_pairs(structure_tuple)
        else:
            pairs = matching_pairs(structure_tuple)
    except ValueError:
        return -math.inf
    score = 0.0
    for left, right in sorted((i, j) for i, j in pairs.items() if i < j):
        if not pair_is_allowed(base_tuple, left, right, min_loop_unpaired=min_loop_unpaired):
            return -math.inf
        score += pair_log_bonus(
            base_tuple[left],
            base_tuple[right],
            pair_log_weight=pair_log_weight,
            au_pair_log_weight=au_pair_log_weight,
            cg_pair_log_weight=cg_pair_log_weight,
        )
    return score


def structure_nll(
    bases: tuple[str, ...] | list[str] | str,
    structure: tuple[str, ...] | list[str] | str,
    *,
    pair_log_weight: float = 0.0,
    au_pair_log_weight: float | None = None,
    cg_pair_log_weight: float | None = None,
    min_loop_unpaired: int = 0,
    log_z: list[list[float]] | None = None,
) -> float:
    base_tuple = tuple(bases)
    if log_z is None:
        log_z = compute_log_partition(
            base_tuple,
            pair_log_weight=pair_log_weight,
            au_pair_log_weight=au_pair_log_weight,
            cg_pair_log_weight=cg_pair_log_weight,
            min_loop_unpaired=min_loop_unpaired,
        )
    score = structure_log_score(
        base_tuple,
        structure,
        pair_log_weight=pair_log_weight,
        au_pair_log_weight=au_pair_log_weight,
        cg_pair_log_weight=cg_pair_log_weight,
        min_loop_unpaired=min_loop_unpaired,
    )
    if score == -math.inf:
        return math.inf
    return log_z[0][len(base_tuple)] - score


def exact_structure_entropy(
    bases: tuple[str, ...] | list[str] | str,
    *,
    pair_log_weight: float = 0.0,
    au_pair_log_weight: float | None = None,
    cg_pair_log_weight: float | None = None,
    min_loop_unpaired: int = 0,
    log_z: list[list[float]] | None = None,
) -> float:
    base_tuple = tuple(bases)
    length = len(base_tuple)
    if log_z is None:
        log_z = compute_log_partition(
            base_tuple,
            pair_log_weight=pair_log_weight,
            au_pair_log_weight=au_pair_log_weight,
            cg_pair_log_weight=cg_pair_log_weight,
            min_loop_unpaired=min_loop_unpaired,
        )
    expected_score = [[0.0 for _ in range(length + 1)] for _ in range(length + 1)]
    for span in range(1, length + 1):
        for i in range(0, length - span + 1):
            j = i + span
            total = log_z[i][j]
            choices: list[tuple[float, float]] = [(log_z[i + 1][j], expected_score[i + 1][j])]
            for k in range(i + 1, j):
                if not pair_is_allowed(base_tuple, i, k, min_loop_unpaired=min_loop_unpaired):
                    continue
                bonus = pair_log_bonus(
                    base_tuple[i],
                    base_tuple[k],
                    pair_log_weight=pair_log_weight,
                    au_pair_log_weight=au_pair_log_weight,
                    cg_pair_log_weight=cg_pair_log_weight,
                )
                choices.append(
                    (
                        bonus + log_z[i + 1][k] + log_z[k + 1][j],
                        bonus + expected_score[i + 1][k] + expected_score[k + 1][j],
                    )
                )
            expected_score[i][j] = sum(math.exp(log_weight - total) * score for log_weight, score in choices)
    return log_z[0][length] - expected_score[0][length]


def parse_frontier_random_position_order(
    structure: tuple[str, ...] | list[str] | str,
    rng: random.Random,
) -> list[int]:
    structure_tuple = tuple(structure)
    pairs = matching_pairs(structure_tuple)
    order: list[int] = []
    frontier: list[tuple[int, int]] = [(0, len(structure_tuple))]

    while frontier:
        span_index = rng.randrange(len(frontier))
        i, j = frontier.pop(span_index)
        if i >= j:
            continue
        token = structure_tuple[i]
        if token == "x":
            order.append(i)
            if i + 1 < j:
                frontier.append((i + 1, j))
            continue
        if token in OPEN_TO_CLOSE:
            k = pairs[i]
            if not (i < k < j):
                raise ValueError("pair crosses interval boundary")
            order.extend([i, k])
            if i + 1 < k:
                frontier.append((i + 1, k))
            if k + 1 < j:
                frontier.append((k + 1, j))
            continue
        raise ValueError("parse visit landed on a close bracket")

    if sorted(order) != list(range(len(structure_tuple))):
        raise AssertionError("parse frontier-random order is not a permutation")
    return order


def random_position_order(length: int, rng: random.Random) -> list[int]:
    order = list(range(length))
    rng.shuffle(order)
    return order


def generate_fixed_sequence_dataset(
    *,
    bases: tuple[str, ...],
    count: int,
    seed: int,
    pair_log_weight: float,
    au_pair_log_weight: float | None = None,
    cg_pair_log_weight: float | None = None,
    min_loop_unpaired: int = 0,
) -> list[RnaStructureExample]:
    rng = random.Random(seed)
    log_z = compute_log_partition(
        bases,
        pair_log_weight=pair_log_weight,
        au_pair_log_weight=au_pair_log_weight,
        cg_pair_log_weight=cg_pair_log_weight,
        min_loop_unpaired=min_loop_unpaired,
    )
    return [
        RnaStructureExample(
            bases=bases,
            structure=sample_secondary_structure(
                bases,
                pair_log_weight=pair_log_weight,
                au_pair_log_weight=au_pair_log_weight,
                cg_pair_log_weight=cg_pair_log_weight,
                min_loop_unpaired=min_loop_unpaired,
                rng=rng,
                log_z=log_z,
            ),
        )
        for _ in range(count)
    ]


def generate_conditional_dataset(
    *,
    count: int,
    length: int,
    seed: int,
    pair_log_weight: float,
    au_pair_log_weight: float | None = None,
    cg_pair_log_weight: float | None = None,
    min_loop_unpaired: int = 0,
) -> list[RnaStructureExample]:
    rng = random.Random(seed)
    examples: list[RnaStructureExample] = []
    for _ in range(count):
        bases = random_rna_sequence(length, rng)
        structure = sample_secondary_structure(
            bases,
            pair_log_weight=pair_log_weight,
            au_pair_log_weight=au_pair_log_weight,
            cg_pair_log_weight=cg_pair_log_weight,
            min_loop_unpaired=min_loop_unpaired,
            rng=rng,
        )
        examples.append(RnaStructureExample(bases=bases, structure=structure))
    return examples


def structure_to_text(structure: tuple[str, ...] | list[str]) -> str:
    return "".join(structure)


def bases_to_text(bases: tuple[str, ...] | list[str]) -> str:
    return "".join(bases)
