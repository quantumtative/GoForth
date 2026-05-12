#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
TREES = ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import RNA  # noqa: E402
import torch  # noqa: E402

from trees.rnaplot_design_highlight import highlight_svg_bases  # noqa: E402
from trees.rna_secondary import BASES  # noqa: E402
from trees.run_matrixmodel_seq2seq_ar_designer_torch import (  # noqa: E402
    BASE_CONDITION_TOKENS,
    PAIRED_UNKNOWN_TOKEN,
    RnaSeq2SeqARTransformer,
    Seq2SeqConfig,
    STRUCTURE_TOKENS_WITH_PAIRED_UNKNOWN,
    UNKNOWN_SIDE_TOKEN,
    encode_full_condition,
)


RT_37C = 0.00198717 * (37.0 + 273.15)
NEG_INF = -1.0e9
VIENNA_ALLOWED_BASE_PAIRS = {
    ("A", "U"),
    ("U", "A"),
    ("G", "C"),
    ("C", "G"),
    ("G", "U"),
    ("U", "G"),
}
VIENNA_DB_CONSTRAINT_FLAGS = RNA.CONSTRAINT_DB_DEFAULT | RNA.CONSTRAINT_DB_ENFORCE_BP
VIENNA_ENFORCED_CONTEXT = RNA.CONSTRAINT_CONTEXT_ALL_LOOPS | RNA.CONSTRAINT_CONTEXT_ENFORCE
SIDE_TO_DOT = str.maketrans({"x": ".", "L": "(", "R": ")"})
DOT_TO_SIDE = str.maketrans({".": "x", "(": "L", ")": "R"})
DOT_TO_PARTIAL_SIDE = str.maketrans({".": "x", "(": "L", ")": "R", "?": "?", "#": "#"})
SIDE_TOKEN_TO_DOT = {"x": ".", "L": "(", "R": ")", UNKNOWN_SIDE_TOKEN: ".", PAIRED_UNKNOWN_TOKEN: "."}
PARTIAL_SIDE_TOKENS = {"x", "L", "R", UNKNOWN_SIDE_TOKEN, PAIRED_UNKNOWN_TOKEN}
BASE_TO_ID = {base: index for index, base in enumerate(BASES)}
ID_TO_BASE = {index: base for base, index in BASE_TO_ID.items()}
IUPAC: dict[str, set[str]] = {
    "A": {"A"},
    "C": {"C"},
    "G": {"G"},
    "U": {"U"},
    "T": {"U"},
    "R": {"A", "G"},
    "Y": {"C", "U"},
    "S": {"G", "C"},
    "W": {"A", "U"},
    "K": {"G", "U"},
    "M": {"A", "C"},
    "B": {"C", "G", "U"},
    "D": {"A", "G", "U"},
    "H": {"A", "C", "U"},
    "V": {"A", "C", "G"},
    "N": set(BASES),
    "X": set(BASES),
    "?": set(BASES),
    "#": set(BASES),
    ".": set(BASES),
    "-": set(BASES),
}
UNSPECIFIED_BASE_TOKENS = {"N", "X", "?", "#", ".", "-"}
SIDE_HIGHLIGHT_ROLE = {
    "L": "left",
    "R": "right",
    "x": "unpaired",
    PAIRED_UNKNOWN_TOKEN: "paired",
    UNKNOWN_SIDE_TOKEN: "null",
}


def now_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def fresh_seed() -> int:
    return random.SystemRandom().randrange(1, 2_147_483_647)


def payload_seed(payload: dict[str, Any]) -> int:
    raw = payload.get("seed")
    if raw is None or str(raw).strip() == "":
        return fresh_seed()
    return int(raw)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:80] or "run"


def side_to_dot(side: str) -> str:
    return side.translate(SIDE_TO_DOT)


def dot_to_side(dot: str) -> str:
    return dot.translate(DOT_TO_SIDE)


def condition_to_dot(side: str) -> str:
    return "".join(SIDE_TOKEN_TO_DOT.get(token, ".") for token in side)


def condition_to_render_dot(side: str) -> str:
    partners = specified_partners_from_condition(side)
    chars = ["."] * len(side)
    for left, right in enumerate(partners):
        if right > left:
            chars[left] = "("
            chars[right] = ")"
    return "".join(chars)


def condition_to_input_text(side: str) -> str:
    token_to_text = {
        "x": ".",
        "L": "(",
        "R": ")",
        UNKNOWN_SIDE_TOKEN: UNKNOWN_SIDE_TOKEN,
        PAIRED_UNKNOWN_TOKEN: PAIRED_UNKNOWN_TOKEN,
    }
    return "".join(token_to_text.get(token, ".") for token in side)


def canonical_partial_side(compact: str) -> str | None:
    tokens: list[str] = []
    for char in compact:
        if char in {"x", "X"}:
            tokens.append("x")
        elif char in {"L", "l"}:
            tokens.append("L")
        elif char in {"R", "r"}:
            tokens.append("R")
        elif char in {UNKNOWN_SIDE_TOKEN, PAIRED_UNKNOWN_TOKEN}:
            tokens.append(char)
        else:
            return None
    return "".join(tokens)


def specified_partners_from_condition(side: str) -> tuple[int, ...]:
    stack: list[int] = []
    partners = [-1] * len(side)
    for index, token in enumerate(side):
        if token == "L":
            stack.append(index)
        elif token == "R":
            if not stack:
                continue
            left = stack.pop()
            partners[left] = index
            partners[index] = left
        elif token in {"x", UNKNOWN_SIDE_TOKEN, PAIRED_UNKNOWN_TOKEN}:
            pass
        else:
            raise ValueError(f"bad structure token {token!r}")
    return tuple(partners)


def normalize_structure_condition(raw: str) -> tuple[str, str, bool]:
    compact = re.sub(r"\s+", "", raw)
    if not compact:
        raise ValueError("structure is empty")
    chars = set(compact)
    if chars <= set(".()?#"):
        side = compact.translate(DOT_TO_PARTIAL_SIDE)
    else:
        side = canonical_partial_side(compact) or ""
    if not side or any(token not in PARTIAL_SIDE_TOKENS for token in side):
        raise ValueError("structure must use dot-bracket .() plus ?/#, or side tokens x/L/R/?/#")
    is_full = all(token in {"x", "L", "R"} for token in side)
    if is_full:
        partners_from_side(side)
    else:
        specified_partners_from_condition(side)
    return condition_to_render_dot(side), side, is_full


def normalize_structure(raw: str) -> tuple[str, str]:
    dot, side, is_full = normalize_structure_condition(raw)
    if not is_full:
        raise ValueError("partial structure tokens ?/# require a partial-condition FSB checkpoint")
    return dot, side


def partners_from_side(side: str) -> tuple[int, ...]:
    stack: list[int] = []
    partners = [-1] * len(side)
    for index, token in enumerate(side):
        if token == "L":
            stack.append(index)
        elif token == "R":
            if not stack:
                raise ValueError(f"unmatched right-pair token at position {index + 1}")
            left = stack.pop()
            partners[left] = index
            partners[index] = left
        elif token == "x":
            pass
        else:
            raise ValueError(f"bad side token {token!r}")
    if stack:
        raise ValueError(f"unmatched left-pair token at position {stack[-1] + 1}")
    return tuple(partners)


def normalize_base_mask(raw: str, length: int) -> tuple[list[set[str]], str]:
    compact = re.sub(r"\s+", "", raw).upper().replace("T", "U")
    if not compact:
        compact = "N" * length
    if len(compact) != length:
        raise ValueError(f"base mask length {len(compact)} does not match structure length {length}")
    allowed: list[set[str]] = []
    canonical = []
    for index, char in enumerate(compact):
        if char not in IUPAC:
            raise ValueError(f"bad base mask token {char!r} at position {index + 1}")
        values = set(IUPAC[char])
        allowed.append(values)
        canonical.append(char)
    return allowed, "".join(canonical)


def display_sequence_from_mask(mask_text: str) -> str:
    return "".join("." if char in UNSPECIFIED_BASE_TOKENS else char for char in mask_text)


def target_label_overrides_from_mask(mask_text: str) -> dict[int, str]:
    return {index: ("" if char in UNSPECIFIED_BASE_TOKENS else char) for index, char in enumerate(mask_text)}


def target_layout_sequence(mask_text: str) -> str:
    return "".join(char if char in BASES else "A" for char in mask_text)


def base_constraint_indices(mask_text: str) -> set[int]:
    return {index for index, char in enumerate(mask_text) if char not in UNSPECIFIED_BASE_TOKENS}


def target_highlights_from_side(side: str) -> dict[int, str]:
    return {index: SIDE_HIGHLIGHT_ROLE[token] for index, token in enumerate(side)}


def tighten_pair_compatible_masks(allowed: list[set[str]], partners: tuple[int, ...]) -> list[set[str]]:
    tightened = [set(values) for values in allowed]
    changed = True
    while changed:
        changed = False
        for left, right in enumerate(partners):
            if right <= left:
                continue
            left_allowed = {
                base
                for base in tightened[left]
                if any((base, other) in VIENNA_ALLOWED_BASE_PAIRS for other in tightened[right])
            }
            right_allowed = {
                base
                for base in tightened[right]
                if any((other, base) in VIENNA_ALLOWED_BASE_PAIRS for other in tightened[left])
            }
            if not left_allowed or not right_allowed:
                raise ValueError(f"base mask is incompatible with pair {left + 1}-{right + 1}")
            if left_allowed != tightened[left]:
                tightened[left] = left_allowed
                changed = True
            if right_allowed != tightened[right]:
                tightened[right] = right_allowed
                changed = True
    return tightened


def exact_partners_for_condition(side: str) -> tuple[int, ...]:
    if any(token in {UNKNOWN_SIDE_TOKEN, PAIRED_UNKNOWN_TOKEN} for token in side):
        return tuple([-1] * len(side))
    return partners_from_side(side)


def sequence_satisfies_mask(sequence: str, allowed: list[set[str]]) -> bool:
    return len(sequence) == len(allowed) and all(base in allowed[index] for index, base in enumerate(sequence))


def compatible(sequence: str, side: str) -> bool:
    partners = partners_from_side(side)
    return all(
        (sequence[left], sequence[right]) in VIENNA_ALLOWED_BASE_PAIRS
        for left, right in enumerate(partners)
        if left < right
    )


def base_condition_tokens(mask_text: str) -> tuple[str, ...]:
    return tuple(char if char in BASES else UNKNOWN_SIDE_TOKEN for char in mask_text)


def encode_workbench_condition(condition_side: str, mask_text: str, conditioning_mode: str) -> tuple[int, ...]:
    has_partial_structure = any(token in {UNKNOWN_SIDE_TOKEN, PAIRED_UNKNOWN_TOKEN} for token in condition_side)
    if conditioning_mode == "full_structure":
        if has_partial_structure:
            raise ValueError("selected checkpoint is full-structure only; choose an FSB checkpoint for ?/# targets")
        return encode_full_condition(condition_side, conditioning_mode=conditioning_mode)
    if conditioning_mode == "structure_with_partial_bases":
        structure_to_id = {token: index for index, token in enumerate(STRUCTURE_TOKENS_WITH_PAIRED_UNKNOWN)}
        base_to_id = {token: index for index, token in enumerate(BASE_CONDITION_TOKENS)}
        width = len(BASE_CONDITION_TOKENS)
        encoded: list[int] = []
        for structure_token, base_token in zip(condition_side, base_condition_tokens(mask_text), strict=True):
            encoded.append(structure_to_id[structure_token] * width + base_to_id[base_token])
        return tuple(encoded)
    if conditioning_mode in {
        "structure_with_paired_unknown",
        "structure_with_partial_bases_and_codons",
        "structure_with_partial_bases_and_codons_no_rho",
    }:
        return encode_full_condition(condition_side, conditioning_mode=conditioning_mode)
    raise ValueError(f"unsupported checkpoint conditioning_mode {conditioning_mode!r}")


def structure_condition_error_counts(condition_side: str, observed_side: str) -> tuple[int, int]:
    wrong = 0
    active = 0
    for expected, observed in zip(condition_side, observed_side, strict=True):
        if expected == UNKNOWN_SIDE_TOKEN:
            continue
        active += 1
        if expected == PAIRED_UNKNOWN_TOKEN:
            wrong += int(observed not in {"L", "R"})
        else:
            wrong += int(observed != expected)
    return wrong, active


def base_mask_error_counts(sequence: str, allowed: list[set[str]], mask_text: str) -> tuple[int, int]:
    wrong = 0
    active = 0
    for index, (base, allowed_bases, mask_char) in enumerate(zip(sequence, allowed, mask_text, strict=True)):
        if mask_char in UNSPECIFIED_BASE_TOKENS:
            continue
        active += 1
        wrong += int(base not in allowed_bases)
    return wrong, active


def add_partial_condition_constraints(fc: Any, condition_side: str) -> None:
    for position, token in enumerate(condition_side, start=1):
        if token == UNKNOWN_SIDE_TOKEN:
            continue
        if token == "x":
            fc.hc_add_up(position, VIENNA_ENFORCED_CONTEXT)
        elif token == PAIRED_UNKNOWN_TOKEN:
            fc.hc_add_bp_nonspecific(position, 0, VIENNA_ENFORCED_CONTEXT)
        elif token == "L":
            fc.hc_add_bp_nonspecific(position, 1, VIENNA_ENFORCED_CONTEXT)
        elif token == "R":
            fc.hc_add_bp_nonspecific(position, -1, VIENNA_ENFORCED_CONTEXT)
        else:
            raise ValueError(f"bad partial-structure token {token!r}")


def partial_constraint_pf_energy(sequence: str, condition_side: str) -> float:
    md = RNA.md()
    md.uniq_ML = 1
    fc = RNA.fold_compound(sequence, md)
    add_partial_condition_constraints(fc, condition_side)
    mfe_dot, mfe_energy = fc.mfe()
    if not str(mfe_dot):
        return math.inf
    fc.exp_params_rescale(mfe_energy)
    _pf_structure, pf_energy = fc.pf()
    return float(pf_energy)


def pair_error_rate(target_side: str, sampled_dot: str) -> float:
    target = partners_from_side(target_side)
    sampled = partners_from_side(dot_to_side(sampled_dot))
    return sum(1 for left, right in zip(target, sampled, strict=True) if left != right) / max(len(target), 1)


def collect_subopt_result(structure: str | None, energy: float, data: dict[str, Any]) -> None:
    if structure is not None:
        data["structures"].append(str(structure))


def mfe_structures(fc: Any) -> list[str]:
    data: dict[str, Any] = {"structures": []}
    fc.subopt_cb(0, collect_subopt_result, data)
    return sorted(set(data["structures"]))


def gc_fraction(sequence: str) -> float:
    return (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1)


def svg_fallback_text(sequence: str, dot: str, message: str) -> str:
    def esc(text: str) -> str:
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    seq_line = esc(sequence[:96] + ("..." if len(sequence) > 96 else ""))
    dot_line = esc(dot[:96] + ("..." if len(dot) > 96 else ""))
    msg_line = esc(message[:120])
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="720" height="150" viewBox="0 0 720 150">
  <rect width="720" height="150" fill="#f7f8fa"/>
  <text x="18" y="32" font-family="monospace" font-size="14" fill="#263449">RNA plot unavailable</text>
  <text x="18" y="62" font-family="monospace" font-size="12" fill="#5f6b7a">{msg_line}</text>
  <text x="18" y="96" font-family="monospace" font-size="12" fill="#263449">{seq_line}</text>
  <text x="18" y="122" font-family="monospace" font-size="12" fill="#263449">{dot_line}</text>
</svg>"""


def render_svg_text(
    sequence: str,
    dot: str,
    *,
    highlights: dict[int, str] | None = None,
    draw_null_bubbles: bool = False,
    base_constraints: set[int] | None = None,
    label_overrides: dict[int, str] | None = None,
) -> str:
    payload = {
        "root": str(ROOT),
        "sequence": sequence,
        "dot": dot,
        "highlights": highlights or {},
        "draw_null_bubbles": draw_null_bubbles,
        "base_constraints": sorted(base_constraints or set()),
        "label_overrides": label_overrides or {},
    }
    child = r"""
import json
import sys
import tempfile
from pathlib import Path

payload = json.load(sys.stdin)
root = payload["root"]
if root not in sys.path:
    sys.path.insert(0, root)

import RNA
from trees.rnaplot_design_highlight import highlight_svg_bases

try:
    with tempfile.TemporaryDirectory(prefix="rna_workbench_svg_child_") as tmp:
        path = Path(tmp) / "structure.svg"
        rc = RNA.plot_structure_svg(str(path), payload["sequence"], payload["dot"])
        if rc == 0:
            raise RuntimeError("ViennaRNA SVG plot failed")
        highlights = {int(index): role for index, role in payload.get("highlights", {}).items()}
        base_constraints = set(int(index) for index in payload.get("base_constraints", []))
        label_overrides = {int(index): label for index, label in payload.get("label_overrides", {}).items()}
        if highlights or base_constraints or label_overrides:
            highlight_svg_bases(
                path,
                highlights,
                draw_null_bubbles=bool(payload.get("draw_null_bubbles", False)),
                base_constraint_indices=base_constraints,
                label_overrides=label_overrides,
            )
        print(json.dumps({"ok": True, "svg": path.read_text(encoding="utf-8")}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc), "type": type(exc).__name__}))
    raise
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", child],
            input=json.dumps(payload),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(TREES),
            timeout=20,
            check=False,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or f"renderer exited {completed.returncode}").strip()
            print(f"[render] child failed: {message[:500]}", file=sys.stderr, flush=True)
            return svg_fallback_text(sequence, dot, message)
        data = json.loads(completed.stdout)
        if data.get("ok") and data.get("svg"):
            return str(data["svg"])
        message = str(data.get("error", "renderer returned no SVG"))
        print(f"[render] child error: {message}", file=sys.stderr, flush=True)
        return svg_fallback_text(sequence, dot, message)
    except Exception as exc:  # noqa: BLE001
        print(f"[render] child exception: {exc}", file=sys.stderr, flush=True)
        return svg_fallback_text(sequence, dot, str(exc))


def target_render_context(
    target_structure: str,
    base_mask: str = "",
    *,
    length: int | None = None,
) -> tuple[dict[int, str], set[int]]:
    if not target_structure.strip():
        return {}, set()
    _target_dot, target_side, _is_full = normalize_structure_condition(target_structure)
    if length is not None and len(target_side) != length:
        raise ValueError("target structure length must match rendered structure length")
    highlights = target_highlights_from_side(target_side)
    if base_mask.strip():
        _allowed, mask_text = normalize_base_mask(base_mask, len(target_side))
        return highlights, base_constraint_indices(mask_text)
    return highlights, set()


def score_sequence(
    sequence: str,
    condition_side: str,
    target_dot: str,
    allowed: list[set[str]],
    mask_text: str,
    *,
    exact_structure: bool,
) -> dict[str, Any]:
    md = RNA.md()
    md.uniq_ML = 1
    fc = RNA.fold_compound(sequence, md)
    mfe_dot, mfe_energy = fc.mfe()
    fc.exp_params_rescale(mfe_energy)
    _pf_structure, pf_energy = fc.pf()
    mask_ok = sequence_satisfies_mask(sequence, allowed)
    observed_side = dot_to_side(mfe_dot)
    if exact_structure:
        target_energy: float | None = float(fc.eval_structure(target_dot))
        is_compatible = compatible(sequence, condition_side) and mask_ok
        if is_compatible:
            logq = (float(pf_energy) - float(target_energy)) / RT_37C
            probability = math.exp(logq) if logq > -745.0 else 0.0
        else:
            logq = NEG_INF
            probability = 0.0
        target_partners = partners_from_side(condition_side)
        observed_partners = partners_from_side(observed_side)
        structure_wrong = sum(
            1 for left, right in zip(target_partners, observed_partners, strict=True) if left != right
        )
        structure_active = len(condition_side)
        mfe_condition_hit = structure_wrong == 0
    else:
        target_energy = None
        structure_wrong, structure_active = structure_condition_error_counts(condition_side, observed_side)
        mfe_condition_hit = structure_wrong == 0
        try:
            constrained_pf_energy = partial_constraint_pf_energy(sequence, condition_side)
            if math.isfinite(constrained_pf_energy):
                logq = (float(pf_energy) - constrained_pf_energy) / RT_37C
                probability = min(1.0, math.exp(logq) if logq > -745.0 else 0.0)
            else:
                logq = NEG_INF
                probability = 0.0
        except Exception:  # noqa: BLE001
            logq = NEG_INF
            probability = 0.0
        is_compatible = mfe_condition_hit and mask_ok
    base_wrong, base_active = base_mask_error_counts(sequence, allowed, mask_text)
    condition_wrong = structure_wrong + base_wrong
    condition_active = structure_active + base_active
    condition_error = condition_wrong / max(condition_active, 1)
    mfes = mfe_structures(fc)
    if exact_structure:
        target_is_mfe = target_dot in mfes
    else:
        target_is_mfe = any(structure_condition_error_counts(condition_side, dot_to_side(dot))[0] == 0 for dot in mfes)
    target_is_umfe = target_is_mfe and len(mfes) == 1
    mfe_condition_error = structure_wrong / max(structure_active, 1)
    return {
        "sequence": sequence,
        "length": len(sequence),
        "target_probability": probability,
        "log_target_probability": logq,
        "log_target_probability_per_nt": logq / max(len(sequence), 1),
        "mfe_dot": mfe_dot,
        "mfe_energy": float(mfe_energy),
        "target_energy": target_energy,
        "pf_energy": float(pf_energy),
        "target_is_mfe": bool(target_is_mfe),
        "target_is_umfe": bool(target_is_umfe),
        "mfe_exact": bool(mfe_condition_hit),
        "mfe_pair_error": mfe_condition_error,
        "mfe_condition_error": mfe_condition_error,
        "condition_error": condition_error,
        "structure_condition_wrong": structure_wrong,
        "structure_condition_active": structure_active,
        "base_condition_wrong": base_wrong,
        "base_condition_active": base_active,
        "condition_hit": bool(mfe_condition_hit and base_wrong == 0),
        "gc_fraction": gc_fraction(sequence),
        "mask_ok": mask_ok,
        "compatible": is_compatible,
        "mfe_structure_count": len(mfes),
    }


@dataclass(frozen=True)
class CheckpointChoice:
    label: str
    path: Path


class ModelCache:
    def __init__(self, choices: dict[str, CheckpointChoice], device_name: str, max_len_override: int) -> None:
        self.choices = choices
        self.device_name = device_name
        self.max_len_override = max_len_override
        self._lock = threading.Lock()
        self._loaded_key: str | None = None
        self._loaded_path: Path | None = None
        self._model: RnaSeq2SeqARTransformer | None = None
        self._device: torch.device | None = None
        self._meta: dict[str, Any] = {}

    def device(self) -> torch.device:
        if self.device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if self.device_name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if self.device_name == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu")

    def resolve_checkpoint(self, key: str, custom_path: str | None) -> tuple[str, Path]:
        if key == "custom":
            if not custom_path:
                raise ValueError("custom checkpoint path is empty")
            path = Path(custom_path)
            if not path.is_absolute():
                path = TREES / path
            return "custom", path
        if key not in self.choices:
            raise ValueError(f"unknown checkpoint choice {key!r}")
        return key, self.choices[key].path

    def load(self, key: str, custom_path: str | None = None) -> tuple[RnaSeq2SeqARTransformer, torch.device, dict[str, Any]]:
        resolved_key, path = self.resolve_checkpoint(key, custom_path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        with self._lock:
            if self._model is not None and self._loaded_key == resolved_key and self._loaded_path == path:
                return self._model, self._device or self.device(), dict(self._meta)
            device = self.device()
            try:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(path, map_location=device)
            config_dict = dict(checkpoint["config"])
            if self.max_len_override > int(config_dict["max_len"]):
                if str(config_dict.get("position_encoding", "")) == "learned":
                    raise ValueError("cannot raise max_len for learned-position checkpoint")
                config_dict["max_len"] = int(self.max_len_override)
            model = RnaSeq2SeqARTransformer(Seq2SeqConfig(**config_dict)).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self._model = model
            self._device = device
            self._loaded_key = resolved_key
            self._loaded_path = path
            self._meta = {
                "checkpoint": str(path),
                "device": str(device),
                "config": config_dict,
                "params": sum(param.numel() for param in model.parameters()),
            }
            return model, device, dict(self._meta)


class Designer:
    def __init__(self, model_cache: ModelCache, outdir: Path) -> None:
        self.model_cache = model_cache
        self.outdir = outdir
        self.run_dir = outdir / "runs"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._gpu_lock = threading.Lock()
        self._design_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._active_design: dict[str, Any] | None = None

    def active_design(self) -> dict[str, Any] | None:
        with self._active_lock:
            if self._active_design is None:
                return None
            active = dict(self._active_design)
        active["elapsed_seconds"] = time.time() - float(active["started_time"])
        return active

    def default_checkpoint_key(self, *, exact_structure: bool, has_base_constraints: bool) -> str:
        if exact_structure and not has_base_constraints:
            candidates = ("pretrained_small", "pretrained_large")
        else:
            candidates = ("fsb_pretrained_small", "fsb_pretrained_large", "fsb_grpo3000", "fsb_grpo4500")
        for key in candidates:
            if key in self.model_cache.choices:
                return key
        mode = "full-structure" if exact_structure else "partial-condition"
        raise ValueError(f"no default {mode} checkpoint is available")

    def sample_sequences(
        self,
        *,
        condition_side: str,
        allowed: list[set[str]],
        mask_text: str,
        count: int,
        temperature: float,
        seed: int,
        batch_size: int,
        checkpoint_key: str,
        custom_checkpoint: str | None,
    ) -> tuple[list[str], dict[str, Any]]:
        model, device, meta = self.model_cache.load(checkpoint_key, custom_checkpoint)
        length = len(condition_side)
        if length > int(model.config.max_len):
            raise ValueError(f"target length {length} exceeds loaded model max_len {model.config.max_len}")
        conditioning_mode = str(meta["config"].get("conditioning_mode", "full_structure"))
        condition = encode_workbench_condition(condition_side, mask_text, conditioning_mode)
        partners = exact_partners_for_condition(condition_side)
        condition_rows = [condition] * count
        partner_rows = [partners] * count
        allowed_ids = [
            torch.tensor([base in pos_allowed for base in BASES], dtype=torch.bool, device=device)
            for pos_allowed in allowed
        ]

        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        results: list[str] = []
        with self._gpu_lock, torch.no_grad():
            for start in range(0, count, batch_size):
                rows = condition_rows[start : start + batch_size]
                row_count = len(rows)
                max_len = length
                src = torch.full((row_count, max_len), model.src_pad_id, dtype=torch.long, device=device)
                src_valid = torch.ones((row_count, max_len), dtype=torch.bool, device=device)
                tgt_valid = torch.ones((row_count, max_len), dtype=torch.bool, device=device)
                generated = torch.full((row_count, max_len), model.tgt_pad_id, dtype=torch.long, device=device)
                partner_tensor = torch.tensor(partner_rows[start : start + batch_size], dtype=torch.long, device=device)
                for row_index, row_ids in enumerate(rows):
                    src[row_index, :max_len] = torch.tensor(row_ids, dtype=torch.long, device=device)

                memory = model.encode(src, src_valid)
                target_positions = model.position_values(tgt_valid, which="tgt")
                cross_cache = model.precompute_cross_attention_cache(memory, src_valid)
                self_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(model.decoder.layers)
                current_input = torch.full((row_count,), model.tgt_bos_id, dtype=torch.long, device=device)
                row_indices = torch.arange(row_count, device=device)
                pair_matrix = torch.zeros((len(BASES), len(BASES)), dtype=torch.bool, device=device)
                for left_base, right_base in VIENNA_ALLOWED_BASE_PAIRS:
                    pair_matrix[BASE_TO_ID[left_base], BASE_TO_ID[right_base]] = True

                for position in range(max_len):
                    logits, self_cache = model.cached_decode_step(
                        current_input,
                        position=position,
                        tgt_position_values=target_positions,
                        memory=memory,
                        src_valid=src_valid,
                        self_cache=self_cache,
                        cross_cache=cross_cache,
                    )
                    step_logits = logits.clone()
                    allowed_matrix = allowed_ids[position].expand(row_count, -1).clone()
                    partner_positions = partner_tensor[:, position]
                    partner_known = partner_positions.ge(0) & partner_positions.lt(position)
                    if bool(partner_known.any().item()):
                        constrained_rows = torch.nonzero(partner_known, as_tuple=False).flatten()
                        left_positions = partner_positions[partner_known]
                        left_tokens = generated[constrained_rows, left_positions]
                        allowed_matrix[constrained_rows] &= pair_matrix[left_tokens]
                    impossible = ~allowed_matrix.any(dim=1)
                    if bool(impossible.any().item()):
                        bad_row = int(torch.nonzero(impossible, as_tuple=False).flatten()[0].item())
                        raise ValueError(
                            f"base mask became impossible at position {position + 1} "
                            f"for sample {start + bad_row + 1}"
                        )
                    step_logits = step_logits.masked_fill(~allowed_matrix, NEG_INF)
                    if temperature <= 0.0:
                        token = step_logits.argmax(dim=-1)
                    else:
                        probabilities = (step_logits / temperature).softmax(dim=-1)
                        if device.type == "mps":
                            token = torch.multinomial(probabilities.cpu(), num_samples=1).squeeze(1).to(device)
                        else:
                            token = torch.multinomial(probabilities, num_samples=1).squeeze(1)
                    generated[row_indices, position] = token
                    current_input = token

                for row_index in range(row_count):
                    ids = generated[row_index, :max_len].detach().cpu().tolist()
                    results.append("".join(ID_TO_BASE[index] for index in ids))
        return results, meta

    def design(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._design_lock.acquire(blocking=False):
            active = self.active_design()
            elapsed = float(active.get("elapsed_seconds", 0.0)) if active else 0.0
            raise RuntimeError(f"another design request is still running ({elapsed:.0f}s elapsed)")
        with self._active_lock:
            self._active_design = {
                "started_utc": now_id(),
                "started_time": time.time(),
                "checkpoint": str(payload.get("checkpoint", "auto")),
                "n": int(payload.get("n", 32) or 32),
                "structure_length": len(re.sub(r"\s+", "", str(payload.get("structure", "")))),
            }
        try:
            return self._design_impl(payload)
        finally:
            with self._active_lock:
                self._active_design = None
            self._design_lock.release()

    def _design_impl(self, payload: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        target_dot, target_side, exact_structure = normalize_structure_condition(str(payload.get("structure", "")))
        allowed, mask_text = normalize_base_mask(str(payload.get("base_mask", "")), len(target_side))
        target_display_sequence = display_sequence_from_mask(mask_text)
        target_labels = target_label_overrides_from_mask(mask_text)
        target_highlights = target_highlights_from_side(target_side)
        requested_base_indices = base_constraint_indices(mask_text)
        if exact_structure:
            allowed = tighten_pair_compatible_masks(allowed, partners_from_side(target_side))
        count = max(1, min(512, int(payload.get("n", 32))))
        temperature = float(payload.get("temperature", 0.1))
        seed = int(payload.get("seed") or int(time.time()) % 2_000_000_000)
        batch_size = max(1, min(512, int(payload.get("batch_size", 128))))
        requested_checkpoint = str(payload.get("checkpoint", "auto") or "auto")
        checkpoint_key = (
            self.default_checkpoint_key(
                exact_structure=exact_structure,
                has_base_constraints=bool(requested_base_indices),
            )
            if requested_checkpoint == "auto"
            else requested_checkpoint
        )
        custom_checkpoint = payload.get("custom_checkpoint")
        if requested_checkpoint == "auto":
            custom_checkpoint = None
        unique_only = bool(payload.get("unique_only", True))

        print(
            (
                f"[design] start checkpoint={checkpoint_key} L={len(target_side)} N={count} "
                f"T={temperature} exact={exact_structure} base_constraints={len(requested_base_indices)}"
            ),
            file=sys.stderr,
            flush=True,
        )
        sequences, model_meta = self.sample_sequences(
            condition_side=target_side,
            allowed=allowed,
            mask_text=mask_text,
            count=count,
            temperature=temperature,
            seed=seed,
            batch_size=batch_size,
            checkpoint_key=checkpoint_key,
            custom_checkpoint=str(custom_checkpoint) if custom_checkpoint else None,
        )
        if unique_only:
            sequences = list(dict.fromkeys(sequences))
        print(f"[design] sampled unique={len(sequences)} elapsed={time.perf_counter() - t0:.2f}s", file=sys.stderr, flush=True)

        score_start = time.perf_counter()
        rows = [
            score_sequence(
                sequence,
                target_side,
                target_dot,
                allowed,
                mask_text,
                exact_structure=exact_structure,
            )
            for sequence in sequences
        ]
        score_seconds = time.perf_counter() - score_start
        print(f"[design] scored rows={len(rows)} score_seconds={score_seconds:.2f}", file=sys.stderr, flush=True)
        rows.sort(
            key=lambda row: (
                float(row["target_probability"]),
                -float(row["condition_error"]),
                bool(row["target_is_umfe"]),
                bool(row["target_is_mfe"]),
                -float(row["mfe_pair_error"]),
            ),
            reverse=True,
        )
        for index, row in enumerate(rows, start=1):
            row["rank"] = index

        best = rows[0] if rows else None
        target_svg = render_svg_text(
            target_layout_sequence(mask_text),
            target_dot,
            highlights=target_highlights,
            base_constraints=requested_base_indices,
            label_overrides=target_labels,
        )
        best_mfe_svg = (
            render_svg_text(
                str(best["sequence"]),
                str(best["mfe_dot"]),
                highlights=target_highlights,
                base_constraints=requested_base_indices,
            )
            if best
            else ""
        )
        print(f"[design] rendered elapsed={time.perf_counter() - t0:.2f}s", file=sys.stderr, flush=True)
        run_id = f"{now_id()}_{safe_name(checkpoint_key)}_L{len(target_dot)}_N{count}"
        result = {
            "run_id": run_id,
            "created_utc": now_id(),
            "request": {
                "structure": target_side,
                "target_dot": target_dot,
                "target_side": target_side,
                "exact_structure": exact_structure,
                "base_mask": mask_text,
                "target_display_sequence": target_display_sequence,
                "base_constraint_indices": sorted(requested_base_indices),
                "n": count,
                "unique_returned": len(rows),
                "temperature": temperature,
                "seed": seed,
                "batch_size": batch_size,
                "checkpoint": checkpoint_key,
                "checkpoint_source": requested_checkpoint,
                "custom_checkpoint": str(custom_checkpoint) if custom_checkpoint else "",
                "unique_only": unique_only,
            },
            "model": model_meta,
            "summary": {
                "length": len(target_dot),
                "target_pair_count": target_side.count("L"),
                "paired_unknown_count": target_side.count(PAIRED_UNKNOWN_TOKEN),
                "active_structure_count": sum(token != UNKNOWN_SIDE_TOKEN for token in target_side),
                "sample_count": count,
                "unique_count": len(rows),
                "seconds": time.perf_counter() - t0,
                "score_seconds": score_seconds,
                "best_probability": best["target_probability"] if best else None,
                "best_logq": best["log_target_probability"] if best else None,
                "best_pair_error": best["mfe_pair_error"] if best else None,
                "best_condition_error": best["condition_error"] if best else None,
                "best_mfe": best["target_is_mfe"] if best else None,
                "best_umfe": best["target_is_umfe"] if best else None,
            },
            "target_svg": target_svg,
            "best_mfe_svg": best_mfe_svg,
            "candidates": rows,
        }
        json_path = self.run_dir / f"{run_id}.json"
        json_path.write_text(json.dumps({k: v for k, v in result.items() if not k.endswith("_svg")}, indent=2) + "\n")
        result["saved_json"] = str(json_path)
        return result

    def sample_structures(self, payload: dict[str, Any]) -> dict[str, Any]:
        sequence = re.sub(r"\s+", "", str(payload.get("sequence", ""))).upper().replace("T", "U")
        if not sequence or any(base not in set(BASES) for base in sequence):
            raise ValueError("sequence must contain only A/U/C/G")
        target = str(payload.get("target_structure", "")).strip()
        base_mask = str(payload.get("base_mask", ""))
        target_dot = ""
        target_side = ""
        render_highlights: dict[int, str] = {}
        render_base_constraints: set[int] = set()
        target_is_full = False
        if target:
            target_dot, target_side, target_is_full = normalize_structure_condition(target)
            if len(target_dot) != len(sequence):
                raise ValueError("target structure length must match sequence length")
            render_highlights, render_base_constraints = target_render_context(target, base_mask, length=len(sequence))
        count = max(1, min(256, int(payload.get("count", 16))))
        seed = payload_seed(payload)
        RNA.init_rand(seed & 0xFFFFFFFF)
        md = RNA.md()
        md.uniq_ML = 1
        fc = RNA.fold_compound(sequence, md)
        mfe_dot, mfe_energy = fc.mfe()
        fc.exp_params_rescale(mfe_energy)
        _pf, pf_energy = fc.pf()
        samples = list(fc.pbacktrack(count))
        observed = Counter([mfe_dot, *samples])
        rows = []
        for index, (dot, seen) in enumerate(observed.most_common(12), start=1):
            rows.append(
                {
                    "rank": index,
                    "dot": dot,
                    "count": seen,
                    "pair_error": (
                        pair_error_rate(target_side, dot)
                        if target_side and target_is_full
                        else (
                            structure_condition_error_counts(target_side, dot_to_side(dot))[0]
                            / max(structure_condition_error_counts(target_side, dot_to_side(dot))[1], 1)
                            if target_side
                            else None
                        )
                    ),
                    "svg": render_svg_text(
                        sequence,
                        dot,
                        highlights=render_highlights,
                        base_constraints=render_base_constraints,
                    ),
                }
            )
        return {
            "sequence": sequence,
            "count": count,
            "seed": seed,
            "mfe_dot": mfe_dot,
            "mfe_energy": float(mfe_energy),
            "pf_energy": float(pf_energy),
            "structures": rows,
        }

    def random_example(self, payload: dict[str, Any]) -> dict[str, Any]:
        length = max(1, min(500, int(payload.get("length") or 80)))
        seed = payload_seed(payload)
        rng = random.Random(seed)
        sequence = "".join(rng.choice(BASES) for _ in range(length))
        md = RNA.md()
        md.uniq_ML = 1
        fc = RNA.fold_compound(sequence, md)
        mfe_dot, mfe_energy = fc.mfe()
        return {
            "sequence": sequence,
            "base_mask": sequence,
            "structure": str(mfe_dot),
            "structure_side": dot_to_side(str(mfe_dot)),
            "mfe_dot": str(mfe_dot),
            "mfe_energy": float(mfe_energy),
            "length": length,
            "seed": seed,
        }

    def mask_constraints(self, payload: dict[str, Any]) -> dict[str, Any]:
        sequence = re.sub(r"\s+", "", str(payload.get("sequence", ""))).upper().replace("T", "U")
        if not sequence or any(base not in set(BASES) for base in sequence):
            raise ValueError("mask source sequence must contain only A/U/C/G")
        _dot, side, is_full = normalize_structure_condition(str(payload.get("structure", "")))
        if not is_full:
            raise ValueError("mask source structure must be a concrete full dot-bracket structure")
        if len(side) != len(sequence):
            raise ValueError("mask source sequence and structure lengths differ")
        mask_sequence = bool(payload.get("mask_sequence", True))
        mask_structure = bool(payload.get("mask_structure", True))
        if not mask_sequence and not mask_structure:
            raise ValueError("nothing to mask")
        seed = payload_seed(payload)
        rng = random.Random(seed)
        length = len(sequence)
        base_hidden = set(rng.sample(range(length), rng.randint(0, length))) if mask_sequence else set()
        paired_degrade_probability = rng.random() if mask_structure else 0.0
        masked_structure = list(side)
        structure_hidden: set[int] = set()
        if mask_structure:
            partners = partners_from_side(side)
            units: list[tuple[int, ...]] = []
            seen: set[int] = set()
            for index, partner in enumerate(partners):
                if index in seen:
                    continue
                if partner >= 0:
                    unit = tuple(sorted((index, partner)))
                    seen.update(unit)
                else:
                    unit = (index,)
                    seen.add(index)
                units.append(unit)
            hidden_units = rng.sample(units, rng.randint(0, len(units)))
            structure_hidden = {index for unit in hidden_units for index in unit}
            for unit in units:
                if any(index in structure_hidden for index in unit):
                    for index in unit:
                        masked_structure[index] = UNKNOWN_SIDE_TOKEN
                elif len(unit) == 2 and rng.random() < paired_degrade_probability:
                    left, right = unit
                    masked_structure[left] = PAIRED_UNKNOWN_TOKEN
                    masked_structure[right] = PAIRED_UNKNOWN_TOKEN
        masked_base = "".join(
            UNKNOWN_SIDE_TOKEN if index in base_hidden else base for index, base in enumerate(sequence)
        )
        return {
            "structure": condition_to_input_text("".join(masked_structure)),
            "structure_side": "".join(masked_structure),
            "base_mask": masked_base,
            "sequence": sequence,
            "length": length,
            "seed": seed,
            "mask_sequence": mask_sequence,
            "mask_structure": mask_structure,
            "hidden_structure_count": len(structure_hidden),
            "hidden_base_count": len(base_hidden),
            "paired_degrade_probability": paired_degrade_probability,
        }


def checkpoint_choices() -> dict[str, CheckpointChoice]:
    def checkpoint_path(env_name: str, default: Path) -> Path:
        override = os.environ.get(env_name)
        if not override:
            return default
        path = Path(override).expanduser()
        return path if path.is_absolute() else TREES / path

    candidates = {
        "fsb_pretrained_small": CheckpointChoice(
            "FSB partial-base pretrained small",
            checkpoint_path(
                "RNA_WORKBENCH_FSB_SMALL",
                TREES / "checkpoints/fsb_partial_base_small.pt",
            ),
        ),
        "pretrained_small": CheckpointChoice(
            "Pretrained full-structure small",
            checkpoint_path(
                "RNA_WORKBENCH_FS_SMALL",
                TREES / "checkpoints/full_structure_small.pt",
            ),
        ),
    }
    return {key: value for key, value in candidates.items() if value.path.exists()}


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "RNAWorkbench/0.1"
    designer: Designer
    static_dir: Path
    choices: dict[str, CheckpointChoice]

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/status":
            self.send_json(
                {
                    "ok": True,
                    "device": str(self.designer.model_cache.device()),
                    "checkpoints": {
                        key: {"label": choice.label, "path": str(choice.path)}
                        for key, choice in self.choices.items()
                    },
                    "output_dir": str(self.designer.outdir),
                    "active_design": self.designer.active_design(),
                }
            )
            return
        if path in {"/", "/index.html"}:
            file_path = self.static_dir / "index.html"
        else:
            file_path = (self.static_dir / path.lstrip("/")).resolve()
            if not str(file_path).startswith(str(self.static_dir.resolve())):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = "text/html; charset=utf-8"
        if file_path.suffix == ".js":
            content_type = "text/javascript; charset=utf-8"
        elif file_path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path
            payload = self.read_json()
            if path == "/api/design":
                self.send_json(self.designer.design(payload))
                return
            if path == "/api/sample_structures":
                self.send_json(self.designer.sample_structures(payload))
                return
            if path == "/api/random_example":
                self.send_json(self.designer.random_example(payload))
                return
            if path == "/api/mask_constraints":
                self.send_json(self.designer.mask_constraints(payload))
                return
            if path == "/api/render":
                sequence = re.sub(r"\s+", "", str(payload.get("sequence", ""))).upper().replace("T", "U")
                dot, _side = normalize_structure(str(payload.get("structure", "")))
                if len(sequence) != len(dot):
                    raise ValueError("sequence and structure lengths differ")
                highlights, base_constraints = target_render_context(
                    str(payload.get("target_structure", "")),
                    str(payload.get("base_mask", "")),
                    length=len(dot),
                )
                self.send_json(
                    {
                        "svg": render_svg_text(
                            sequence,
                            dot,
                            highlights=highlights,
                            base_constraints=base_constraints,
                        )
                    }
                )
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self.send_json({"ok": False, "error": str(exc), "type": type(exc).__name__}, status=400)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local RNA design workbench.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--max-len-override", type=int, default=500)
    parser.add_argument("--outdir", type=Path, default=TREES / "outputs/rna_workbench")
    args = parser.parse_args()

    choices = checkpoint_choices()
    if not choices:
        raise SystemExit("no known checkpoints found")
    model_cache = ModelCache(choices, args.device, args.max_len_override)
    designer = Designer(model_cache, args.outdir)

    class Handler(WorkbenchHandler):
        pass

    Handler.designer = designer
    Handler.static_dir = Path(__file__).resolve().parent / "static"
    Handler.choices = choices

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"RNA workbench listening on http://{args.host}:{args.port}", flush=True)
    print(f"device={model_cache.device()} outdir={args.outdir}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
