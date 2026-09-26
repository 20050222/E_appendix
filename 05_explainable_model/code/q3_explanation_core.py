"""Model-agnostic explanation calculations for Question 3.

The module does not know the Question 2 network class. A caller supplies
full-model and ablated-model outputs, so the same code can be used with the
actual frozen checkpoint after it is restored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


MODALITIES = ("text", "audio", "vision")


def _as_probability_vector(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("class probabilities must be a one-dimensional vector with at least two classes")
    if not np.isfinite(array).all():
        raise ValueError("class probabilities contain NaN or Inf")
    return array


def _as_scalar(value: Any) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 1 or not np.isfinite(array).all():
        raise ValueError("regression output must be one finite scalar")
    return float(array[0])


def _normalized_abs(values: Mapping[str, float], eps: float) -> tuple[dict[str, float], str]:
    denominator = float(sum(abs(value) for value in values.values()))
    status = "ok" if denominator > eps else "near_zero"
    return ({key: abs(value) / (denominator + eps) for key, value in values.items()}, status)


def _argmax_stable(values: Mapping[str, float]) -> str:
    return max(MODALITIES, key=lambda modality: (values.get(modality, float("-inf")), -MODALITIES.index(modality)))


@dataclass(frozen=True)
class ModalContribution:
    predicted_class: int
    classification_signed: dict[str, float]
    classification_abs: dict[str, float]
    classification_normalized: dict[str, float]
    regression_signed: dict[str, float]
    regression_abs: dict[str, float]
    regression_normalized: dict[str, float]
    classification_main_modality: str
    regression_main_modality: str
    classification_status: str
    regression_status: str


def compute_modal_contributions(
    full_probabilities: Sequence[float],
    ablated_probabilities: Mapping[str, Sequence[float]],
    full_intensity: float,
    ablated_intensities: Mapping[str, float],
    predicted_class: int | None = None,
    eps: float = 1e-8,
) -> ModalContribution:
    """Compute signed and normalized sensitivity for text/audio/vision ablations."""

    full_prob = _as_probability_vector(full_probabilities)
    target = int(np.argmax(full_prob)) if predicted_class is None else int(predicted_class)
    if target < 0 or target >= full_prob.size:
        raise ValueError("predicted_class is outside the probability vector")
    full_value = _as_scalar(full_intensity)

    cls_signed: dict[str, float] = {}
    reg_signed: dict[str, float] = {}
    for modality in MODALITIES:
        if modality not in ablated_probabilities or modality not in ablated_intensities:
            raise KeyError(f"missing ablation output for {modality}")
        ablated_prob = _as_probability_vector(ablated_probabilities[modality])
        if ablated_prob.size != full_prob.size:
            raise ValueError(f"class count mismatch for {modality}")
        cls_signed[modality] = float(full_prob[target] - ablated_prob[target])
        reg_signed[modality] = float(full_value - _as_scalar(ablated_intensities[modality]))

    cls_norm, cls_status = _normalized_abs(cls_signed, eps)
    reg_norm, reg_status = _normalized_abs(reg_signed, eps)
    return ModalContribution(
        predicted_class=target,
        classification_signed=cls_signed,
        classification_abs={key: abs(value) for key, value in cls_signed.items()},
        classification_normalized=cls_norm,
        regression_signed=reg_signed,
        regression_abs={key: abs(value) for key, value in reg_signed.items()},
        regression_normalized=reg_norm,
        classification_main_modality=_argmax_stable(cls_norm),
        regression_main_modality=_argmax_stable(reg_norm),
        classification_status=cls_status,
        regression_status=reg_status,
    )


def make_window_specs(length: int, window_size: int, stride: int | None = None) -> list[dict[str, int]]:
    """Create deterministic half-open windows and include the final boundary."""

    if length <= 0 or window_size <= 0 or window_size > length:
        raise ValueError("length and window_size must satisfy 0 < window_size <= length")
    stride = window_size if stride is None else stride
    if stride <= 0:
        raise ValueError("stride must be positive")
    starts = list(range(0, length - window_size + 1, stride))
    last_start = length - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return [{"start": start, "end": start + window_size} for start in starts]


def mask_fraction(mask: Sequence[bool] | None, start: int, end: int) -> float | None:
    """Return the observed fraction in a half-open window, if a mask exists."""

    if mask is None:
        return None
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if start < 0 or end > values.size or start >= end:
        raise ValueError("invalid window for mask")
    return float(values[start:end].mean())


def score_local_windows(
    full_target_probability: float,
    full_intensity: float,
    windows_by_modality: Mapping[str, Sequence[Mapping[str, Any]]],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Score local windows from caller-provided occluded forward outputs.

    Each window mapping must contain ``start``, ``end``, ``target_probability``
    and ``intensity``. Optional ``valid_fraction`` and ``position_unit`` are
    copied into the output for auditability.
    """

    rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        candidates = []
        for window in windows_by_modality.get(modality, []):
            start = int(window["start"])
            end = int(window["end"])
            delta_cls = float(full_target_probability - float(window["target_probability"]))
            delta_reg = float(full_intensity - float(window["intensity"]))
            candidates.append({
                "modality": modality,
                "start": start,
                "end": end,
                "delta_classification_signed": delta_cls,
                "delta_classification_abs": abs(delta_cls),
                "delta_regression_signed": delta_reg,
                "delta_regression_abs": abs(delta_reg),
                "valid_fraction": window.get("valid_fraction"),
                "position_unit": window.get("position_unit", "normalized_bin"),
            })
        cls_order = sorted(candidates, key=lambda row: (-row["delta_classification_abs"], row["start"], row["end"]))
        reg_order = sorted(candidates, key=lambda row: (-row["delta_regression_abs"], row["start"], row["end"]))
        cls_rank = {id(row): rank for rank, row in enumerate(cls_order, start=1)}
        reg_rank = {id(row): rank for rank, row in enumerate(reg_order, start=1)}
        for row in candidates:
            row["classification_rank"] = cls_rank[id(row)]
            row["regression_rank"] = reg_rank[id(row)]
            row["selected_classification"] = top_k is None or row["classification_rank"] <= top_k
            row["selected_regression"] = top_k is None or row["regression_rank"] <= top_k
            rows.append(row)
    return rows


def jaccard(first: Sequence[int], second: Sequence[int]) -> float:
    """Compute set Jaccard overlap, treating two empty sets as one."""

    first_set, second_set = set(first), set(second)
    union = first_set | second_set
    return 1.0 if not union else len(first_set & second_set) / len(union)

