"""Inter-rater agreement metrics for binary judge validation.

Provides percent agreement, Cohen's kappa (two raters), and Fleiss' kappa
(K raters) over binary (pass/fail) labels. Pure Python, no external deps.
"""

from collections.abc import Sequence


def percent_agreement(rater_a: Sequence[bool], rater_b: Sequence[bool]) -> float:
    """Fraction of items where both raters agree. Returns 0.0-1.0."""
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must have the same length")
    n = len(rater_a)
    if n == 0:
        raise ValueError("inputs must be non-empty")
    agree = sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a == b)
    return agree / n


def cohen_kappa(rater_a: Sequence[bool], rater_b: Sequence[bool]) -> float:
    """Cohen's kappa for two raters on binary labels. Returns -1.0-1.0."""
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must have the same length")
    n = len(rater_a)
    if n == 0:
        raise ValueError("inputs must be non-empty")

    po = percent_agreement(rater_a, rater_b)

    a_true = sum(1 for x in rater_a if x) / n
    a_false = 1.0 - a_true
    b_true = sum(1 for x in rater_b if x) / n
    b_false = 1.0 - b_true

    pe = a_true * b_true + a_false * b_false

    if pe == 1.0:
        # All ratings identical across both raters; kappa undefined.
        if po == 1.0:
            return 1.0
        raise ValueError(
            "expected agreement is 1.0 but observed agreement is not; kappa is undefined"
        )

    return (po - pe) / (1.0 - pe)


def fleiss_kappa(ratings: Sequence[Sequence[bool]]) -> float:
    """Fleiss' kappa for K raters on binary labels."""
    n = len(ratings)
    if n == 0:
        raise ValueError("ratings must be non-empty")

    k = len(ratings[0])
    if k == 0:
        raise ValueError("each item must have at least one rater")
    if any(len(item) != k for item in ratings):
        raise ValueError("all items must have the same number of raters")
    if k < 2:
        raise ValueError("Fleiss' kappa requires at least two raters per item")

    # Per-item counts of each category (True, False).
    # n_ij = number of raters that assigned category j to item i.
    per_item_true: list[int] = [sum(1 for r in item if r) for item in ratings]
    per_item_false: list[int] = [k - t for t in per_item_true]

    # P_i: extent of rater agreement for item i.
    # P_i = (1 / (k * (k - 1))) * (sum_j n_ij^2 - k)
    p_items: list[float] = [
        (t * t + f * f - k) / (k * (k - 1))
        for t, f in zip(per_item_true, per_item_false, strict=True)
    ]
    p_bar = sum(p_items) / n

    # p_j: proportion of all assignments to category j.
    total = n * k
    p_true = sum(per_item_true) / total
    p_false = sum(per_item_false) / total
    pe_bar = p_true * p_true + p_false * p_false

    if pe_bar == 1.0:
        # All raters chose the same single category across all items.
        if p_bar == 1.0:
            return 1.0
        raise ValueError(
            "expected agreement is 1.0 but observed agreement is not; kappa is undefined"
        )

    return (p_bar - pe_bar) / (1.0 - pe_bar)
