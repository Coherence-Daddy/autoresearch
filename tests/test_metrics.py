"""Tests for autoresearch.metrics."""

import pytest

from autoresearch.metrics import cohen_kappa, fleiss_kappa, percent_agreement

# ---------------------------------------------------------------------------
# percent_agreement
# ---------------------------------------------------------------------------


class TestPercentAgreement:
    def test_all_agree(self) -> None:
        assert percent_agreement([True, True, False], [True, True, False]) == 1.0

    def test_all_disagree(self) -> None:
        assert percent_agreement([True, True, False], [False, False, True]) == 0.0

    def test_half_agreement(self) -> None:
        assert percent_agreement([True, True, False, False], [True, False, False, True]) == 0.5

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            percent_agreement([True, False], [True, False, True])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            percent_agreement([], [])


# ---------------------------------------------------------------------------
# cohen_kappa
# ---------------------------------------------------------------------------


class TestCohenKappa:
    def test_known_answer(self) -> None:
        # rater_a = [T,T,F,F,T], rater_b = [T,F,F,T,T]
        # Agreements: items 0, 2, 4 -> po = 0.6
        # marginals: a_T=3/5=0.6, a_F=0.4; b_T=3/5=0.6, b_F=0.4
        # pe = 0.6*0.6 + 0.4*0.4 = 0.52
        # kappa = (0.6 - 0.52) / (1 - 0.52) = 0.08 / 0.48 = 0.166666...
        rater_a = [True, True, False, False, True]
        rater_b = [True, False, False, True, True]
        result = cohen_kappa(rater_a, rater_b)
        assert round(result, 3) == 0.167

    def test_perfect_agreement(self) -> None:
        # All True -> pe = 1.0 and po = 1.0 -> defined as 1.0
        assert cohen_kappa([True, True, True], [True, True, True]) == 1.0

    def test_perfect_agreement_mixed_labels(self) -> None:
        # Both raters give same mixed labels -> po=1.0, pe<1.0 -> kappa=1.0
        rater_a = [True, False, True, False]
        rater_b = [True, False, True, False]
        assert cohen_kappa(rater_a, rater_b) == 1.0

    def test_perfect_disagreement_balanced(self) -> None:
        # Balanced labels with full disagreement -> kappa = -1.0.
        # rater_a = [T,F,T,F], rater_b = [F,T,F,T]
        # po = 0, marginals: a_T=0.5, b_T=0.5 -> pe = 0.5
        # kappa = (0 - 0.5) / 0.5 = -1.0
        rater_a = [True, False, True, False]
        rater_b = [False, True, False, True]
        assert cohen_kappa(rater_a, rater_b) == -1.0

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            cohen_kappa([True, False], [True, False, True])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            cohen_kappa([], [])

    def test_perfect_agreement_all_false(self) -> None:
        # Both raters all False -> pe = 1.0, po = 1.0 -> kappa defined as 1.0.
        assert cohen_kappa([False, False], [False, False]) == 1.0


# ---------------------------------------------------------------------------
# fleiss_kappa
# ---------------------------------------------------------------------------


class TestFleissKappa:
    def test_known_answer(self) -> None:
        # 3 raters, 5 items.
        # Item counts (T, F): (3,0), (2,1), (0,3), (1,2), (2,1)
        # P_i = (t^2 + f^2 - 3) / 6 = [1.0, 1/3, 1.0, 1/3, 1/3]
        # P_bar = (1 + 1/3 + 1 + 1/3 + 1/3) / 5 = 3/5 = 0.6
        # totals: T=8, F=7 of 15. p_T=8/15, p_F=7/15
        # Pe_bar = (8/15)^2 + (7/15)^2 = 113/225 ~= 0.502222
        # kappa = (0.6 - 113/225) / (1 - 113/225) = 22/112 ~= 0.196428
        ratings = [
            [True, True, True],
            [True, True, False],
            [False, False, False],
            [True, False, False],
            [True, True, False],
        ]
        result = fleiss_kappa(ratings)
        assert round(result, 4) == round(22 / 112, 4)

    def test_perfect_agreement(self) -> None:
        ratings = [
            [True, True, True],
            [False, False, False],
            [True, True, True],
        ]
        assert fleiss_kappa(ratings) == 1.0

    def test_perfect_agreement_all_same_label(self) -> None:
        # All raters say True for all items -> pe_bar = 1, p_bar = 1 -> 1.0
        ratings = [
            [True, True, True],
            [True, True, True],
        ]
        assert fleiss_kappa(ratings) == 1.0

    def test_mismatched_rater_counts_raises(self) -> None:
        ratings = [
            [True, True, True],
            [True, False],
            [False, False, False],
        ]
        with pytest.raises(ValueError):
            fleiss_kappa(ratings)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            fleiss_kappa([])

    def test_single_rater_raises(self) -> None:
        with pytest.raises(ValueError):
            fleiss_kappa([[True], [False], [True]])
