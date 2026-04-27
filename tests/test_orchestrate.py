"""Tests for ``autoresearch.orchestrate.run_validation`` (mocked, no live API)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autoresearch.orchestrate import run_validation
from autoresearch.schemas import Eval, JudgeRun, Output, TestInput, ValidateConfig


def _make_config(tmp_path: Path, judge_reruns: int = 3) -> ValidateConfig:
    skill = tmp_path / "skill.md"
    skill.write_text("# stub skill\nRespond.\n", encoding="utf-8")
    return ValidateConfig(
        target_skill=skill,
        generator_model="stub-model",
        judge_model="stub-model",
        judge_reruns=judge_reruns,
        runs_per_input=1,
        inputs=[
            TestInput(name="in_a", prompt="A"),
            TestInput(name="in_b", prompt="B"),
        ],
        evals=[
            Eval(name="eval_x", question="x?"),
            Eval(name="eval_y", question="y?"),
        ],
    )


def _patch_pipeline(monkeypatch, *, judge_verdicts):
    """Replace Generator.run + Judge.score with deterministic fakes.

    ``judge_verdicts`` is a callable (output, eval_criterion, rater) -> bool.
    """
    from autoresearch import orchestrate as orch

    def fake_gen_run(self, test_input, run_index=0):
        return Output(
            input_name=test_input.name,
            run_index=run_index,
            text=f"text-for-{test_input.name}-{run_index}",
            model="stub-model",
        )

    def fake_judge_score(self, output, eval_criterion, rater_label):
        return JudgeRun(
            output_key=output.key,
            eval_name=eval_criterion.name,
            rater=rater_label,
            verdict=judge_verdicts(output, eval_criterion, rater_label),
            reasoning="mocked",
        )

    monkeypatch.setattr(orch.Generator, "run", fake_gen_run)
    monkeypatch.setattr(orch.Judge, "score", fake_judge_score)


def test_counts_and_unanimous_judges_passes_test_retest(tmp_path, monkeypatch):
    config = _make_config(tmp_path)

    # All judges always say True for eval_x; always False for eval_y.
    def verdicts(_o, e, _r):
        return e.name == "eval_x"

    _patch_pipeline(monkeypatch, judge_verdicts=verdicts)

    report = run_validation(
        config,
        client=MagicMock(),
        skip_human=False,
        human_grader=lambda o, e: e.name == "eval_x",  # matches judge perfectly
    )

    assert report.n_outputs == 2  # 2 inputs * 1 run
    assert report.n_evals == 2
    assert report.judge_reruns == 3

    # Perfect agreement -> kappa fallback to 1.0.
    assert report.test_retest_kappa_per_eval["eval_x"] == 1.0
    assert report.test_retest_kappa_per_eval["eval_y"] == 1.0
    assert report.pass_test_retest is True

    # Judge-1 vs human: identical -> agreement 1.0.
    assert report.judge_human_agreement_per_eval["eval_x"] == 1.0
    assert report.judge_human_agreement_per_eval["eval_y"] == 1.0
    assert report.pass_judge_human is True


def test_disagreeing_judges_fail_test_retest(tmp_path, monkeypatch):
    config = _make_config(tmp_path, judge_reruns=3)

    # judge_1 always True, judge_2 always False, judge_3 always True.
    # Across 2 items this is maximally inconsistent => low kappa.
    def verdicts(_o, _e, rater):
        return rater in {"judge_1", "judge_3"}

    _patch_pipeline(monkeypatch, judge_verdicts=verdicts)

    report = run_validation(
        config,
        client=MagicMock(),
        skip_human=True,
    )

    # Not unanimous -> fleiss returns a real (low) value below 0.7.
    for name in ("eval_x", "eval_y"):
        assert report.test_retest_kappa_per_eval[name] < 0.7
    assert report.pass_test_retest is False


def test_skip_human_marks_judge_human_failed_with_note(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    _patch_pipeline(monkeypatch, judge_verdicts=lambda _o, _e, _r: True)

    report = run_validation(
        config,
        client=MagicMock(),
        skip_human=True,
    )

    assert report.judge_human_kappa_per_eval == {}
    assert report.judge_human_agreement_per_eval == {}
    assert report.pass_judge_human is False
    assert any("skipped human grading" in n for n in report.notes)


def test_human_grader_is_invoked_once_per_output_eval(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    _patch_pipeline(monkeypatch, judge_verdicts=lambda _o, _e, _r: True)

    calls: list[tuple[str, str]] = []

    def grader(output, eval_criterion):
        calls.append((output.key, eval_criterion.name))
        return True

    run_validation(config, client=MagicMock(), skip_human=False, human_grader=grader)

    # 2 outputs * 2 evals = 4 calls, all distinct.
    assert len(calls) == 4
    assert len(set(calls)) == 4


def test_default_human_grader_required_when_not_skipped(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    _patch_pipeline(monkeypatch, judge_verdicts=lambda _o, _e, _r: True)

    with pytest.raises(RuntimeError, match="human_grader required"):
        run_validation(config, client=MagicMock(), skip_human=False, human_grader=None)
