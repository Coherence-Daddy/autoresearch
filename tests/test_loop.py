"""Tests for the optimizer loop (mocked — no live API)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.client import AnthropicClient
from autoresearch.generator import Generator
from autoresearch.judge import Judge
from autoresearch.loop import (
    ExperimentRecord,
    cleanup_dir,
    run_optimizer,
    score_skill,
)
from autoresearch.mutator import Mutation
from autoresearch.schemas import Eval, JudgeRun, Output, TestInput, ValidateConfig


@pytest.fixture
def client(mocker):
    return mocker.MagicMock(spec=AnthropicClient)


@pytest.fixture
def tmp_skill(tmp_path: Path) -> Path:
    p = tmp_path / "skill.md"
    p.write_text("# Original Skill\n\nDo the thing.\n", encoding="utf-8")
    return p


def _config(tmp_skill: Path) -> ValidateConfig:
    return ValidateConfig(
        target_skill=tmp_skill,
        inputs=[TestInput(name="i1", prompt="hello"), TestInput(name="i2", prompt="world")],
        evals=[
            Eval(name="short", question="short?"),
            Eval(name="happy", question="happy?"),
        ],
        max_experiments=2,
    )


def test_score_skill_pass_rate(mocker, client, tmp_skill) -> None:
    """score_skill computes pass rate as judge passes / (inputs * evals * runs)."""
    config = _config(tmp_skill)
    generator = Generator(client=client, skill_path=tmp_skill, model="m")
    mocker.patch.object(
        generator,
        "run",
        side_effect=lambda ti, run_index=0: Output(
            input_name=ti.name, run_index=run_index, text=f"out-{ti.name}", model="m"
        ),
    )
    judge = Judge(client=client, model="m")
    # Alternate pass/fail across all judge calls — 2 inputs x 2 evals = 4 calls.
    verdicts = iter([True, False, True, True])
    mocker.patch.object(
        judge,
        "score",
        side_effect=lambda output, eval_c, rater_label: JudgeRun(
            output_key=output.key,
            eval_name=eval_c.name,
            rater=rater_label,
            verdict=next(verdicts),
        ),
    )

    result = score_skill(
        skill_text="anything",
        inputs=config.inputs,
        evals=config.evals,
        generator=generator,
        judge=judge,
    )
    assert result.n_total == 4
    assert result.n_pass == 3
    assert result.pass_rate == 0.75


def test_run_optimizer_keeps_improvement_and_discards_regression(
    mocker, client, tmp_skill, tmp_path
) -> None:
    """End-to-end: baseline → improve → regress. Verify keep/discard logic + log."""
    config = _config(tmp_skill)

    # Mock score_skill at the loop layer so we control train scores deterministically.
    train_scores = iter(
        [0.50, 0.80, 0.70]
    )  # baseline, exp1 (better → keep), exp2 (worse → discard)

    def fake_score_skill(*, skill_text, inputs, evals, generator, judge, runs_per_input=1):
        from autoresearch.loop import ScoreResult

        return ScoreResult(pass_rate=next(train_scores), n_pass=0, n_total=0)

    mocker.patch("autoresearch.loop.score_skill", side_effect=fake_score_skill)

    # Two distinct mutations.
    mutations = iter(
        [
            Mutation(description="add brevity rule", new_skill_text="# v1\nKeep brief.\n"),
            Mutation(description="add formality rule", new_skill_text="# v2\nBe formal.\n"),
        ]
    )
    mocker.patch(
        "autoresearch.loop.Mutator.propose",
        side_effect=lambda current, failures: next(mutations),
    )

    log_dir = tmp_path / "run"
    records = run_optimizer(config, client=client, log_dir=log_dir)

    assert [r.experiment for r in records] == [0, 1, 2]
    assert records[0].status == "baseline"
    assert records[1].status == "keep"
    assert records[2].status == "discard"

    # Best skill is the kept one.
    best = (log_dir / "SKILL.md.best").read_text(encoding="utf-8")
    assert "Keep brief." in best

    # Snapshot was saved for the kept experiment, none for the discarded.
    assert (log_dir / "snapshots" / "exp-0001.md").exists()
    assert not (log_dir / "snapshots" / "exp-0002.md").exists()

    # JSONL log contains 3 lines.
    lines = (log_dir / "log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_run_optimizer_no_op_mutation_is_discarded(mocker, client, tmp_skill, tmp_path) -> None:
    """If the mutator returns the same skill text, no scoring happens — discard fast."""
    config = _config(tmp_skill)
    config.max_experiments = 1
    baseline_text = tmp_skill.read_text(encoding="utf-8")

    score_calls = 0

    def fake_score_skill(*, skill_text, inputs, evals, generator, judge, runs_per_input=1):
        from autoresearch.loop import ScoreResult

        nonlocal score_calls
        score_calls += 1
        return ScoreResult(pass_rate=0.5, n_pass=0, n_total=0)

    mocker.patch("autoresearch.loop.score_skill", side_effect=fake_score_skill)
    mocker.patch(
        "autoresearch.loop.Mutator.propose",
        return_value=Mutation(description="no op", new_skill_text=baseline_text),
    )

    log_dir = tmp_path / "run-noop"
    records = run_optimizer(config, client=client, log_dir=log_dir)

    # baseline + 1 experiment.
    assert len(records) == 2
    assert records[1].status == "discard"
    assert "no-op" in records[1].description
    # Only the baseline triggered score_skill.
    assert score_calls == 1


def test_cleanup_dir_removes_tree(tmp_path: Path) -> None:
    target = tmp_path / "removable"
    target.mkdir()
    (target / "file").write_text("x")
    cleanup_dir(target)
    assert not target.exists()
    # Idempotent on missing dirs.
    cleanup_dir(target)


def test_experiment_record_as_dict() -> None:
    rec = ExperimentRecord(
        experiment=3, train_score=0.9, holdout_score=None, status="keep", description="x"
    )
    d = rec.as_dict()
    assert d["experiment"] == 3
    assert d["train_score"] == 0.9
    assert d["holdout_score"] is None
