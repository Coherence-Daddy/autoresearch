"""Phase 2: smallest viable optimizer loop.

Greedy single-mutation. Score = pass rate over (input x eval) pairs from one
judge run. Mutations that improve train score are kept. Mutations that don't
are reverted. Holdout is scored every K experiments to flag overfitting.
Logs JSONL to runs/<timestamp>/log.jsonl. Snapshots kept skills.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .client import AnthropicClient
from .generator import Generator
from .judge import Judge
from .mutator import Mutator
from .schemas import Eval, JudgeRun, Output, TestInput, ValidateConfig


@dataclass
class ScoreResult:
    """Outcome of scoring a skill against a list of inputs and evals."""

    pass_rate: float
    n_pass: int
    n_total: int
    outputs: list[Output] = field(default_factory=list)
    judge_runs: list[JudgeRun] = field(default_factory=list)


@dataclass
class ExperimentRecord:
    """One experiment in the optimizer loop."""

    experiment: int
    train_score: float
    holdout_score: float | None
    status: str  # "baseline" | "keep" | "discard"
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "train_score": self.train_score,
            "holdout_score": self.holdout_score,
            "status": self.status,
            "description": self.description,
        }


def score_skill(
    *,
    skill_text: str,
    inputs: list[TestInput],
    evals: list[Eval],
    generator: Generator,
    judge: Judge,
    runs_per_input: int = 1,
) -> ScoreResult:
    """Run the skill on every input x eval pair once and return pass rate."""
    outputs: list[Output] = []
    runs: list[JudgeRun] = []
    n_pass = 0
    n_total = 0

    generator.set_skill_text(skill_text)
    for test_input in inputs:
        for run_index in range(runs_per_input):
            output = generator.run(test_input, run_index=run_index)
            outputs.append(output)
            for eval_criterion in evals:
                run = judge.score(output, eval_criterion, rater_label="judge_optimize")
                runs.append(run)
                n_total += 1
                if run.verdict:
                    n_pass += 1

    pass_rate = (n_pass / n_total) if n_total else 0.0
    return ScoreResult(
        pass_rate=pass_rate, n_pass=n_pass, n_total=n_total, outputs=outputs, judge_runs=runs
    )


def run_optimizer(
    config: ValidateConfig,
    *,
    client: AnthropicClient,
    log_dir: Path | None = None,
    on_record: Callable[[ExperimentRecord], None] | None = None,
) -> list[ExperimentRecord]:
    """Run the greedy single-mutation optimizer loop.

    Returns the list of experiment records (including baseline).
    The kept skill is saved to ``log_dir / SKILL.md.best``.
    """
    log_dir = log_dir or _default_log_dir(config.target_skill)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log.jsonl"
    snapshots = log_dir / "snapshots"
    snapshots.mkdir(exist_ok=True)

    baseline_text = config.target_skill.read_text(encoding="utf-8")
    (log_dir / "SKILL.md.baseline").write_text(baseline_text, encoding="utf-8")

    generator = Generator(
        client=client, skill_path=config.target_skill, model=config.generator_model
    )
    judge = Judge(client=client, model=config.judge_model)
    mutator = Mutator(client=client, model=config.mutator_model)

    records: list[ExperimentRecord] = []
    best_text = baseline_text

    # Experiment 0: baseline.
    baseline_score = score_skill(
        skill_text=baseline_text,
        inputs=config.inputs,
        evals=config.evals,
        generator=generator,
        judge=judge,
        runs_per_input=config.runs_per_input,
    )
    holdout_score = _maybe_holdout(0, config, generator, judge, baseline_text)
    rec = ExperimentRecord(
        experiment=0,
        train_score=baseline_score.pass_rate,
        holdout_score=holdout_score,
        status="baseline",
        description="original skill — no changes",
    )
    records.append(rec)
    _log_record(log_path, rec)
    if on_record:
        on_record(rec)

    best_score = baseline_score.pass_rate
    last_score_result = baseline_score

    consecutive_at_target = 0
    stuck_count = 0
    history: list[str] = []
    for exp_id in range(1, config.max_experiments + 1):
        # Identify failures from the most recent kept score result.
        failures = _failure_examples(last_score_result, config.evals)
        force_divergent = stuck_count >= 3
        mutation = mutator.propose(
            best_text, failures, history=history, force_divergent=force_divergent
        )

        candidate_text = mutation.new_skill_text
        if candidate_text == best_text:
            rec = ExperimentRecord(
                experiment=exp_id,
                train_score=best_score,
                holdout_score=None,
                status="discard",
                description=f"no-op mutation: {mutation.description}",
            )
            records.append(rec)
            _log_record(log_path, rec)
            if on_record:
                on_record(rec)
            continue

        candidate_score = score_skill(
            skill_text=candidate_text,
            inputs=config.inputs,
            evals=config.evals,
            generator=generator,
            judge=judge,
            runs_per_input=config.runs_per_input,
        )

        if candidate_score.pass_rate > best_score:
            best_text = candidate_text
            best_score = candidate_score.pass_rate
            last_score_result = candidate_score
            status = "keep"
            stuck_count = 0
            (snapshots / f"exp-{exp_id:04d}.md").write_text(candidate_text, encoding="utf-8")
        else:
            status = "discard"
            stuck_count += 1
        history.append(f"[{status}] {mutation.description}")

        holdout_score = _maybe_holdout(exp_id, config, generator, judge, best_text)
        rec = ExperimentRecord(
            experiment=exp_id,
            train_score=candidate_score.pass_rate,
            holdout_score=holdout_score,
            status=status,
            description=mutation.description,
        )
        records.append(rec)
        _log_record(log_path, rec)
        if on_record:
            on_record(rec)

        if best_score >= config.stop_at_pass_rate:
            consecutive_at_target += 1
            if consecutive_at_target >= 3:
                break
        else:
            consecutive_at_target = 0

    (log_dir / "SKILL.md.best").write_text(best_text, encoding="utf-8")
    return records


def _failure_examples(score: ScoreResult, evals: list[Eval]) -> list[tuple[Output, Eval, JudgeRun]]:
    """Build the failures list a Mutator expects from a ScoreResult."""
    by_key: dict[str, Output] = {o.key: o for o in score.outputs}
    eval_by_name: dict[str, Eval] = {e.name: e for e in evals}
    failures: list[tuple[Output, Eval, JudgeRun]] = []
    for run in score.judge_runs:
        if run.verdict:
            continue
        output = by_key.get(run.output_key)
        eval_obj = eval_by_name.get(run.eval_name)
        if output is None or eval_obj is None:
            continue
        failures.append((output, eval_obj, run))
    return failures


def _maybe_holdout(
    exp_id: int,
    config: ValidateConfig,
    generator: Generator,
    judge: Judge,
    skill_text: str,
) -> float | None:
    """Score holdout if configured and on schedule."""
    if not config.holdout:
        return None
    if exp_id != 0 and exp_id % config.holdout_every != 0:
        return None
    result = score_skill(
        skill_text=skill_text,
        inputs=config.holdout,
        evals=config.evals,
        generator=generator,
        judge=judge,
        runs_per_input=1,
    )
    return result.pass_rate


def _log_record(log_path: Path, record: ExperimentRecord) -> None:
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.as_dict()) + "\n")


def _default_log_dir(skill_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return Path("runs") / f"{skill_path.stem}-{timestamp}"


__all__ = ["ExperimentRecord", "ScoreResult", "cleanup_dir", "run_optimizer", "score_skill"]


def cleanup_dir(path: Path) -> None:
    """Remove a run directory tree (used by tests)."""
    if path.exists():
        shutil.rmtree(path)
