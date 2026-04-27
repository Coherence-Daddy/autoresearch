"""Orchestration pipeline for the Phase 1 judge-validation gate.

Wires Generator + Judge + metrics into a single ``run_validation`` call that
produces a ``ValidationReport`` consumed by the CLI.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from .client import AnthropicClient
from .generator import Generator
from .judge import Judge
from .metrics import cohen_kappa, fleiss_kappa, percent_agreement
from .schemas import Eval, JudgeRun, Output, ValidateConfig

# Gate thresholds.
TEST_RETEST_THRESHOLD = 0.7
JUDGE_HUMAN_THRESHOLD = 0.8


HumanGrader = Callable[[Output, Eval], bool]


class ValidationReport(BaseModel):
    """Aggregated validation results for a single config run."""

    n_outputs: int
    n_evals: int
    judge_reruns: int
    test_retest_kappa_per_eval: dict[str, float]
    judge_human_kappa_per_eval: dict[str, float]
    judge_human_agreement_per_eval: dict[str, float]
    pass_test_retest: bool
    pass_judge_human: bool
    notes: list[str] = Field(default_factory=list)


def _safe_fleiss(ratings: list[list[bool]]) -> tuple[float, str | None]:
    """Compute Fleiss' kappa with edge-case fallbacks.

    Returns (kappa, note). ``note`` is a non-empty string when a fallback path
    was taken, otherwise ``None``.
    """
    if not ratings:
        return 0.0, "no items to score"
    try:
        return fleiss_kappa(ratings), None
    except ValueError:
        # Either zero variance across raters or some other degeneracy.
        # Decide based on observed agreement.
        all_agree = all(all(r == item[0] for r in item) for item in ratings)
        if all_agree:
            return 1.0, "fleiss_kappa undefined (all raters unanimous); reporting 1.0"
        return 0.0, "fleiss_kappa undefined (degenerate input); reporting 0.0"


def _safe_cohen(a: list[bool], b: list[bool]) -> tuple[float, str | None]:
    """Compute Cohen's kappa with edge-case fallbacks."""
    if not a:
        return 0.0, "no items to score"
    try:
        return cohen_kappa(a, b), None
    except ValueError:
        if a == b:
            return 1.0, "cohen_kappa undefined (perfect agreement, zero variance); reporting 1.0"
        return 0.0, "cohen_kappa undefined (degenerate input); reporting 0.0"


def _default_human_grader(_output: Output, _eval: Eval) -> bool:
    """Fallback used when no human_grader is supplied and skip_human is False."""
    raise RuntimeError(
        "human_grader required when skip_human=False; the CLI supplies an interactive one"
    )


def run_validation(
    config: ValidateConfig,
    *,
    client: AnthropicClient | None = None,
    skip_human: bool = False,
    human_grader: HumanGrader | None = None,
) -> ValidationReport:
    """Run the full Phase 1 validation pipeline and return a report.

    See module docstring for the contract; behaviour mirrors the spec.
    """
    if client is None:
        client = AnthropicClient()

    generator = Generator(client, config.target_skill, config.generator_model)
    judge = Judge(client, config.judge_model)

    notes: list[str] = []

    # 1. Generate outputs for every (input, run_index).
    outputs: list[Output] = []
    for test_input in config.inputs:
        for run_index in range(config.runs_per_input):
            outputs.append(generator.run(test_input, run_index=run_index))

    # 2. Run the judge K times per (output, eval).
    judge_runs: list[JudgeRun] = []
    for output in outputs:
        for eval_criterion in config.evals:
            for k in range(config.judge_reruns):
                rater_label = f"judge_{k + 1}"
                judge_runs.append(judge.score(output, eval_criterion, rater_label))

    # 3. Optionally collect human grades (one per (output, eval)).
    human_runs: list[JudgeRun] = []
    if skip_human:
        notes.append("skipped human grading (--skip-human); judge-vs-human gate not evaluated")
    else:
        grader = human_grader or _default_human_grader
        for output in outputs:
            for eval_criterion in config.evals:
                verdict = bool(grader(output, eval_criterion))
                human_runs.append(
                    JudgeRun(
                        output_key=output.key,
                        eval_name=eval_criterion.name,
                        rater="human",
                        verdict=verdict,
                        reasoning="human grader",
                    )
                )

    # Index judge_runs by (output_key, eval_name, rater) for fast lookup.
    by_key: dict[tuple[str, str, str], JudgeRun] = {
        (jr.output_key, jr.eval_name, jr.rater): jr for jr in judge_runs
    }
    human_by_key: dict[tuple[str, str], JudgeRun] = {
        (hr.output_key, hr.eval_name): hr for hr in human_runs
    }

    # 4. Per-eval Fleiss' kappa across the K judge reruns.
    test_retest: dict[str, float] = {}
    for eval_criterion in config.evals:
        ratings: list[list[bool]] = []
        for output in outputs:
            row: list[bool] = []
            for k in range(config.judge_reruns):
                rater = f"judge_{k + 1}"
                jr = by_key[(output.key, eval_criterion.name, rater)]
                row.append(jr.verdict)
            ratings.append(row)
        kappa, note = _safe_fleiss(ratings)
        test_retest[eval_criterion.name] = kappa
        if note:
            notes.append(f"{eval_criterion.name} test-retest: {note}")

    # 5. Per-eval Cohen's kappa + percent agreement (judge_1 vs human).
    judge_human_kappa: dict[str, float] = {}
    judge_human_agreement: dict[str, float] = {}
    if not skip_human:
        for eval_criterion in config.evals:
            judge_verdicts: list[bool] = []
            human_verdicts: list[bool] = []
            for output in outputs:
                jr = by_key[(output.key, eval_criterion.name, "judge_1")]
                hr = human_by_key[(output.key, eval_criterion.name)]
                judge_verdicts.append(jr.verdict)
                human_verdicts.append(hr.verdict)
            kappa, note = _safe_cohen(judge_verdicts, human_verdicts)
            judge_human_kappa[eval_criterion.name] = kappa
            if note:
                notes.append(f"{eval_criterion.name} judge-vs-human: {note}")
            judge_human_agreement[eval_criterion.name] = percent_agreement(
                judge_verdicts, human_verdicts
            )

    # 6. Compute pass/fail flags.
    pass_test_retest = bool(test_retest) and all(
        v >= TEST_RETEST_THRESHOLD for v in test_retest.values()
    )
    if skip_human or not judge_human_agreement:
        pass_judge_human = False
    else:
        pass_judge_human = all(v >= JUDGE_HUMAN_THRESHOLD for v in judge_human_agreement.values())

    return ValidationReport(
        n_outputs=len(outputs),
        n_evals=len(config.evals),
        judge_reruns=config.judge_reruns,
        test_retest_kappa_per_eval=test_retest,
        judge_human_kappa_per_eval=judge_human_kappa,
        judge_human_agreement_per_eval=judge_human_agreement,
        pass_test_retest=pass_test_retest,
        pass_judge_human=pass_judge_human,
        notes=notes,
    )
