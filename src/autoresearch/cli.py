"""Typer CLI entrypoint for ``autoresearch``."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from .client import AnthropicClient
from .generator import Generator
from .judge import Judge
from .loop import ExperimentRecord, run_optimizer, score_skill
from .orchestrate import (
    JUDGE_HUMAN_THRESHOLD,
    TEST_RETEST_THRESHOLD,
    ValidationReport,
    run_validation,
)
from .schemas import Eval, Output, ValidateConfig

load_dotenv()

app = typer.Typer(help="Phase 1 judge-validation gate.")

_console = Console()


@app.callback()
def _root() -> None:
    """Root callback — keeps ``validate`` as an explicit subcommand."""


def _interactive_human_grader(output: Output, eval_criterion: Eval) -> bool:
    """Default Rich-based interactive grader used by the CLI."""
    _console.rule(f"[bold]Grade: {output.input_name} (run {output.run_index})[/bold]")
    _console.print(Panel.fit(output.text, title="Output", border_style="cyan", padding=(1, 2)))
    _console.print(f"[bold]Eval:[/bold] {eval_criterion.name}")
    _console.print(f"[bold]Question:[/bold] {eval_criterion.question}")
    if eval_criterion.pass_condition:
        _console.print(f"[green]PASS IF:[/green] {eval_criterion.pass_condition}")
    if eval_criterion.fail_condition:
        _console.print(f"[red]FAIL IF:[/red] {eval_criterion.fail_condition}")
    return Confirm.ask("Pass?", default=True)


def _render_report(report: ValidationReport) -> None:
    """Render the validation report to the console with rich tables and a banner."""
    # Test-retest table.
    tr_table = Table(title="Test-retest reliability (Fleiss' kappa across judge reruns)")
    tr_table.add_column("Eval", style="bold")
    tr_table.add_column("Kappa", justify="right")
    tr_table.add_column("Threshold", justify="right")
    tr_table.add_column("Pass?", justify="center")
    for name, kappa in report.test_retest_kappa_per_eval.items():
        ok = kappa >= TEST_RETEST_THRESHOLD
        tr_table.add_row(
            name,
            f"{kappa:.3f}",
            f"{TEST_RETEST_THRESHOLD:.2f}",
            "[green]PASS[/green]" if ok else "[red]FAIL[/red]",
        )
    _console.print(tr_table)

    # Judge-vs-human table.
    jh_table = Table(title="Judge vs. human (Cohen's kappa + percent agreement)")
    jh_table.add_column("Eval", style="bold")
    jh_table.add_column("Cohen kappa", justify="right")
    jh_table.add_column("% agreement", justify="right")
    jh_table.add_column("Threshold", justify="right")
    jh_table.add_column("Pass?", justify="center")
    if report.judge_human_agreement_per_eval:
        for name, agreement in report.judge_human_agreement_per_eval.items():
            kappa = report.judge_human_kappa_per_eval.get(name, float("nan"))
            ok = agreement >= JUDGE_HUMAN_THRESHOLD
            jh_table.add_row(
                name,
                f"{kappa:.3f}",
                f"{agreement:.3f}",
                f"{JUDGE_HUMAN_THRESHOLD:.2f}",
                "[green]PASS[/green]" if ok else "[red]FAIL[/red]",
            )
    else:
        jh_table.add_row("—", "—", "—", f"{JUDGE_HUMAN_THRESHOLD:.2f}", "[yellow]SKIPPED[/yellow]")
    _console.print(jh_table)

    if report.notes:
        _console.print("[dim]Notes:[/dim]")
        for note in report.notes:
            _console.print(f"  • {note}")

    # Final banner.
    if report.pass_test_retest and report.pass_judge_human:
        _console.print(
            Panel.fit(
                "[bold white on green] GATE PASSED — proceed to Phase 2 [/bold white on green]",
                border_style="green",
            )
        )
    else:
        reasons: list[str] = []
        if not report.pass_test_retest:
            reasons.append("test-retest reliability below 0.7 on at least one eval")
        if not report.pass_judge_human:
            reasons.append("judge-vs-human agreement below 0.8 (or skipped)")
        body = "[bold white on red] GATE FAILED — do not build the optimizer [/bold white on red]"
        body += "\n" + "\n".join(f"  • {r}" for r in reasons)
        _console.print(Panel.fit(body, border_style="red"))


@app.command()
def validate(
    config_path: Path = typer.Argument(  # noqa: B008 - typer relies on call-in-default
        ..., exists=True, dir_okay=False, readable=True
    ),
    skip_human: bool = typer.Option(False, "--skip-human", help="Skip interactive human grading."),
) -> None:
    """Run the judge-validation gate against the config."""
    config = ValidateConfig.from_yaml(config_path)
    client = AnthropicClient()

    report = run_validation(
        config,
        client=client,
        skip_human=skip_human,
        human_grader=None if skip_human else _interactive_human_grader,
    )

    _render_report(report)

    if not (report.pass_test_retest and report.pass_judge_human):
        raise typer.Exit(code=1)


@app.command()
def optimize(
    config_path: Path = typer.Argument(  # noqa: B008
        ..., exists=True, dir_okay=False, readable=True
    ),
    max_experiments: int = typer.Option(
        None, "--max-experiments", help="Override config.max_experiments."
    ),
) -> None:
    """Run the Phase 2 greedy optimizer loop on the target skill."""
    config = ValidateConfig.from_yaml(config_path)
    if max_experiments is not None:
        config.max_experiments = max_experiments
    client = AnthropicClient()

    table = Table(title="Optimizer experiments (live)")
    table.add_column("Exp", justify="right")
    table.add_column("Status")
    table.add_column("Train", justify="right")
    table.add_column("Holdout", justify="right")
    table.add_column("Change")

    def _on_record(rec: ExperimentRecord) -> None:
        status_color = {"baseline": "blue", "keep": "green", "discard": "red"}.get(rec.status, "")
        train = f"{rec.train_score:.3f}"
        holdout = f"{rec.holdout_score:.3f}" if rec.holdout_score is not None else "—"
        _console.print(
            f"[{status_color}]exp {rec.experiment:>3}[/] "
            f"[{status_color}]{rec.status:<8}[/] "
            f"train={train} holdout={holdout}  {rec.description}"
        )

    _console.rule(f"[bold]Optimizing {config.target_skill.name}[/bold]")
    records = run_optimizer(config, client=client, on_record=_on_record)

    baseline = records[0].train_score
    final_best = max(r.train_score for r in records if r.status in ("baseline", "keep"))
    delta = (final_best - baseline) * 100
    color = "green" if delta > 0 else "yellow"
    _console.print(
        Panel.fit(
            f"[bold {color}]Baseline {baseline:.3f} → best {final_best:.3f}  "
            f"(Δ {delta:+.1f}pp)[/bold {color}]\n"
            f"Experiments: {len(records) - 1} attempted, "
            f"{sum(1 for r in records if r.status == 'keep')} kept",
            border_style=color,
        )
    )


@app.command()
def compare(
    config_path: Path = typer.Argument(  # noqa: B008
        ..., exists=True, dir_okay=False, readable=True
    ),
    skills: list[Path] = typer.Argument(  # noqa: B008
        ..., help="Skill .md files to score side-by-side."
    ),
    use_holdout: bool = typer.Option(
        False, "--holdout", help="Score against the holdout inputs instead of train."
    ),
) -> None:
    """Score multiple skill files against the same config and print a comparison table.

    Each skill is run through the generator+judge pipeline once per (input, eval).
    Use this to compare baseline vs optimizer-best vs your hand-edit on equal terms.
    """
    config = ValidateConfig.from_yaml(config_path)
    client = AnthropicClient()
    generator = Generator(
        client=client, skill_path=config.target_skill, model=config.generator_model
    )
    judge = Judge(client=client, model=config.judge_model)

    inputs = config.holdout if use_holdout else config.inputs
    if not inputs:
        _console.print(
            "[red]No inputs to score (holdout requested but config.holdout is empty).[/red]"
        )
        raise typer.Exit(code=1)

    eval_names = [e.name for e in config.evals]
    per_skill_per_eval: dict[str, dict[str, tuple[int, int]]] = {}
    overall: dict[str, float] = {}

    for skill_path in skills:
        if not skill_path.exists():
            _console.print(f"[red]Missing skill file: {skill_path}[/red]")
            raise typer.Exit(code=1)
        text = skill_path.read_text(encoding="utf-8")
        _console.print(f"[dim]Scoring {skill_path}...[/dim]")
        result = score_skill(
            skill_text=text,
            inputs=inputs,
            evals=config.evals,
            generator=generator,
            judge=judge,
            runs_per_input=config.runs_per_input,
        )
        # Tally per-eval pass counts.
        eval_tally: dict[str, tuple[int, int]] = {name: (0, 0) for name in eval_names}
        for run in result.judge_runs:
            passed, total = eval_tally.get(run.eval_name, (0, 0))
            eval_tally[run.eval_name] = (passed + (1 if run.verdict else 0), total + 1)
        per_skill_per_eval[str(skill_path)] = eval_tally
        overall[str(skill_path)] = result.pass_rate

    pool = "holdout" if use_holdout else "train"
    table = Table(title=f"Skill comparison ({pool} inputs)")
    table.add_column("Eval", style="bold")
    for skill_path in skills:
        table.add_column(skill_path.name, justify="right")
    for eval_name in eval_names:
        row = [eval_name]
        for skill_path in skills:
            tally = per_skill_per_eval[str(skill_path)][eval_name]
            row.append(f"{tally[0]}/{tally[1]}")
        table.add_row(*row)
    overall_row = ["[bold]Overall[/bold]"]
    for skill_path in skills:
        overall_row.append(f"[bold]{overall[str(skill_path)]:.3f}[/bold]")
    table.add_row(*overall_row)
    _console.print(table)

    # Highlight the winner.
    best_path = max(overall, key=lambda k: overall[k])
    _console.print(
        Panel.fit(
            f"[bold green]Winner ({pool}): {best_path}  ({overall[best_path]:.3f})[/bold green]",
            border_style="green",
        )
    )
