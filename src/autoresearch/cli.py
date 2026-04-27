"""Typer CLI entrypoint for ``autoresearch``."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from .client import AnthropicClient
from .orchestrate import (
    JUDGE_HUMAN_THRESHOLD,
    TEST_RETEST_THRESHOLD,
    ValidationReport,
    run_validation,
)
from .schemas import Eval, Output, ValidateConfig

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
