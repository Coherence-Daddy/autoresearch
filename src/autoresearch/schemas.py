"""Pydantic v2 data models for the autoresearch judge-validation gate.

These models define the wire format used by the generator, judge, orchestrator,
and CLI layers. Keep them dependency-light — only Pydantic and pyyaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Eval(BaseModel):
    """A single binary evaluation criterion applied to a generated output."""

    name: str
    question: str
    pass_condition: str = ""
    fail_condition: str = ""


class TestInput(BaseModel):
    """A named prompt fed to the target skill."""

    # Tell pytest not to try to collect this Pydantic model as a test class.
    __test__ = False

    name: str
    prompt: str


class Output(BaseModel):
    """A single skill output captured by the Generator."""

    input_name: str
    run_index: int  # 0-based; multiple outputs per input allowed
    text: str
    model: str

    @property
    def key(self) -> str:
        """Stable identifier matching JudgeRun.output_key."""
        return f"{self.input_name}__{self.run_index}"


class JudgeRun(BaseModel):
    """A single (output, eval, rater) verdict produced by the Judge."""

    output_key: str  # f"{input_name}__{run_index}"
    eval_name: str
    rater: str  # e.g. "judge_1", "judge_2", "judge_3", "human"
    verdict: bool  # True = pass, False = fail
    reasoning: str = ""


class ValidateConfig(BaseModel):
    """Top-level run configuration loaded from YAML."""

    target_skill: Path
    inputs: list[TestInput]
    holdout: list[TestInput] = Field(default_factory=list)
    evals: list[Eval] = Field(min_length=1, max_length=8)
    generator_model: str = "claude-sonnet-4-6"
    judge_model: str = "claude-sonnet-4-6"
    mutator_model: str = "claude-opus-4-7"
    judge_reruns: int = 3  # K — how many times to re-judge each output
    runs_per_input: int = 1  # how many outputs per test input

    # Phase 2 optimizer knobs.
    max_experiments: int = 20
    holdout_every: int = 5
    stop_at_pass_rate: float = 0.95

    @classmethod
    def from_yaml(cls, path: Path) -> ValidateConfig:
        """Load a ValidateConfig from a YAML file on disk."""
        with Path(path).open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)
