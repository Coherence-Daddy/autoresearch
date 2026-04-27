"""Tests for autoresearch.schemas."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from autoresearch.schemas import Eval, JudgeRun, Output, TestInput, ValidateConfig


def test_validate_config_from_yaml(tmp_path: Path) -> None:
    """A well-formed YAML file parses into a ValidateConfig."""
    cfg_path = tmp_path / "config.yaml"
    skill_path = tmp_path / "skill.md"
    skill_path.write_text("# skill", encoding="utf-8")

    payload = {
        "target_skill": str(skill_path),
        "inputs": [
            {"name": "first", "prompt": "Say hello"},
            {"name": "second", "prompt": "Say goodbye"},
        ],
        "evals": [
            {
                "name": "polite",
                "question": "Is the response polite?",
                "pass_condition": "uses please/thank you",
                "fail_condition": "rude tone",
            }
        ],
        "generator_model": "claude-sonnet-4-6",
        "judge_model": "claude-sonnet-4-6",
        "judge_reruns": 5,
        "runs_per_input": 2,
    }
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = ValidateConfig.from_yaml(cfg_path)

    assert cfg.target_skill == skill_path
    assert len(cfg.inputs) == 2
    assert isinstance(cfg.inputs[0], TestInput)
    assert cfg.inputs[0].name == "first"
    assert len(cfg.evals) == 1
    assert isinstance(cfg.evals[0], Eval)
    assert cfg.evals[0].name == "polite"
    assert cfg.judge_reruns == 5
    assert cfg.runs_per_input == 2
    assert cfg.generator_model == "claude-sonnet-4-6"


def test_validate_config_defaults(tmp_path: Path) -> None:
    """Optional fields fall back to documented defaults."""
    payload = {
        "target_skill": str(tmp_path / "skill.md"),
        "inputs": [{"name": "x", "prompt": "y"}],
        "evals": [{"name": "e", "question": "q?"}],
    }
    cfg = ValidateConfig.model_validate(payload)
    assert cfg.judge_reruns == 3
    assert cfg.runs_per_input == 1
    assert cfg.generator_model == "claude-sonnet-4-6"
    assert cfg.judge_model == "claude-sonnet-4-6"
    assert cfg.evals[0].pass_condition == ""
    assert cfg.evals[0].fail_condition == ""


def test_validate_config_rejects_empty_evals(tmp_path: Path) -> None:
    """``evals`` is constrained to min_length=1."""
    payload = {
        "target_skill": str(tmp_path / "skill.md"),
        "inputs": [{"name": "x", "prompt": "y"}],
        "evals": [],
    }
    with pytest.raises(ValidationError):
        ValidateConfig.model_validate(payload)


def test_validate_config_rejects_too_many_evals(tmp_path: Path) -> None:
    """``evals`` is constrained to max_length=8."""
    payload = {
        "target_skill": str(tmp_path / "skill.md"),
        "inputs": [{"name": "x", "prompt": "y"}],
        "evals": [{"name": f"e{i}", "question": "q?"} for i in range(9)],
    }
    with pytest.raises(ValidationError):
        ValidateConfig.model_validate(payload)


def test_output_key_helper() -> None:
    """Output.key matches the f-string used by JudgeRun.output_key."""
    out = Output(input_name="alpha", run_index=2, text="hi", model="m")
    assert out.key == "alpha__2"


def test_judge_run_roundtrip() -> None:
    """JudgeRun model_dump round-trips through model_validate."""
    jr = JudgeRun(
        output_key="alpha__0",
        eval_name="polite",
        rater="judge_1",
        verdict=True,
        reasoning="ok",
    )
    again = JudgeRun.model_validate(jr.model_dump())
    assert again == jr
