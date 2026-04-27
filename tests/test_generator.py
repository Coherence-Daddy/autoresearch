"""Tests for autoresearch.generator (mocked — no live API calls)."""

from __future__ import annotations

from pathlib import Path

from pytest_mock import MockerFixture

from autoresearch.client import AnthropicClient
from autoresearch.generator import Generator
from autoresearch.schemas import Output, TestInput


def test_generator_run_uses_skill_as_system_prompt(tmp_path: Path, mocker: MockerFixture) -> None:
    """``Generator.run`` should send the SKILL.md content as the system prompt."""
    skill_text = "# Skill\n\nYou are a helpful test skill."
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text(skill_text, encoding="utf-8")

    fake_complete = mocker.patch.object(AnthropicClient, "complete", return_value="generated text")
    # Avoid hitting real env / SDK construction.
    mocker.patch("autoresearch.client.anthropic.Anthropic")

    client = AnthropicClient(api_key="test-key")
    gen = Generator(client=client, skill_path=skill_path, model="claude-sonnet-4-6")

    test_input = TestInput(name="greeting", prompt="say hi")
    out = gen.run(test_input, run_index=0)

    fake_complete.assert_called_once()
    kwargs = fake_complete.call_args.kwargs
    assert kwargs["system"] == skill_text
    assert kwargs["user"] == "say hi"
    assert kwargs["model"] == "claude-sonnet-4-6"

    assert isinstance(out, Output)
    assert out.input_name == "greeting"
    assert out.run_index == 0
    assert out.text == "generated text"
    assert out.model == "claude-sonnet-4-6"


def test_generator_run_index_propagates(tmp_path: Path, mocker: MockerFixture) -> None:
    """run_index passed to .run() flows through to the resulting Output."""
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("system", encoding="utf-8")

    mocker.patch.object(AnthropicClient, "complete", return_value="x")
    mocker.patch("autoresearch.client.anthropic.Anthropic")

    client = AnthropicClient(api_key="test-key")
    gen = Generator(client=client, skill_path=skill_path, model="m")

    out = gen.run(TestInput(name="t", prompt="p"), run_index=4)
    assert out.run_index == 4
    assert out.key == "t__4"
