"""Tests for autoresearch.judge (mocked — no live API calls)."""

from __future__ import annotations

from pytest_mock import MockerFixture

from autoresearch.client import AnthropicClient
from autoresearch.judge import Judge
from autoresearch.schemas import Eval, Output


def _make_judge(mocker: MockerFixture, response: str) -> tuple[Judge, object]:
    """Build a Judge whose underlying client returns ``response``."""
    mocker.patch("autoresearch.client.anthropic.Anthropic")
    fake_complete = mocker.patch.object(AnthropicClient, "complete", return_value=response)
    client = AnthropicClient(api_key="test-key")
    return Judge(client=client, model="claude-sonnet-4-6"), fake_complete


def _sample_pair() -> tuple[Output, Eval]:
    output = Output(
        input_name="greeting",
        run_index=0,
        text="Hello there, partner!",
        model="claude-sonnet-4-6",
    )
    criterion = Eval(
        name="polite",
        question="Is the response polite?",
        pass_condition="uses friendly tone",
        fail_condition="rude or hostile",
    )
    return output, criterion


def test_judge_parses_yes(mocker: MockerFixture) -> None:
    """A YES first line yields verdict=True and captures the reasoning."""
    judge, _ = _make_judge(mocker, "YES\nthe output meets criterion.")
    output, criterion = _sample_pair()

    run = judge.score(output, criterion, rater_label="judge_1")

    assert run.verdict is True
    assert run.output_key == "greeting__0"
    assert run.eval_name == "polite"
    assert run.rater == "judge_1"
    assert run.reasoning == "the output meets criterion."


def test_judge_parses_no(mocker: MockerFixture) -> None:
    """A NO first line yields verdict=False."""
    judge, _ = _make_judge(mocker, "NO\nfails because tone is rude.")
    output, criterion = _sample_pair()

    run = judge.score(output, criterion, rater_label="judge_2")

    assert run.verdict is False
    assert run.reasoning == "fails because tone is rude."


def test_judge_parse_failure_defaults_to_false(mocker: MockerFixture) -> None:
    """Unrecognized output → verdict=False with parse_failed reasoning."""
    judge, _ = _make_judge(mocker, "garbage")
    output, criterion = _sample_pair()

    run = judge.score(output, criterion, rater_label="judge_3")

    assert run.verdict is False
    assert run.reasoning.startswith("parse_failed")


def test_judge_user_prompt_includes_question_and_output(mocker: MockerFixture) -> None:
    """The user prompt sent to the model includes the eval question + the output text."""
    judge, fake_complete = _make_judge(mocker, "YES\nlooks good")
    output, criterion = _sample_pair()

    judge.score(output, criterion, rater_label="judge_1")

    fake_complete.assert_called_once()
    kwargs = fake_complete.call_args.kwargs
    user_prompt = kwargs["user"]
    assert criterion.question in user_prompt
    assert output.text in user_prompt
    assert criterion.pass_condition in user_prompt
    assert criterion.fail_condition in user_prompt
    # System prompt is the fixed binary-criterion instruction.
    assert "YES" in kwargs["system"] and "NO" in kwargs["system"]
    assert kwargs["model"] == "claude-sonnet-4-6"
