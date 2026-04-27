"""Tests for the Mutator (mocked — no live API)."""

from __future__ import annotations

import pytest

from autoresearch.client import AnthropicClient
from autoresearch.mutator import Mutation, Mutator
from autoresearch.schemas import Eval, JudgeRun, Output


@pytest.fixture
def client(mocker):
    return mocker.MagicMock(spec=AnthropicClient)


def _failure() -> tuple[Output, Eval, JudgeRun]:
    output = Output(input_name="i1", run_index=0, text="too long output", model="m")
    eval_c = Eval(name="brevity", question="Is it short?", pass_condition="<10 chars")
    run = JudgeRun(
        output_key="i1__0",
        eval_name="brevity",
        rater="judge_1",
        verdict=False,
        reasoning="too long",
    )
    return output, eval_c, run


def test_propose_returns_parsed_mutation(client) -> None:
    client.complete.return_value = (
        "DESCRIPTION: tightened the brevity rule to a hard cap.\n\n"
        "NEW_SKILL:\n```markdown\n---\nname: x\n---\n# Hello\n\nKeep it under 10 chars.\n```"
    )
    mut = Mutator(client=client, model="claude-opus-4-7")
    result = mut.propose("---\nname: x\n---\n# Hello\n", failures=[_failure()])
    assert isinstance(result, Mutation)
    assert "tightened" in result.description
    assert "Keep it under 10 chars." in result.new_skill_text


def test_propose_handles_no_fence_gracefully(client) -> None:
    client.complete.return_value = "DESCRIPTION: ok\n\nNEW_SKILL: but no code fence here"
    mut = Mutator(client=client, model="m")
    result = mut.propose("original", failures=[])
    assert result.description.startswith("parse_failed")
    assert result.new_skill_text == "original"


def test_propose_handles_empty_skill_body(client) -> None:
    client.complete.return_value = "DESCRIPTION: ok\n\nNEW_SKILL:\n```markdown\n\n```"
    mut = Mutator(client=client, model="m")
    result = mut.propose("original", failures=[])
    assert result.description == "parse_failed: empty skill body"
    assert result.new_skill_text == "original"


def test_user_prompt_includes_failure_reasoning(client) -> None:
    client.complete.return_value = "DESCRIPTION: x\n\nNEW_SKILL:\n```markdown\nupdated body\n```"
    mut = Mutator(client=client, model="m")
    mut.propose("original", failures=[_failure()])
    sent_user = client.complete.call_args.kwargs["user"]
    assert "too long" in sent_user
    assert "brevity" in sent_user
    assert "original" in sent_user


def test_propose_no_failures_still_works(client) -> None:
    client.complete.return_value = (
        "DESCRIPTION: harden against unobserved failure.\n\n"
        "NEW_SKILL:\n```markdown\nhardened skill\n```"
    )
    mut = Mutator(client=client, model="m")
    result = mut.propose("original", failures=[])
    assert result.new_skill_text == "hardened skill"


def test_user_prompt_includes_history_when_provided(client) -> None:
    client.complete.return_value = (
        "DESCRIPTION: try a new angle.\n\nNEW_SKILL:\n```markdown\nv2\n```"
    )
    mut = Mutator(client=client, model="m")
    mut.propose(
        "original",
        failures=[_failure()],
        history=["[discard] added rule X", "[discard] added rule Y"],
    )
    sent_user = client.complete.call_args.kwargs["user"]
    assert "PRIOR ATTEMPTS" in sent_user
    assert "added rule X" in sent_user
    assert "added rule Y" in sent_user


def test_user_prompt_force_divergent_changes_instruction(client) -> None:
    client.complete.return_value = (
        "DESCRIPTION: divergent attempt.\n\nNEW_SKILL:\n```markdown\ndiv\n```"
    )
    mut = Mutator(client=client, model="m")
    mut.propose("original", failures=[_failure()], force_divergent=True)
    sent_user = client.complete.call_args.kwargs["user"]
    assert "DIVERGENCE MODE" in sent_user


def test_user_prompt_includes_eval_pass_fail_conditions(client) -> None:
    """Regression: previously the mutator received '(see eval suite)' instead of the question."""
    client.complete.return_value = "DESCRIPTION: ok.\n\nNEW_SKILL:\n```markdown\nx\n```"
    mut = Mutator(client=client, model="m")
    mut.propose("original", failures=[_failure()])
    sent_user = client.complete.call_args.kwargs["user"]
    assert "Is it short?" in sent_user
    assert "<10 chars" in sent_user
    assert "(see eval suite)" not in sent_user
