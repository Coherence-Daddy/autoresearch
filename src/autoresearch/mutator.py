"""Mutator: ask an LLM for ONE targeted edit to a skill prompt."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .client import AnthropicClient
from .schemas import Eval, JudgeRun, Output

_SYSTEM = (
    "You are an expert prompt engineer. You will be shown a Claude Code skill "
    "(a markdown prompt file) and a small set of test cases where the skill failed. "
    "Propose ONE targeted edit to the skill that you believe will fix the most "
    "common failure pattern.\n\n"
    "Rules:\n"
    "- Make exactly one focused change. Do not rewrite the whole skill.\n"
    "- The change should be specific and address a concrete failure pattern, "
    "not a vague 'be better' instruction.\n"
    "- Preserve the existing frontmatter and overall structure.\n"
    "- Reply in this exact format and nothing else:\n\n"
    "DESCRIPTION: <one sentence describing what you changed and why>\n\n"
    "NEW_SKILL:\n"
    "```markdown\n"
    "<the FULL updated skill markdown, including frontmatter>\n"
    "```"
)


@dataclass(frozen=True)
class Mutation:
    """A proposed edit to a skill prompt."""

    description: str
    new_skill_text: str


class Mutator:
    """Calls a model to propose one targeted edit per call."""

    def __init__(self, client: AnthropicClient, model: str) -> None:
        self._client = client
        self._model = model

    def propose(
        self,
        current_skill: str,
        failures: list[tuple[Output, Eval, JudgeRun]],
        *,
        history: list[str] | None = None,
        force_divergent: bool = False,
    ) -> Mutation:
        """Return a single Mutation proposed by the model.

        ``failures`` is a list of (output, eval, judge_run) where the judge
        rated the output as failing the eval. Includes the judge's reasoning.

        ``history`` is the running list of "[status] description" strings from
        prior experiments, so the mutator avoids re-proposing tried ideas.

        ``force_divergent`` instructs the mutator to deliberately try a
        different angle (used after several stuck experiments).
        """
        user = _build_user_prompt(current_skill, failures, history or [], force_divergent)
        text = self._client.complete(
            system=_SYSTEM,
            user=user,
            model=self._model,
            max_tokens=4096,
            temperature=1.0,
        )
        return _parse(text, fallback=current_skill)


def _build_user_prompt(
    current_skill: str,
    failures: list[tuple[Output, Eval, JudgeRun]],
    history: list[str],
    force_divergent: bool,
) -> str:
    if not failures:
        body = (
            "No failures observed. Propose ONE small clarification or anti-pattern "
            "that would harden the skill against an unobserved failure mode."
        )
    else:
        chunks = []
        for i, (output, eval_criterion, judge_run) in enumerate(failures[:8], start=1):
            criterion_line = (
                f"Eval: {eval_criterion.name} — {eval_criterion.question}\n"
                f"Pass if: {eval_criterion.pass_condition or '(unspecified)'}\n"
                f"Fail if: {eval_criterion.fail_condition or '(unspecified)'}\n"
            )
            chunks.append(
                f"--- FAILURE {i} ---\n"
                f"{criterion_line}"
                f"Judge reason: {judge_run.reasoning}\n"
                f"Output text:\n{output.text}\n"
            )
        body = "\n".join(chunks)

    history_block = ""
    if history:
        joined = "\n".join(f"  - {h}" for h in history[-12:])
        history_block = (
            f"\nPRIOR ATTEMPTS (most recent last) — do not propose the same idea again:\n{joined}\n"
        )

    divergence_note = ""
    if force_divergent:
        divergence_note = (
            "\nDIVERGENCE MODE: prior attempts have not improved the score. "
            "Propose an edit that takes a clearly different angle from the prior attempts — "
            "consider removing or restructuring an existing instruction, adding a worked "
            "example, or reordering sections by priority. Do not just rephrase prior tries.\n"
        )

    return (
        f"CURRENT SKILL:\n```markdown\n{current_skill}\n```\n\n"
        f"OBSERVED FAILURES:\n{body}\n"
        f"{history_block}"
        f"{divergence_note}"
        "\nPropose one targeted edit per the rules in the system prompt."
    )


_SKILL_FENCE = re.compile(r"```(?:markdown)?\s*\n(.*?)\n```", re.DOTALL)


def _parse(text: str, *, fallback: str) -> Mutation:
    """Extract DESCRIPTION + NEW_SKILL from the model response.

    On parse failure, return the fallback skill unchanged with a description
    flagging the parse error so the caller can revert cleanly.
    """
    desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?:\n\n|\nNEW_SKILL)", text, re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else "parse_failed: no DESCRIPTION"

    skill_match = _SKILL_FENCE.search(text)
    if skill_match is None:
        return Mutation(
            description=f"parse_failed: no markdown fence found ({description!r})",
            new_skill_text=fallback,
        )
    new_skill = skill_match.group(1).strip()
    if not new_skill:
        return Mutation(description="parse_failed: empty skill body", new_skill_text=fallback)
    return Mutation(description=description, new_skill_text=new_skill)
