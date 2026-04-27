"""Judge: binary YES/NO scorer for an Output against an Eval criterion."""

from __future__ import annotations

from .client import AnthropicClient
from .schemas import Eval, JudgeRun, Output

_SYSTEM_PROMPT = (
    "You evaluate outputs against a single binary criterion. "
    "Reply with exactly 'YES' or 'NO' on the first line, then on the next line "
    "one short sentence of reasoning. No preamble, no markdown."
)


def _build_user_prompt(output: Output, eval_criterion: Eval) -> str:
    """Render the user-facing judge prompt."""
    return (
        f"CRITERION: {eval_criterion.question}\n\n"
        f"PASS IF: {eval_criterion.pass_condition}\n"
        f"FAIL IF: {eval_criterion.fail_condition}\n\n"
        "--- OUTPUT TO EVALUATE ---\n"
        f"{output.text}\n"
        "--- END OUTPUT ---"
    )


def _parse_verdict(raw: str) -> tuple[bool, str]:
    """Parse the judge response into (verdict, reasoning).

    First non-empty line decides the verdict. Anything that doesn't start with
    YES or NO falls through to ``False`` with a parse_failed reasoning string.
    """
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return False, f"parse_failed: {raw[:100]}"

    first = lines[0]
    rest = " ".join(lines[1:]).strip()

    upper = first.upper()
    if upper.startswith("YES"):
        return True, rest
    if upper.startswith("NO"):
        return False, rest
    return False, f"parse_failed: {raw[:100]}"


class Judge:
    """Asks an LLM judge to score outputs against a binary criterion."""

    def __init__(self, client: AnthropicClient, model: str) -> None:
        """Capture the client and judge model name."""
        self._client = client
        self._model = model

    def score(self, output: Output, eval_criterion: Eval, rater_label: str) -> JudgeRun:
        """Score ``output`` against ``eval_criterion`` and return a JudgeRun."""
        user_prompt = _build_user_prompt(output, eval_criterion)
        raw = self._client.complete(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            model=self._model,
        )
        verdict, reasoning = _parse_verdict(raw)
        return JudgeRun(
            output_key=f"{output.input_name}__{output.run_index}",
            eval_name=eval_criterion.name,
            rater=rater_label,
            verdict=verdict,
            reasoning=reasoning,
        )
