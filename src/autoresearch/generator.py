"""Generator: runs the target Claude Code skill on a TestInput."""

from __future__ import annotations

from pathlib import Path

from .client import AnthropicClient
from .schemas import Output, TestInput


class Generator:
    """Loads a skill markdown file as system prompt and runs test inputs through it."""

    def __init__(self, client: AnthropicClient, skill_path: Path, model: str) -> None:
        """Capture the client, skill path, and model to use for completions."""
        self._client = client
        self._skill_path = Path(skill_path)
        self._model = model
        self._system_prompt: str | None = None

    @property
    def skill_path(self) -> Path:
        """Return the current skill path."""
        return self._skill_path

    @skill_path.setter
    def skill_path(self, value: Path) -> None:
        """Re-point the generator at a new skill file and reset the cache."""
        self._skill_path = Path(value)
        self._system_prompt = None

    def set_skill_text(self, text: str) -> None:
        """Override the cached system prompt directly (skips the disk read).

        Used by the optimizer loop to score arbitrary skill bodies without
        having to write them to a file first.
        """
        self._system_prompt = text

    def _load_skill(self) -> str:
        """Read and cache the skill markdown file."""
        if self._system_prompt is None:
            self._system_prompt = self._skill_path.read_text(encoding="utf-8")
        return self._system_prompt

    def run(self, test_input: TestInput, run_index: int = 0) -> Output:
        """Send ``test_input.prompt`` through the skill and return an Output."""
        system = self._load_skill()
        text = self._client.complete(
            system=system,
            user=test_input.prompt,
            model=self._model,
        )
        return Output(
            input_name=test_input.name,
            run_index=run_index,
            text=text,
            model=self._model,
        )
