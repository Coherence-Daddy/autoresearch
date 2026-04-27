# autoresearch

> Phase 1 deliverable: a **judge-validation gate** for Claude Code skill optimization.

Before optimizing a skill, you have to know your scoring instrument works. This tool measures two things:

1. **Test–retest reliability** — does the LLM judge give the same answer on the same input twice? (Fleiss' kappa across K reruns.)
2. **Judge–human agreement** — does the LLM judge agree with you when you grade the same outputs by hand? (Cohen's kappa + percent agreement.)

If both gates pass (kappa ≥ 0.7, agreement ≥ 0.8) the methodology is worth building on. If either fails, the optimizer is built on sand — stop and ship the negative-result writeup instead.

## Why this exists

The full autoresearch spec ([`docs/specs/2026-04-27-autoresearch.md`](docs/specs/2026-04-27-autoresearch.md)) was reviewed by an LLM council ([`council-transcript-2026-04-27-140738.md`](council-transcript-2026-04-27-140738.md)). Verdict: don't build the optimizer until the judge is validated. This package is the validation gate.

## Install

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run the gate

```bash
autoresearch validate examples/sample-config.yaml
```

The CLI:
1. Generates outputs for each test input using the target skill (or loads pre-existing outputs).
2. Re-runs the judge K times per (output, eval) pair.
3. Prompts you to hand-grade each output.
4. Prints a report: kappa, agreement, pass/fail per gate.

## What it does NOT do

- Mutate prompts. (Phase 2.)
- Optimize anything. (Phase 2.)
- Run a dashboard. (Phase 2.)

If Phase 1 passes, Phase 2 builds on this same package. If Phase 1 fails, this package becomes the artifact for the negative-result episode.
