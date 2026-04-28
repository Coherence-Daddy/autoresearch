---
name: autoresearch
description: "Use this skill to test, optimize, or compare Claude Code skill files using the autoresearch tool. Covers setup, writing eval configs, running the optimizer, and interpreting results."
---

# Autoresearch: Test and Optimize Skill Files

The autoresearch tool measures whether a skill file actually causes the AI to do what it says — and automatically finds improvements when it doesn't.

## Repo location

```
/Users/exe/Downloads/Claude/CD-skill-research/
```

## Setup

The repo has a virtual environment already configured:

```bash
cd /Users/exe/Downloads/Claude/CD-skill-research
source .venv/bin/activate
```

`ANTHROPIC_API_KEY` is loaded automatically from `.env` in the repo root.

## Three commands

### 1. `validate` — check if your judge is trustworthy

Before optimizing, confirm the judge gives consistent results:

```bash
autoresearch validate plan-config.yaml
```

Runs the judge multiple times on the same outputs and asks you to hand-grade some. Reports Fleiss' kappa (test-retest) and Cohen's kappa (judge vs human). Both should be ≥ 0.7 before you trust the optimizer's scores.

### 2. `optimize` — automatically improve a skill

```bash
autoresearch optimize plan-config.yaml
```

Runs up to 20 mutation experiments. Each experiment:
1. Mutates the current best skill (using Opus)
2. Scores it on train inputs (using Haiku as the "student" and judge)
3. Keeps it if score improves, discards if not
4. Checks holdout inputs every 5 experiments to detect overfitting

Saves results to `runs/<skill-name>-<timestamp>/`:
- `SKILL.md.baseline` — original skill
- `SKILL.md.best` — best version found
- `log.jsonl` — per-experiment scores and descriptions
- `snapshots/` — each kept skill version

### 3. `compare` — head-to-head stability test

```bash
autoresearch compare plan-config.yaml skill-a.md skill-b.md --holdout --repeat 5
```

Scores two (or more) skills side-by-side. `--repeat 5` runs 5 independent rounds and reports mean ± stdev. Reports a z-statistic: **z ≥ 2 means the lift is real, not noise**.

Always use `--repeat 5` before trusting a result. Single-shot comparisons are dominated by generator temperature variance and will lie to you.

## Writing an eval config

An eval config is a YAML file that tells autoresearch what to test. See `plan-config.yaml` for a full example. Key fields:

```yaml
target_skill: plan-skill.md        # skill file to optimize
generator_model: claude-haiku-4-5-20251001  # "student" model
judge_model: claude-haiku-4-5-20251001      # grader model
mutator_model: claude-opus-4-7              # mutation proposer
runs_per_input: 3                  # average over N generations (reduces noise)

inputs:                            # 2 train inputs minimum
  - name: my_test
    prompt: "The prompt you'd give the skill in real use."

holdout:                           # 4 inputs that optimizer never sees
  - name: generalization_test
    prompt: "A different prompt to check the fix actually generalizes."

evals:                             # 2-5 binary pass/fail criteria
  - name: has_required_section
    question: "Does the output contain X?"
    pass_condition: "Describe what a passing output looks like."
    fail_condition: "Describe what a failing output looks like."
```

**Eval design rules that matter:**
- Keep evals binary — yes/no, not scored 1-10
- At least one eval should be locally satisfiable (pass/fail from a short output fragment). Global properties like "every step has X" are harder for the optimizer to fix.
- If the baseline scores 1.0 on all evals, there's nothing to optimize — make the evals harder
- If the baseline scores 0.0 on all evals, the evals or prompts are broken — fix those first

## Interpreting results

| Signal | What it means |
|---|---|
| Baseline train ≈ holdout | Evals are measuring something real |
| Baseline train >> holdout | Train inputs may be too easy / narrow |
| All mutations discard | Mutator is stuck; usually means evals need a locally-fixable failure mode |
| `keep` with description mentioning "remove" | Often the best fixes — less verbose skills perform better |
| `keep` with description mentioning "add constraint" | Second most common win — explicit forbidden-format lists, output budgets |
| z ≥ 2 | Lift is statistically real |
| z ≥ 5 | Very strong signal |
| stdev = 0.000 on best | Perfect consistency — as clean as it gets |

## Existing results

Two real skills have been validated end-to-end:

**Tweet skill** (`examples/sample-config.yaml`)
- Baseline → best: +16pp holdout, z=3.22
- Fix: banned emojis inflating byte counts

**Plan skill** (`plan-config.yaml`)
- Baseline → best: 0.388 → 1.000 holdout, z=19.56
- Fix: removed verbose self-review section; added output budget with hard caps + self-check recovery instruction; added mandatory step format template with forbidden-format list
- The baseline **never** produced `every_step_has_verify` or `has_done_when` on holdout inputs (0/12). The optimized version is perfect (12/12) with zero variance across 5 rounds.

## Common failure modes

**Mutator gets stuck proposing process changes**
The mutator may spend many experiments on "write the file in reverse order" or "write a skeleton first" when the real fix is a declarative output constraint ("your response MUST end with X"). This happens when the failing eval requires a global output property. If you see 5+ consecutive discards all describing process changes, the evals need a locally-fixable failure mode added.

**Single-shot compare results are unreliable**
Generator temperature means two runs of the same skill on the same input can score differently. Always use `--repeat 5`. A result without repeat is anecdote, not evidence.

**No-op mutations near the end of a run**
If the mutator's output file gets truncated (e.g., the skill it produces is identical to the current best), the optimizer logs a no-op and skips holdout scoring for that experiment. This can cause the final holdout check to be skipped. Always run `compare --holdout --repeat 5` after `optimize` to get the definitive generalization verdict.
