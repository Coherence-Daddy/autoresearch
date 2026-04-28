# Autoresearch

> An LLM-as-judge optimizer for Claude Code skill files. Validates judge reliability, runs greedy mutation experiments, produces stability-tested results with z-stat verdicts.

**Real result:** Optimized the `cd:plan` skill from 0.388 → 1.000 on holdout inputs (z=19.56, stdev=0.000 across 5 rounds).

**Walkthrough:** [coherencedaddy.com/tutorials/autoresearch](https://coherencedaddy.com/tutorials/autoresearch) — 11-slide visual tutorial.

## Install

### As a Claude Code plugin

```
/plugin marketplace add Coherence-Daddy/autoresearch
/plugin install autoresearch@autoresearch
/reload-plugins
```

### As a standalone CLI

```bash
git clone https://github.com/Coherence-Daddy/autoresearch
cd autoresearch
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
```

## Three commands

```bash
autoresearch validate my-config.yaml             # check judge reliability
autoresearch optimize my-config.yaml             # run mutation loop
autoresearch compare my-config.yaml a.md b.md \
  --holdout --repeat 5                            # head-to-head with z-stat
```

## How it works

1. **Validate the judge** — Fleiss' kappa across K reruns, Cohen's kappa vs hand grading. Both ≥ 0.7 means the judge is trustworthy.
2. **Optimize** — Opus mutates the skill, Haiku scores it on train inputs, the loop keeps changes that improve the score and discards the rest.
3. **Compare** — `--repeat 5` runs the comparison five times to wash out generator-temperature noise. Reports mean, stdev, and a z-stat.

## Eval design rules

- Keep evals **binary** (yes/no, not scored 1-10).
- At least one eval must be **locally satisfiable** — pass/fail determinable from a short output fragment. Pure global properties give the mutator no gradient.
- If the baseline scores 1.000, there's nothing to optimize. Make the evals harder.
- If the baseline scores 0.000, the evals or prompts are broken. Fix those first.

## See also

- The skill file itself: [`skills/autoresearch/SKILL.md`](skills/autoresearch/SKILL.md)
- Sample configs: [`examples/sample-config.yaml`](examples/sample-config.yaml), [`plan-config.yaml`](plan-config.yaml)
- Methodology spec: [`docs/specs/2026-04-27-autoresearch.md`](docs/specs/2026-04-27-autoresearch.md)

## License

MIT
