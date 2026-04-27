# Autoresearch — Autonomous Skill Optimization Tool

## Problem

Most Claude Code skills work ~70% of the time. The remaining failure rate is invisible — there's no measurement loop, so improvements are vibes-driven and regressions ship silently. Skill authors need a way to (a) measure a skill's actual pass rate against binary criteria, (b) iterate on the prompt with evidence, and (c) prove the resulting skill generalizes beyond the inputs they tested.

## Goal

Ship a standalone tool — invoked via a Claude Code skill — that takes any target SKILL.md, a small set of test inputs, and a binary eval suite, then runs an autonomous experimentation loop that produces a measurably-better SKILL.md, a complete changelog, a holdout-validated score, and a live HTML dashboard.

## Approach

**Standalone repo `autoresearch`** containing a Python CLI harness + a thin SKILL.md that lives in cd-skills (linked after the harness ships).

The CLI does the heavy lifting: model calls, scoring, mutation, dashboard. The skill is the front door — it gathers context from the user, writes the config file, then shells out to the CLI.

Per-role configurable model backends. Default routing:

- **Generator** (runs the target skill on each test input) → Ollama VPS via OpenAI-compatible client.
- **Judge** (binary eval scorer) → Ollama cloud free tier.
- **Mutator** (proposes the next SKILL.md edit) → Anthropic API.

All three roles are swappable via `config.yaml` (Ollama VPS / Ollama cloud / HF Inference / Anthropic). The CLI ships with a single OpenAI-compatible HTTP client that talks to all four (Anthropic via its native SDK).

Execution mode is hybrid:

- **Mode A (default, pure-prompt):** Load SKILL.md as a system prompt, send the test input as a user message, capture the assistant text. Works for skills whose output *is text* (specs, copy, plans, evals, designs).
- **Mode B (escape hatch, headless Claude Code):** Spawn `claude -p "<input>" --output-format json` in a temp directory with the skill installed. Captures real tool use. Used when the target skill is side-effecting. Declared per-skill in config.

Mutation strategy is **greedy single-mutation with periodic shake-up**. One targeted edit per experiment, kept if score rises, reverted otherwise. After N stuck experiments (default 10) the mutator is asked for a deliberately divergent edit (remove longest section, reorder sections, etc.) to escape local maxima.

Overfitting guard is **train/holdout split**. User provides a train set; optimizer mutates against it. User optionally provides a holdout set; if missing, the harness auto-generates one via the mutator model at setup time. Holdout is scored every K experiments (default 5) and at run end. Dashboard plots train and holdout pass rates as separate lines. If train climbs while holdout flattens for 3 consecutive checkpoints, raise an "overfitting" flag and stop.

**Rejected alternatives:**

- *Skill-internal autoresearch (no external harness)* — same agent generating, judging, and mutating biases scoring; can't run unattended; can't use cheap/local models for volume work.
- *All-Ollama, single model* — mutator quality bottlenecks the whole loop; you stop improving at whatever your local model's ceiling is.
- *Population/beam search mutation* — better exploration but the changelog stops being human-readable; greedy + shake-up captures most of the benefit.
- *No overfitting guard* — the spec's stated goal of "didn't overfit" without a mechanism is wishful thinking; with 5 train inputs the optimizer will memorize them.

## Design

### Repository layout

```
autoresearch/
├── README.md
├── LICENSE
├── pyproject.toml                     # Python package, CLI entry point
├── src/autoresearch/
│   ├── __init__.py
│   ├── cli.py                         # `autoresearch run`, `autoresearch dashboard`, `autoresearch resume`
│   ├── config.py                      # YAML loader, schema validation
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── openai_compatible.py       # talks to Ollama VPS / Ollama cloud / HF
│   │   └── anthropic.py               # native Anthropic SDK
│   ├── roles.py                       # Generator, Judge, Mutator — all dispatch to the right client per config
│   ├── modes/
│   │   ├── pure_prompt.py             # Mode A
│   │   └── headless_cc.py             # Mode B (subprocess to `claude` CLI)
│   ├── eval_suite.py                  # parses + applies binary evals
│   ├── loop.py                        # the experiment loop, shake-up, holdout checkpoints
│   ├── holdout.py                     # auto-generation of holdout inputs
│   ├── persistence.py                 # writes results.tsv, results.json, changelog.md, baseline backup
│   └── dashboard.py                   # generates dashboard.html (self-contained, Chart.js from CDN)
├── tests/
│   ├── test_eval_suite.py
│   ├── test_loop.py
│   ├── test_clients.py                # using `respx`/`vcrpy` cassettes
│   └── fixtures/
│       └── toy_skill/                 # a deliberately-flawed SKILL.md used to verify the loop converges
└── examples/
    └── config.example.yaml
```

### Config file (`config.yaml`)

```yaml
target_skill: ~/.claude/skills/diagram-generator/SKILL.md
mode: pure_prompt                     # or headless_cc

inputs:
  train:
    - "OAuth flow diagram"
    - "CI/CD pipeline"
    - "microservices architecture"
  holdout: auto                       # or a list of strings

evals:
  - name: legibility
    question: "Is all text fully legible with no truncated or overlapping words?"
    pass: "Every word complete and readable"
    fail: "Any word hidden, overlapping, or cut off"
  # ... 3-6 total

runs_per_experiment: 5
shake_up_after: 10                    # stuck experiments before forced divergent mutation
holdout_every: 5                      # checkpoint frequency
budget_cap: null                      # or an integer
stop_at_pass_rate: 0.95               # 95% for 3 consecutive experiments

models:
  generator:
    provider: ollama_vps
    base_url: https://your-vps/v1
    model: gemma3:27b
    api_key_env: OLLAMA_VPS_KEY       # optional
  judge:
    provider: ollama_cloud
    base_url: https://ollama.com/v1
    model: gpt-oss:120b
    api_key_env: OLLAMA_CLOUD_KEY
  mutator:
    provider: anthropic
    model: claude-opus-4-7
    api_key_env: ANTHROPIC_API_KEY
```

### CLI surface

```
autoresearch init <target_skill>      # interactive: writes config.yaml + auto-generates holdout if asked
autoresearch run [config.yaml]        # runs the loop; opens dashboard; resumable
autoresearch resume <run_dir>         # picks up from last experiment
autoresearch dashboard <run_dir>      # opens the HTML dashboard for an existing run
autoresearch report <run_dir>         # prints final summary to stdout
```

### Data flow per experiment

```
                    ┌─────────────────────────────────────┐
                    │  current SKILL.md (under mutation)  │
                    └──────────────────┬──────────────────┘
                                       │
                          ┌────────────┴────────────┐
                          │                         │
                     train inputs              holdout inputs
                     (every exp)            (every K experiments)
                          │                         │
                          ▼                         ▼
                  ┌───────────────┐        ┌───────────────┐
                  │   Generator   │        │   Generator   │
                  │  (mode A or B)│        │  (mode A or B)│
                  │   N runs each │        │   N runs each │
                  └───────┬───────┘        └───────┬───────┘
                          │                        │
                          ▼                        ▼
                  ┌───────────────┐        ┌───────────────┐
                  │     Judge     │        │     Judge     │
                  │ binary scoring│        │ binary scoring│
                  └───────┬───────┘        └───────┬───────┘
                          │                        │
                          ▼                        ▼
                   train_score                holdout_score
                          │                        │
                          └────────────┬───────────┘
                                       ▼
                          ┌────────────────────────┐
                          │   keep / discard /     │
                          │   shake-up / stop      │
                          └────────────┬───────────┘
                                       ▼
                              persistence + dashboard
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │       Mutator          │
                          │  (fires every          │
                          │   experiment; reads    │
                          │   failing outputs +    │
                          │   changelog so far;    │
                          │   forced-divergent     │
                          │   on shake-up)         │
                          └────────────┬───────────┘
                                       │
                                       ▼
                              next SKILL.md edit
```

### Output directory (per run)

```
autoresearch-runs/<skill-name>-<timestamp>/
├── dashboard.html              # self-contained, auto-refresh every 10s
├── results.json                # data file the dashboard polls
├── results.tsv                 # human-readable experiment log
├── changelog.md                # detailed mutation log (the most valuable artifact)
├── SKILL.md.baseline           # the original
├── SKILL.md.experiments/       # snapshot per kept experiment, for diffs/rollback
│   ├── 0001.md
│   ├── 0003.md
│   └── ...
├── outputs/                    # raw model outputs per experiment, for re-scoring later
│   └── exp-<N>/run-<M>/<input-slug>.txt
└── config.snapshot.yaml        # frozen copy of the config used
```

### Dashboard

Single-file HTML. Auto-refreshes by polling `results.json` every 10s. Sections:

- **Status banner** — "Running experiment N" / "Idle" / "Complete" / "Overfitting flagged — stopped".
- **Score progression** — Chart.js line chart with two series: train pass rate and holdout pass rate. Holdout drawn dashed since it's checkpointed, not continuous.
- **Experiment table** — id, train score, holdout score (if checkpointed at this experiment), status (baseline/keep/discard/shake-up), one-line description of the mutation.
- **Per-eval breakdown** — for each eval, count of passes / total runs across the most recent experiment + sparkline over time.
- **Final summary** (when `status: complete`) — baseline → final delta, total experiments, keep rate, top 3 most-impactful mutations linked to changelog entries.

Styling: white background, soft pastels (matches your existing brand from cd-skills/give-obsidian-a-memory aesthetic), one clean sans-serif. No emoji.

### Skill front-door (lives in cd-skills *after* the harness is shipped and tested)

A short SKILL.md that:

1. Confirms `autoresearch` is installed (checks `which autoresearch`); if not, prints install command and stops.
2. Gathers the seven config inputs interactively (target skill path, train inputs, evals, runs/experiment, shake-up threshold, budget cap, holdout strategy).
3. Writes `config.yaml` to `autoresearch-runs/<skill-name>-<timestamp>/`.
4. Shells out: `autoresearch run <config>`.
5. Opens the dashboard.
6. Hands control to the CLI — does NOT try to drive the loop from inside the Claude Code session.

This separation matters: the loop runs for hours; a Claude Code session shouldn't be the parent process.

### Resumability

The loop writes `state.json` after every experiment (current SKILL.md hash, last kept experiment id, RNG seed, stuck-counter). `autoresearch resume <run_dir>` picks up from the last completed experiment. Crashes or Ctrl-C don't lose progress.

### Cost / rate-limit handling

- Per-role rate-limit + retry-with-backoff via `tenacity`.
- A `--dry-run` flag estimates total tokens and dollars from the config + a single experiment's worth of calls before committing.
- Mutator fires once per experiment (the cheapest of the three roles by call volume — one call vs `runs × evals` for the judge), so cost stays dominated by judge throughput. The Anthropic-mutator default is justified on quality, not frequency.

## Edge cases

- **Target skill references files in `references/`** — pure-prompt mode prepends linked reference files to the system prompt automatically (skill author's intent preserved).
- **Mode B and Ollama** — if `mode: headless_cc` and generator points at Ollama, the harness sets `ANTHROPIC_BASE_URL` env for the spawned `claude` process to route through the user's Ollama setup (matches `use-ollama-to-enhance-claude` pattern).
- **Judge inconsistency** — if the same output gets different scores on re-evaluation (>10% disagreement), surface a "judge instability" warning. Fix is upstream (better evals or stronger judge model).
- **Mutator proposes a no-op edit** — detected by hash comparison; counts as a stuck experiment without burning a generator pass.
- **All evals already pass at baseline** — print "skill already at ceiling for these evals; either lower the bar or write harder evals" and exit without mutating.
- **Holdout auto-generation fails** — fall back to "no holdout, train-only run" with a clear warning printed and logged.
- **Network drops mid-experiment** — current experiment's results discarded, retry from start of experiment N (not from run start).
- **Two simultaneous runs on the same skill** — `state.json` lock file; second invocation refuses with a clear error.

## Out of scope

- Optimizing skills *automatically* on a schedule (cron / GitHub Action). Manual invocation only in v1.
- A web UI for editing the SKILL.md directly. Dashboard is read-only.
- Population-based mutation (beam search, genetic, etc.). Greedy + shake-up only.
- Cross-skill optimization (training one skill against another's output). Single-skill only.
- Auto-generating eval criteria from a skill description. The user writes the evals (with help from the existing eval-guide).
- Pricing / billing dashboards beyond the `--dry-run` estimator.
- Public registry of optimized skills. Each user runs their own.
- Plugin-publishing automation. The cd-skills front-door SKILL.md is added by hand after the harness is verified.

## Success criteria

1. **Convergence on a known-bad fixture.** `tests/fixtures/toy_skill/` is a deliberately-flawed skill (e.g., generates output that fails 2 of 4 evals at baseline). Running autoresearch against it for ≤20 experiments must improve the train score by ≥15 percentage points AND the holdout score by ≥10 percentage points. This is the integration test.
2. **Per-role backend routing works.** Three pytest cases mock the three role clients independently and verify the right model is called for each role given the config.
3. **Resumability.** Test: kill the process mid-experiment, resume, check that `results.tsv` has no duplicate or missing experiment ids and the final state matches an uninterrupted run on the same RNG seed.
4. **Dashboard is live.** Manual: run autoresearch, open `dashboard.html`, watch the line chart advance every 10 seconds without page reload.
5. **Overfitting flag fires.** Test: a synthetic scenario where train inputs are easy and holdout is hard. The harness must raise the overfitting flag and stop within 5 checkpoints of the divergence.
6. **Dry-run estimator within 20% of actual.** Run `--dry-run`, then a real 5-experiment run; estimated tokens must be within 20% of measured.
7. **Skill front-door integrates cleanly.** After the harness ships, the cd-skills SKILL.md gathers config, invokes the CLI, opens the dashboard. Exercised once on a real skill (e.g., `cd-skills/skills/brainstorm/SKILL.md`) end-to-end.
