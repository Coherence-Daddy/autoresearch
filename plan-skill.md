---
name: plan
description: "Turn a written spec into a checkbox plan. Use after cd:brainstorm or when the user provides a spec and wants the implementation broken into reviewable steps. Ends by handing off to cd:execute."
---

# Plan an Implementation

Take a written spec and produce a checkbox plan a human (or another agent) can execute step by step. Steps are sized for review — each completes in one bounded chunk of work.

## Hard gate

```
Do NOT begin implementation, run tests, or edit non-plan files until the
plan file is written, the user has approved it, and it is committed.
```

## Checklist

1. **Load the spec** — read the spec file the user references. If there isn't one, stop and invoke `cd:brainstorm`.
2. **Survey the surface** — read the files the spec touches. Note conventions: testing framework, file structure, naming.
3. **Decompose** — break the spec into ordered steps. See *Step sizing* below.
4. **Identify parallelism** — mark steps that can run in parallel (different agents, no shared state).
5. **Write the plan file** — `docs/plans/YYYY-MM-DD-<slug>.md`.
6. **Self-review** — see *Plan self-review*.
7. **User approves the file** — wait for written approval.
8. **Commit** — `git commit -m "plan: <topic>"`.
9. **Handoff** — invoke `cd:execute` (or `cd:parallel` if there are independent steps).

## Step sizing

Each step should:

- Be completable in one focused chunk (~15-60 minutes of agent time).
- End with a *verifiable* outcome — test passes, command succeeds, file diff is reviewable.
- Have one *primary file* it touches (or be explicit about why it touches several).

Too big: "Implement the auth system." Decompose.
Too small: "Add an import statement." Fold into the step that uses it.

## Plan file format

`docs/plans/YYYY-MM-DD-<slug>.md`:

```markdown
# Plan: <Title>

**Spec:** [docs/specs/...](../specs/...)

## Steps

- [ ] **1. <step title>**
  - **Files:** <primary file(s)>
  - **Action:** <one paragraph: what changes and why>
  - **Verify:** <how to confirm it worked — command, test, observation>
  - **Depends on:** <step IDs, or "none">
  - **Parallel-safe:** <yes / no>

- [ ] **2. <step title>**
  - ...

## Order

<rendered ordering — sequential where dependencies require it, parallel groups where they don't>

Group A (sequential): 1 → 2 → 3
Group B (parallel after step 3): 4, 5, 6
Final: 7

## Done when

<rolled-up success criteria from the spec, plus: all checkboxes ticked, all step `Verify` commands pass.>
```

## Plan self-review

Before handing off, scan the plan and fix:

- **No step has a `Verify` line.** Every step needs one.
- **Two steps both edit the same file with no ordering.** Pick an order or merge them.
- **A step's `Files` lists more than 3 files** without justification — that step is too big.
- **A step's `Action` says "refactor" or "improve"** without naming the change concretely.
- **Plan touches things outside the spec.** Either remove them or update the spec.

## Handoff

After the user approves the plan file:

- Commit: `git add docs/plans/<file> && git commit -m "plan: <topic>"`.
- Invoke `cd:execute` (or `cd:parallel` if `Order` shows parallel groups).

The terminal state is *invoking `cd:execute`* or *`cd:parallel`*. Do not start implementing inside this skill.
