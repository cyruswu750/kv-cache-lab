# CLAUDE.md — kv-cache-lab

Project-specific rules for working in this repository.

## Scope
- Implement only the milestone explicitly authorized in the current task.
  Core implementation, tests, scripts, and related documentation may all be
  edited within that scope without per-file approval.
- Briefly state intended changes, then proceed. Ask only when a consequential
  ambiguity or an action outside the authorized scope blocks progress.
- Do not add features, dependencies, or abstractions unrelated to the current
  milestone.
- Stop when the authorized milestone is complete.

## Sources of truth
- Read `kv-cache-plan.md` and `docs/design.md` if they exist. Do not invent
  missing documents or treat planned features as implemented.
- Explicit current user instructions override conflicting project documents.
  Record material design changes and their reasons (in `docs/decisions.md`).
  Otherwise, report unresolved design conflicts before changing behavior.

## Code
- Use simple, readable Python and PyTorch code. Follow the actual installed
  APIs (inspect them) and existing project conventions.
- Preserve correctness checks. Do not hide programming or CUDA errors behind
  broad cache-miss fallback handlers.

## Environment
- Use the isolated project environment (`.venv/`). Do not modify system
  packages, drivers, global environments, shell startup files, or unrelated
  projects. No sudo.
- Keep environments, model weights, credentials, and large generated files out
  of git.

## Testing and evidence
- Run relevant tests. Never weaken assertions or silently loosen numerical
  tolerances to obtain a passing result.
- Distinguish PASS, FAIL, BLOCKED, and NOT RUN. Never claim a test or benchmark
  succeeded without actual execution.
- Keep raw experimental evidence and reproducible commands. Clearly separate
  diagnostic timings from validated performance claims.
- Explain key algorithms, invariants, ownership, and edge cases in each
  milestone's walkthrough.

## Git and attribution
- Do not commit, push, create remotes, or tag releases unless requested.
- Defer AI-attribution wording until the user requests it at project
  completion; never claim AI-generated code was handwritten by the user.

## Handoff
- Finish each task with changed files, actual verification results, remaining
  limitations, and the next acceptance commands.
