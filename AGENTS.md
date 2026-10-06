# Diffusion-ARC Agent Guidance

## Project context

- Read `README.md` before planning. It defines the authoritative project scope,
  architecture, milestones, and evaluation rules; changes require user approval.
- Preserve its training boundaries and protections against data leakage.
  Keep supporting research and benchmark context in `docs/`.
- Keep this guidance concise. Add nested `AGENTS.md` files only for genuinely
  different local workflows or constraints.

## Working agreement

For substantial implementations, architectural changes, or unclear scope:

1. **Discover:** inspect relevant files, repository state, and existing changes.
2. **Clarify:** collect material questions and ask them together.
3. **Propose:** describe the intended result, affected files, approach,
   validation, and relevant risks. Keep detail proportional to the change.
4. **Approve:** obtain approval for the proposal before implementation.
5. **Implement:** complete the approved scope without unnecessary pauses.
   Revisit approval when a material change in scope or risk arises.
6. **Verify and report:** run appropriate checks and report results and limits.

- Explicit requests for small, clearly scoped changes, such as comments,
  documentation edits, or straightforward fixes, authorize those changes
  directly. Do not require another proposal or confirmation.
- Commit, push, or open a PR only when requested. That request is sufficient
  authorization; choose a suitable title or message without another approval
  round. A commit request does not authorize a push. Include only agreed changes.
- Model/data downloads, remote jobs, and paid services require explicit
  authorization. State their scope and relevant cost/time limits beforehand;
  do not ask again when that scope is already approved.
- Preserve unrelated user changes. Never commit weights, datasets, checkpoints,
  profiler traces, secrets, or other large generated artifacts.

## Engineering and research standards

- Prefer the smallest clear implementation that tests the current hypothesis.
  Keep experiments configuration-driven and justify new dependencies.
- When interpreting new findings or proposing next steps, explicitly identify
  implications for other components and prior design decisions. Name obvious
  opportunities to transfer an approach, remove workarounds, or simplify earlier
  optimizations. Do not treat existing solutions as settled merely because they
  have already been implemented. Distinguish measured results from hypotheses
  and identify the smallest useful validation.
- Name files and modules after their component or responsibility. Use concise
  docstrings for purpose and contracts, and short comments at meaningful phase
  boundaries or where intent is not obvious. Explain the flow through combined
  components; avoid narrating every line or repeating self-explanatory code.
- Make names reveal the domain, stored value, and action: qualify metadata
  such as `validation_manifest`, distinguish batch plans from tensor batches,
  and expose side effects such as loading or broadcasting in function names.
  Avoid generic names such as `core` and stale strategy names for shared code.
  Update callers and documentation together; preserve persisted format names
  and explicitly required compatibility entrypoints when renaming code.
- Validate external inputs, configuration, and checkpoint contracts at ingestion
  or explicit preflight. Make frozen/trainable boundaries testable. Trust
  validated internal objects instead of repeating checks in every forward.
- Trust prepared datasets and pinned model snapshots. Do not rescan, rehash,
  parse, or check every source file at startup, resume, Dataset construction,
  or epoch transitions. Reuse preparation manifests and recorded hashes;
  read individual tasks when consumed and regenerate only augmentation keys.
  Full integrity audits belong to initial CPU preparation or explicitly
  requested diagnostics, never automatic GPU startup or a hidden fallback.
  Retain lightweight configuration, split-boundary, and checkpoint-metadata
  checks without traversing the dataset or rereading model weights.
- Keep hot-path checks for changing conditions that matter to correctness.
  Put expensive tensor scans in tests, diagnostics, or justified debug modes;
  run stable preflights explicitly rather than hiding them in first-forward
  flags. Avoid silent fallbacks that change the experiment being measured.

## Documentation and evidence

- Create new documentation files or reports only when explicitly requested by
  the user. Do not routinely ask whether to create them or offer to do so.
  Summarize routine work in the response; update existing documentation only
  when needed and keep changes proportional to the task. The guidance below
  does not authorize creating new documentation files.
- Write compact, self-contained explanations in plain language for readers
  unfamiliar with the conversation. Explain methods, results, and limitations;
  avoid repeated commands and incidental environment details in reports.
- Use diagrams, plots, or screenshots when they make the content easier to
  understand. Keep factual research claims traceable to primary sources and
  make uncertainty, evidence gaps, and evaluation limitations explicit.
- Store reproducibility details in experiment artifacts: seeds, configurations,
  data/model/package revisions, precision, hardware and distributed strategy,
  commands, and source identity. Link these details when they help the reader.

## Validation

- Validation is complete when checks cover the changed behavior, relevant
  failure modes, and agreed acceptance criteria; it does not mean running every
  available check. For documentation, review the diff; for localized code changes,
  run targeted tests; for numerical changes, compare relevant outputs, losses,
  or gradients. Avoid tests that merely mirror the implementation.
- Use broader suites or integration runs when shared interfaces, training
  behavior, or concrete regression risks warrant them. Once relevant checks pass,
  stop unless new changes, failures, or unresolved correctness questions justify
  more testing. Do not repeat equivalent checks just for additional reassurance.
- Establish single-process correctness before distributed execution. Use small
  smoke tests and overfitting runs before expensive training; compare numerical
  agreement as well as throughput when changing execution strategies.
- Report what was actually run, its outcome, and remaining limitations.

Repository unit tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests
```

## Communication

- Communicate with the user in German unless requested otherwise.
- Keep repository documentation, code, identifiers, and comments in English.
- Be direct, lead with the conclusion, and keep explanations easy to follow.
- Separate facts, measurements, assumptions, and proposals. State conflicting
  evidence and material risks plainly; group related questions together.

## Subagents

- Handle small and medium tasks directly by default, including localized fixes,
  documentation edits, routine searches, and targeted validation.
- Use subagents for substantial work with independent, bounded subtasks when
  parallel execution is likely to save meaningful time or a focused independent
  review addresses a concrete risk. Delegation does not require a separate user
  request. Weigh the benefit against context duplication, token use, and
  coordination overhead; task size alone is not enough.
- Start with the smallest useful number of subagents and add more only for
  distinct work that materially benefits. Avoid duplicate investigations and
  routine chains of further delegation. Assign clear ownership and avoid
  concurrent edits to overlapping files.
- The primary agent owns integration, validation, and the final report.
