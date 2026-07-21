# Diffusion-ARC Agent Guidance

## Repository source of truth

- Read `README.md` before planning work.
- Treat its project scope, research questions, milestones, required baselines,
  and non-goals as authoritative.
- Do not change the research hypothesis, milestone order, or initial non-goals
  without explicit user approval.
- Keep this root guidance concise. Add nested `AGENTS.md` files only when a
  subtree develops genuinely different commands or constraints.

## Mandatory gated workflow

For every request that could modify the repository or external state:

1. **Discover**
   - Inspect the relevant files, repository state, documentation, and existing
     user changes using read-only operations.
   - Do not edit files or run commands expected to modify project state.

2. **Clarify**
   - Identify all currently visible material ambiguities, assumptions, and
     decisions.
   - Ask the user the open questions in one consolidated batch.
   - Do not continue to implementation while a material question is unresolved.
   - If there are no open questions, state that explicitly.

3. **Propose**
   - Present a concrete implementation plan containing:
     - objective and acceptance criteria,
     - agreed decisions and assumptions,
     - files expected to change,
     - implementation approach,
     - validation strategy,
     - risks and explicit non-goals.
   - Distinguish required work from optional follow-ups.

4. **Wait for approval**
   - Do not implement or modify anything until the user gives unambiguous
     approval after reviewing the proposal.
   - A task request by itself is not approval to skip this gate.
   - Approval applies to one coherent, immediately preceding plan.

5. **Implement**
   - After approval, complete the approved scope without unnecessary pauses.
   - Preserve unrelated user changes and keep the patch focused.
   - If a new material ambiguity, risk, dependency, or scope change appears,
     stop and return to the clarification and approval steps.

6. **Verify and report**
   - Run validation proportional to the change.
   - Report the commands run, results obtained, remaining limitations, and
     deviations from the approved plan.
   - Never imply that a test, benchmark, or experiment passed if it was not run.

Repository mutation includes creating, editing, deleting, renaming, or formatting
files; installing dependencies; downloading models or datasets; generating
artifacts; staging or committing changes; and triggering external actions.

Model downloads, dataset downloads, paid services, remote jobs, and substantial
GPU training runs must be named explicitly in the plan and approved separately.

## Project invariants

Unless the user explicitly approves a scope change:

- Keep NVARC/Qwen3-4B frozen.
- Keep the FLUX VAE frozen.
- Train the complete FLUX transformer for the primary approach; LoRA is a
  required baseline, not the default training method.
- Train the NVARC-to-FLUX adapter and ARC color head.
- Postpone the output-shape head until the corresponding milestone.
- Use a fixed 30x30 canvas with ten ARC colors plus one padding/outside class.
- Provide the correct output dimensions during the initial implementation.
- Verify tokenizer, checkpoint, hidden-layer, tensor-shape, and dtype
  compatibility rather than assuming it.
- Split training and validation at the ARC task level.
- Never allow augmented variants of held-out tasks into training data.
- Treat exact-grid accuracy as the primary quality result.
- Preserve the required baselines before attributing gains to FLUX pretraining,
  flow matching, NVARC conditioning, or model capacity.
- Do not introduce milestone-5 extensions before the core hypothesis has been
  validated.

## Engineering and research standards

- Prefer the smallest implementation that can test the current hypothesis.
- Keep experiments configuration-driven and reproducible.
- Record relevant seeds, data split identifiers, model and package revisions,
  precision, hardware topology, distributed strategy, configuration, and command.
- Make frozen/trainable parameter boundaries explicit and testable.
- Add assertions or checks for tensor shapes, dtypes, non-finite values, and
  distributed assumptions at critical interfaces.
- Avoid silent fallbacks that can change the experiment being measured.
- Validate single-process correctness before DDP or FSDP2.
- Use tiny smoke tests and tiny-set overfitting before expensive training.
- Compare distributed strategies for numerical agreement as well as throughput.
- Do not commit model weights, datasets, checkpoints, profiler traces, secrets,
  or other large generated artifacts.
- Do not add dependencies unless they are justified in the approved plan.
- Preserve existing worktree changes. Never overwrite or revert unrelated work.
- Do not stage, commit, push, or create a pull request unless separately requested
  and approved.

## Validation commands

No canonical environment, lint, test, or training commands exist yet.

When the project scaffold is approved and introduced, add the actual commands
here in the same change. Do not invent commands that the repository does not
support.

## Communication

- Communicate with the user in German unless requested otherwise.
- Keep repository documentation, code, identifiers, and comments in English.
- Lead with the conclusion.
- Separate verified facts, assumptions, proposals, and measured results.
- Surface research risks and conflicting evidence instead of hiding them.
- Batch related questions so the user can make one coherent decision.

## Subagents

- Use subagents only when the user explicitly requests delegation or parallel
  agent work.
- Prefer them for independent, read-heavy exploration, review, test analysis, or
  log analysis.
- Do not allow multiple agents to edit overlapping files concurrently.
- The primary agent remains responsible for decisions, integration, validation,
  and the final report.
