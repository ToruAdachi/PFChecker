# Project Rules for Codex (PFChecker)

## Scope / Priorities
- Primary goal: implement requested changes with minimal risk.
- Preserve existing behavior unless the change explicitly requires otherwise.
- Keep diffs small and reviewable.

## Language / Runtime
- Python 3.x
- Do not assume dependency managers (poetry/pipenv) unless they exist in the repo.

## How to Run
- Do NOT invent run commands.
- Before running anything, locate and propose the correct command(s) based on the repo files
  (e.g., README, Makefile, pyproject.toml, requirements.txt, entry scripts).
- If the correct run command is unclear, ask for it.

## Testing (No existing tests)
- Assume there are no tests unless a tests/ directory or pytest config exists.
- Do NOT introduce a test framework or CI by default.
- Only add tests if the user explicitly requests it.
- If adding tests is requested, keep it minimal (smoke test level) and follow existing tooling.

## Code Style / Refactoring
- Follow existing style.
- Do NOT reformat or refactor unrelated code.
- Avoid large structural changes unless explicitly requested.

## Files / Safety
- Modify only files necessary for the task.
- Do NOT delete files unless explicitly instructed.
- Do NOT change .gitignore, project structure, or dependencies unless requested.

## Git Workflow
- Do NOT commit automatically.
- Summarize changed files and the rationale at the end of the work.
# Project Rules for Codex (PFChecker)

## Scope / Priorities
- Primary goal: implement requested changes with minimal risk.
- Preserve existing behavior unless the change explicitly requires otherwise.
- Keep diffs small and reviewable.

## Language / Runtime
- Python 3.x
- Do not assume dependency managers (poetry/pipenv) unless they exist in the repo.

## How to Run
- Do NOT invent run commands.
- Before running anything, locate and propose the correct command(s) based on the repo files
  (e.g., README, Makefile, pyproject.toml, requirements.txt, entry scripts).
- If the correct run command is unclear, ask for it.

## Testing (No existing tests)
- Assume there are no tests unless a tests/ directory or pytest config exists.
- Do NOT introduce a test framework or CI by default.
- Only add tests if the user explicitly requests it.
- If adding tests is requested, keep it minimal (smoke test level) and follow existing tooling.

## Code Style / Refactoring
- Follow existing style.
- Do NOT reformat or refactor unrelated code.
- Avoid large structural changes unless explicitly requested.

## Files / Safety
- Modify only files necessary for the task.
- Do NOT delete files unless explicitly instructed.
- Do NOT change .gitignore, project structure, or dependencies unless requested.

## Git Workflow
- Do NOT commit automatically.
- Summarize changed files and the rationale at the end of the work.
