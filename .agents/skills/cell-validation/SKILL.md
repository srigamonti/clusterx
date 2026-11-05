---
name: cell-validation
description: Run targeted tests and regression checks for changes to CELL Python code.
---

1. Read AGENTS.md and inspect the working-tree diff.
2. Activate the existing project virtual environment.
3. Select tests for the changed code and affected consumers.
4. Run test commands from the repository root; pytest isolates working directories
   automatically. Use `tmp_path` for additional file paths.
5. Report results and explain any checks skipped.
