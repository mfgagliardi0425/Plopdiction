Pre-commit hooks

This repository includes a `.pre-commit-config.yaml` to enforce formatting and basic linting.

To enable locally:

```powershell
# Install pre-commit in your virtualenv
.venv\Scripts\python.exe -m pip install pre-commit

# Install the Git hook scripts
.venv\Scripts\pre-commit install

# Run checks once
.venv\Scripts\pre-commit run --all-files
```

Recommended hooks: `black`, `isort`, and `ruff` (auto-fix), plus `trailing-whitespace` and `end-of-file-fixer`.
