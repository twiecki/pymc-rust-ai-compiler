"""Auto-format generated code using ruff."""

from __future__ import annotations

import subprocess


def format_python_code(code: str) -> str:
    """Format Python code using ruff format.

    Falls back to the original code if ruff is not available or formatting fails.
    """
    try:
        result = subprocess.run(
            ["ruff", "format", "--quiet", "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return code
