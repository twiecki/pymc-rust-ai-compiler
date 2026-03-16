"""CLI for the Transpailer — AI-powered transpilation between computational frameworks."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import click

_SKILLS_DIR = Path(__file__).parent / "skills"

# Mapping from (source, target) pairs to relevant skill files
_SKILL_MAP: dict[tuple[str, str], list[str]] = {
    ("stan", "pymc"): ["stan_to_pymc.md"],
    ("stan", "rust"): ["stan.md"],
    ("pymc", "rust"): ["pymc_optimization.md"],
    ("jax", "pytorch"): ["jax_to_pytorch.md"],
    ("pytorch", "jax"): ["pytorch_to_jax.md"],
    ("pytorch", "rust"): ["pytorch_to_rust.md"],
}

# Framework aliases
_ALIASES: dict[str, str] = {
    "pm": "pymc",
    "numpyro": "numpyro",
    "np": "numpyro",
    "tf": "tensorflow",
    "torch": "pytorch",
    "pt": "pytorch",
    "rs": "rust",
    "turing": "turing.jl",
    "brms": "brms",
}

# File extension → framework
_EXT_MAP: dict[str, str] = {
    ".stan": "stan",
    ".bug": "bugs",
    ".bugs": "bugs",
    ".jl": "julia",
    ".r": "r",
    ".R": "r",
}

# Python import patterns → framework
_IMPORT_PATTERNS: list[tuple[str, str]] = [
    (r"import\s+pymc|from\s+pymc", "pymc"),
    (r"import\s+numpyro|from\s+numpyro", "numpyro"),
    (r"import\s+torch|from\s+torch", "pytorch"),
    (r"import\s+jax|from\s+jax", "jax"),
    (r"import\s+tensorflow|from\s+tensorflow|import\s+tf", "tensorflow"),
    (r"import\s+pyro\b|from\s+pyro\b", "pyro"),
    (r"import\s+stan|from\s+cmdstanpy|from\s+pystan", "stan"),
]


def _detect_framework(code: str, filename: str) -> str:
    """Auto-detect source framework from file extension and content."""
    ext = Path(filename).suffix.lower()
    if ext in _EXT_MAP:
        return _EXT_MAP[ext]

    if ext in (".py", ""):
        for pattern, framework in _IMPORT_PATTERNS:
            if re.search(pattern, code):
                return framework

    raise click.UsageError(
        f"Cannot auto-detect source framework for '{filename}'. Use --from to specify it explicitly."
    )


def _normalize_framework(name: str) -> str:
    """Normalize framework name (handle aliases)."""
    return _ALIASES.get(name.lower(), name.lower())


def _load_skills(source: str, target: str) -> str:
    """Load relevant skill files for this transpilation pair."""
    filenames = _SKILL_MAP.get((source, target), [])
    parts = []
    for fname in filenames:
        skill_path = _SKILLS_DIR / fname
        if skill_path.exists():
            parts.append(skill_path.read_text())
    return "\n\n".join(parts)


_SYSTEM_PROMPT = """\
You are the Transpailer — an expert AI that transpiles code between computational \
frameworks. You produce clean, idiomatic code in the target framework that is \
functionally equivalent to the input.

Supported frameworks include (but are not limited to):
- Probabilistic programming: Stan, PyMC, NumPyro, Turing.jl, BUGS, brms
- Deep learning: PyTorch, JAX, TensorFlow
- Systems: Rust, C++

Rules:
1. Output ONLY the transpiled code — no explanations, no markdown fences.
2. Preserve the mathematical/computational semantics exactly.
3. Use idiomatic patterns of the target framework.
4. Include all necessary imports.
5. Translate language-specific conventions (e.g. 1-based → 0-based indexing).
6. Keep variable names as close to the original as possible.
"""


def _transpile(
    code: str,
    source: str,
    target: str,
    *,
    model: str = "claude-sonnet-4-20250514",
    verbose: bool = False,
) -> str:
    """Call Claude to transpile code from source to target framework."""
    import anthropic

    skills = _load_skills(source, target)

    system = _SYSTEM_PROMPT
    if skills:
        system += f"\n\n# Domain knowledge\n\n{skills}"

    user_msg = f"Transpile the following {source} code to {target}.\n\n```\n{code}\n```"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise click.ClickException(
            "ANTHROPIC_API_KEY environment variable is required. Set it with: export ANTHROPIC_API_KEY=sk-..."
        )

    client = anthropic.Anthropic(api_key=api_key)

    if verbose:
        click.echo(f"Calling {model}...", err=True)

    response = client.messages.create(
        model=model,
        max_tokens=16384,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    result = response.content[0].text

    # Strip markdown code fences if the model wrapped its output
    result = re.sub(r"^```\w*\n", "", result)
    result = re.sub(r"\n```\s*$", "", result)

    if verbose:
        usage = response.usage
        click.echo(
            f"Done. Tokens: {usage.input_tokens} in / {usage.output_tokens} out",
            err=True,
        )

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────


@click.group()
@click.version_option(package_name="transpailer")
def cli():
    """Transpailer: AI-powered transpilation between computational frameworks."""


@cli.command()
@click.argument("input_file", type=click.Path(exists=True), required=False)
@click.option(
    "--to",
    "target",
    required=True,
    help="Target framework (pymc, pytorch, jax, stan, numpyro, tensorflow, rust, ...)",
)
@click.option(
    "--from",
    "source",
    default=None,
    help="Source framework (auto-detected from file extension/content if omitted)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path (default: stdout)",
)
@click.option(
    "--model",
    default="claude-sonnet-4-20250514",
    help="Claude model to use",
    show_default=True,
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output on stderr")
def convert(
    input_file: str | None,
    target: str,
    source: str | None,
    output: str | None,
    model: str,
    verbose: bool,
) -> None:
    """Transpile code from one framework to another.

    \b
    Examples:
      transpailer convert model.stan --to pymc
      transpailer convert train.py --to jax
      transpailer convert model.py --to pytorch
      cat model.stan | transpailer convert --to pymc
    """
    # Read input
    if input_file:
        code = Path(input_file).read_text()
        filename = Path(input_file).name
    elif not sys.stdin.isatty():
        code = sys.stdin.read()
        filename = "stdin"
    else:
        raise click.UsageError(
            "No input file provided and no data on stdin. Usage: transpailer convert <file> --to <framework>"
        )

    target = _normalize_framework(target)

    # Detect source framework
    if source:
        source = _normalize_framework(source)
    else:
        source = _detect_framework(code, filename)
        if verbose:
            click.echo(f"Detected source: {source}", err=True)

    if source == target:
        raise click.UsageError(f"Source and target are the same: {source}")

    if verbose:
        click.echo(f"Transpiling {source} → {target}...", err=True)

    result = _transpile(code, source, target, model=model, verbose=verbose)

    # Write output
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(result)
        if verbose:
            click.echo(f"Written to {output}", err=True)
    else:
        click.echo(result)
