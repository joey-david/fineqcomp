"""Bounded MBPP execution with a strict generated-code allowlist."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ALLOWED_IMPORTS = {
    "bisect",
    "collections",
    "functools",
    "hashlib",
    "heapq",
    "itertools",
    "math",
    "operator",
    "re",
    "statistics",
    "string",
    "typing",
}
DANGEROUS_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}


def extract_code(text: str) -> str:
    match = re.search(
        r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    )
    return (match.group(1) if match else text).strip()


def validate_generated_code(
    code: str, allow_eval: bool = False
) -> tuple[bool, str | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return False, f"syntax: {error.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name.split(".")[0] not in ALLOWED_IMPORTS for alias in node.names
            ):
                return False, "unsafe import"
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] not in ALLOWED_IMPORTS:
                return False, "unsafe import"
        elif (
            isinstance(node, ast.Name)
            and node.id in DANGEROUS_NAMES
            and not (allow_eval and node.id == "eval")
        ):
            return False, f"unsafe name: {node.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, "dunder access"
    return True, None


def _limits() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    except (ImportError, OSError, ValueError):
        pass


def _run_python_tests(
    code: str, tests: str, timeout: float, allow_eval: bool = False
) -> dict[str, Any]:
    safe, reason = validate_generated_code(code, allow_eval=allow_eval)
    if not safe:
        return {"passed": False, "status": "rejected", "detail": reason}
    with tempfile.TemporaryDirectory(prefix="fineqcomp-mbpp-") as temp_dir:
        script = Path(temp_dir) / "candidate.py"
        script.write_text(code + "\n\n" + tests + "\n")
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(script)],
                cwd=temp_dir,
                env={"PYTHONHASHSEED": "0", "PATH": os.environ.get("PATH", "")},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                preexec_fn=_limits if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "status": "timeout", "detail": None}
    return {
        "passed": result.returncode == 0,
        "status": "passed" if result.returncode == 0 else "failed_tests",
        "detail": result.stderr[-500:] or None,
    }


def run_mbpp_tests(
    code_text: str, tests: list[str], timeout: float = 3.0
) -> dict[str, Any]:
    return _run_python_tests(extract_code(code_text), "\n".join(tests), timeout)


def run_humaneval_tests(
    completion: str,
    code_prefix: str,
    tests: str,
    entry_point: str,
    timeout: float = 3.0,
) -> dict[str, Any]:
    """Run one HumanEval completion in the same bounded subprocess as MBPP."""
    fenced = re.search(
        r"```(?:python)?\s*(.*?)```", completion, flags=re.DOTALL | re.IGNORECASE
    )
    extracted = fenced.group(1).strip() if fenced else completion.rstrip()
    code = extracted if f"def {entry_point}" in extracted else code_prefix + extracted
    test_program = tests + f"\ncheck({entry_point})"
    return _run_python_tests(
        code,
        test_program,
        timeout,
        allow_eval=entry_point == "do_algebra",
    )
