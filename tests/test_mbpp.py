from __future__ import annotations

from fineqcomp.mbpp import (
    extract_code,
    run_humaneval_tests,
    run_mbpp_tests,
    validate_generated_code,
)


def test_extract_and_run_safe_code():
    response = "```python\ndef square(x):\n    return x * x\n```"
    assert extract_code(response).startswith("def square")
    result = run_mbpp_tests(response, ["assert square(4) == 16"])
    assert result["passed"] is True
    assert result["status"] == "passed"


def test_reject_unsafe_code_and_report_failed_tests():
    safe, reason = validate_generated_code("import os\nos.system('true')")
    assert safe is False
    assert reason == "unsafe import"
    failed = run_mbpp_tests("def square(x): return x", ["assert square(4) == 16"])
    assert failed["passed"] is False
    assert failed["status"] == "failed_tests"


def test_humaneval_completion_is_joined_to_the_function_prefix():
    result = run_humaneval_tests(
        "    return x * x\n",
        'def square(x):\n    """Return x squared."""\n',
        "def check(candidate):\n    assert candidate(4) == 16",
        "square",
    )

    assert result["passed"] is True


def test_humaneval_limits_eval_to_the_canonical_algebra_task():
    completion = "    return eval('1 + 1')\n"
    tests = "def check(candidate):\n    assert candidate([], []) == 2"

    assert run_humaneval_tests(
        completion, "def do_algebra(a, b):\n", tests, "do_algebra"
    )["passed"]
    assert not run_humaneval_tests(
        completion, "def other(a, b):\n", tests, "other"
    )["passed"]
