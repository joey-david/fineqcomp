from __future__ import annotations

from fineqcomp.mbpp import extract_code, run_mbpp_tests, validate_generated_code


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
