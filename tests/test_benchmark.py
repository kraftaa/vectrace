"""Regression suite over the canonical wrong-answer cases.

Loads ``examples/benchmark/cases.json`` and asserts each case's expected
``support_status`` is reproduced by ``output.support.assess_answer_support``.
This is the contract the public README's incident example relies on; if you
change the assessor, update fixtures and version-bump cases.json's
schema_version.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from output.support import assess_answer_support

CASES_PATH = Path(__file__).resolve().parents[1] / "examples" / "benchmark" / "cases.json"


def _load_cases() -> list[dict]:
    with CASES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases", [])
    if not cases:
        raise AssertionError(f"No benchmark cases loaded from {CASES_PATH}")
    return cases


class BenchmarkSupportStatusTests(unittest.TestCase):
    def test_all_cases_match_expected_support_status(self) -> None:
        for case in _load_cases():
            with self.subTest(case=case["id"]):
                retrieval = {
                    "query_text": case["query_text"],
                    "final_answer": case["final_answer"],
                }
                result = assess_answer_support(
                    retrieval=retrieval,
                    evidence_text=case["evidence_text"],
                )
                self.assertEqual(
                    result["status"],
                    case["expected_support_status"],
                    msg=(
                        f"case={case['id']}: expected {case['expected_support_status']!r}, "
                        f"got {result['status']!r} (reason={result['reason']!r})"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
