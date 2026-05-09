"""Answer-support assessment helpers.

Pure functions that compare a final answer against an evidence snippet using
rule-based day-constraint checks plus lexical overlap.
"""

from __future__ import annotations

import re


def extract_day_constraint(text: str) -> tuple[str, int] | None:
    lowered = text.lower()
    patterns = [
        (r"\bafter\s+(\d+)\s+days?\b", "after"),
        (r"\bwithin\s+(\d+)\s+days?\b", "within"),
        (r"\bup to\s+(\d+)\s+days?\b", "within"),
    ]
    for pattern, mode in patterns:
        match = re.search(pattern, lowered)
        if match:
            try:
                return mode, int(match.group(1))
            except ValueError:
                return None
    return None


def tokenize_support_text(text: str) -> set[str]:
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
        "get", "i", "if", "in", "is", "it", "of", "on", "or", "the", "to",
        "was", "what", "with", "you",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in stop_words
    }


def answer_polarity(answer: str) -> str:
    normalized = answer.strip().lower()
    if normalized.startswith(("yes", "yeah", "yep")):
        return "yes"
    if normalized.startswith(("no", "nope")):
        return "no"
    return "unknown"


def assess_answer_support(retrieval: dict | None, evidence_text: str | None) -> dict:
    if retrieval is None:
        return {
            "status": "unclear",
            "reason": "No retrieval context was provided.",
            "details": {"method": "rule_plus_overlap"},
        }

    final_answer = retrieval.get("final_answer")
    query_text = retrieval.get("query_text")
    if not final_answer:
        return {
            "status": "unclear",
            "reason": "Final answer is missing.",
            "details": {"method": "rule_plus_overlap"},
        }
    if not evidence_text:
        return {
            "status": "unclear",
            "reason": "Evidence snippet is missing.",
            "details": {"method": "rule_plus_overlap"},
        }

    polarity = answer_polarity(str(final_answer))
    details: dict[str, object] = {"method": "rule_plus_overlap", "answer_polarity": polarity}

    if query_text:
        query_constraint = extract_day_constraint(str(query_text))
        evidence_constraint = extract_day_constraint(str(evidence_text))
        if query_constraint and evidence_constraint:
            details["query_day_constraint"] = {
                "mode": query_constraint[0],
                "days": query_constraint[1],
            }
            details["evidence_day_constraint"] = {
                "mode": evidence_constraint[0],
                "days": evidence_constraint[1],
            }
            if (
                query_constraint[0] == "after"
                and evidence_constraint[0] == "within"
                and query_constraint[1] > evidence_constraint[1]
            ):
                if polarity == "yes":
                    return {
                        "status": "unsupported",
                        "reason": (
                            f"Question asks about after {query_constraint[1]} days, "
                            f"but evidence limits to within {evidence_constraint[1]} days."
                        ),
                        "details": details,
                    }
                if polarity == "no":
                    return {
                        "status": "supported",
                        "reason": (
                            f"Evidence limits refunds to within {evidence_constraint[1]} days; "
                            f"question asks about after {query_constraint[1]} days."
                        ),
                        "details": details,
                    }

    answer_tokens = tokenize_support_text(str(final_answer))
    evidence_tokens = tokenize_support_text(str(evidence_text))
    if polarity in {"yes", "no"} and len(answer_tokens) <= 2:
        details["answer_tokens"] = sorted(answer_tokens)
        return {
            "status": "unclear",
            "reason": "Answer is polarity-only; not enough lexical content for overlap scoring.",
            "details": details,
        }
    if not answer_tokens:
        return {
            "status": "unclear",
            "reason": "Final answer lacks enough terms for comparison.",
            "details": details,
        }
    overlap = sorted(answer_tokens.intersection(evidence_tokens))
    overlap_ratio = len(overlap) / len(answer_tokens)
    details["answer_tokens"] = sorted(answer_tokens)
    details["overlap_terms"] = overlap
    details["overlap_ratio"] = round(overlap_ratio, 4)

    if overlap_ratio >= 0.7:
        return {
            "status": "supported",
            "reason": "Answer terms align with retrieved evidence snippet.",
            "details": details,
        }
    if overlap_ratio == 0:
        return {
            "status": "unsupported",
            "reason": "Answer terms have weak overlap with retrieved evidence snippet.",
            "details": details,
        }
    return {
        "status": "unclear",
        "reason": "Evidence overlap is partial; manual review recommended.",
        "details": details,
    }
