"""The one case schema shared by every case file (EVALUATION_PROTOCOL.md §7.1, Lesson 6 §8).

Every case, LitSearch or diagnostic, has exactly these fields. A field that doesn't
apply to a case type is present with a null/empty value, never missing.
"""

TOP_LEVEL = {
    "case_id": str,
    "case_set_version": str,
    "corpus_version": str,
    "split": str,                     # "dev" | "test"
    "source": str,                    # "litsearch:<query_set>" | "manual"
    "coverage_purpose": str,          # "estimate" | "diagnostic"
    "slice": str,
    "hard_case_tags": list,
    "hypothesis": str,
    "request": dict,
    "reference": dict,
    "expected_behavior": str,         # "answer" | "partial" | "abstain"
    "expected_parse": dict,
    "q1_scored": bool,
    "sufficiency_label": bool,
    "surveys_count": bool,
    "rubric_id": str,
    "label_author": str,
    "label_date": str,
    "label_notes": str,
    "source_details": (dict, type(None)),   # LitSearch row details; null for diagnostic cases
}
REQUEST = {"query": str, "audience": str, "date_scope": (dict, type(None)), "allowed_sources": str}
REFERENCE = {"relevant_ids": list, "relevance_rule": (dict, type(None)), "acceptable_alternative_ids": list,
             "wrong_scope_ids": list, "required_facts": list}
EXPECTED_PARSE = {"topic": (str, type(None)), "year_from": (int, type(None)),
                  "year_to": (int, type(None)), "author": (str, type(None))}

ALLOWED = {
    "split": {"dev", "test"},
    "coverage_purpose": {"estimate", "diagnostic"},
    "expected_behavior": {"answer", "partial", "abstain"},
}
HARD_CASE_TAGS = {"ambiguity", "missing_evidence", "conflicting_sources", "multi_source",
                  "noise_in_context", "false_premise"}
RULE_KEYS = {"title_regex", "year", "author", "primary_category"}


def validate(case: dict) -> list[str]:
    """Return a list of problems; an empty list means the case follows the schema."""
    cid = case.get("case_id", "?")
    errors = []

    def check(obj, spec, where):
        missing = set(spec) - set(obj)
        extra = set(obj) - set(spec)
        if missing:
            errors.append(f"{cid}: {where} missing {sorted(missing)}")
        if extra:
            errors.append(f"{cid}: {where} has unexpected {sorted(extra)}")
        for key, typ in spec.items():
            if key in obj and not isinstance(obj[key], typ):
                errors.append(f"{cid}: {where}{key} has type {type(obj[key]).__name__}")

    check(case, TOP_LEVEL, "")
    if isinstance(case.get("request"), dict):
        check(case["request"], REQUEST, "request.")
    if isinstance(case.get("reference"), dict):
        check(case["reference"], REFERENCE, "reference.")
        rule = case["reference"].get("relevance_rule")
        if isinstance(rule, dict) and (set(rule) - RULE_KEYS or "title_regex" not in rule):
            errors.append(f"{cid}: relevance_rule keys must be a subset of {sorted(RULE_KEYS)} incl. title_regex")
    if isinstance(case.get("expected_parse"), dict):
        check(case["expected_parse"], EXPECTED_PARSE, "expected_parse.")
    for key, values in ALLOWED.items():
        if case.get(key) not in values:
            errors.append(f"{cid}: {key}={case.get(key)!r} not in {sorted(values)}")
    bad_tags = set(case.get("hard_case_tags") or []) - HARD_CASE_TAGS
    if bad_tags:
        errors.append(f"{cid}: unknown hard_case_tags {sorted(bad_tags)}")
    if case.get("expected_behavior") == "abstain" and case.get("sufficiency_label"):
        errors.append(f"{cid}: abstain case should have sufficiency_label false")
    return errors
