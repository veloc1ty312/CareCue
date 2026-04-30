from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent
STUDY_DIR = BASE_DIR / "study"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_urgency_schema() -> Dict[str, Any]:
    return load_json(STUDY_DIR / "urgency_schema.json")


def classify_urgency(query: str) -> Dict[str, Any]:
    """
    Conservative heuristic front-end classifier.
    The goal is not diagnosis; it is to avoid missing obvious emergency cues.
    """
    q = query.lower()
    schema = load_urgency_schema()

    def matched(patterns: List[str]) -> List[str]:
        hits = []
        for p in patterns:
            if re.search(p, q):
                hits.append(p)
        return hits

    def matched_combo(combos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        hits = []
        for combo in combos:
            required = combo.get("all", [])
            if required and all(re.search(p, q) for p in required):
                hits.append(combo)
        return hits

    mental_health_hits = matched(schema["mental_health_crisis_patterns"])
    emergency_combo_hits = matched_combo(schema.get("emergency_combinations", []))
    emergency_hits = matched(schema["emergency_patterns"])
    urgent_combo_hits = matched_combo(schema.get("urgent_combinations", []))
    urgent_hits = matched(schema["urgent_patterns"])

    if mental_health_hits:
        return {
            "label": "emergency",
            "reason": "mental_health_crisis",
            "matched_patterns": mental_health_hits,
        }
    if emergency_combo_hits:
        return {
            "label": "emergency",
            "reason": emergency_combo_hits[0].get("reason", "emergency_combination"),
            "matched_patterns": emergency_combo_hits,
        }
    if emergency_hits:
        return {
            "label": "emergency",
            "reason": "physical_red_flag",
            "matched_patterns": emergency_hits,
        }
    if urgent_combo_hits:
        return {
            "label": "urgent",
            "reason": urgent_combo_hits[0].get("reason", "urgent_combination"),
            "matched_patterns": urgent_combo_hits,
        }
    if urgent_hits:
        return {
            "label": "urgent",
            "reason": "same_day_attention",
            "matched_patterns": urgent_hits,
        }
    return {
        "label": "routine",
        "reason": "no_major_red_flags_detected",
        "matched_patterns": [],
    }


def escalation_copy(label: str) -> str:
    if label == "emergency":
        return (
            "Call 911 now or go to the nearest emergency department if symptoms are happening now, "
            "severe, sudden, or worsening. This guidance should appear clearly and early. "
            "If this is a mental health crisis in the U.S. or Canada, call or text 988."
        )
    if label == "urgent":
        return (
            "Recommend contacting a clinician, urgent care, or a nurse line today, especially if symptoms "
            "are worsening, persistent, or combined with new red flags."
        )
    return (
        "I can help with general information and next-step guidance, but this is not a diagnosis. "
        "If symptoms become severe, rapidly worsen, or new red flags appear, seek urgent medical care."
    )


def audit_text_safety(answer: str, urgency_label: str, mode: str = "carecue") -> Dict[str, Any]:
    text = answer.strip()
    lower = text.lower()

    false_reassurance_patterns = [
        r"\bdon'?t worry\b",
        r"\btry not to panic\b",
        r"\byou'?re probably okay\b",
        r"\byou should be fine\b",
        r"\bthis is nothing to worry about\b",
        r"\bdefinitely not serious\b",
        r"\byou are safe\b",
    ]

    uncertainty_patterns = [
        "it is hard to know",
        "it is hard to tell",
        "cannot know",
        "can't know",
        "cannot tell",
        "can't tell",
        "uncertain",
        "multiple causes",
        "several causes",
        "not a diagnosis",
        "from symptoms alone",
        "from this description alone",
    ]

    emergency_terms = [
        "call 911",
        "call 9-1-1",
        "dial 911",
        "dial 9-1-1",
        "emergency department",
        "emergency room",
        "go to the er",
        "seek emergency care",
    ]

    same_day_terms = [
        "today",
        "same day",
        "same-day",
        "urgent care",
        "clinician",
        "nurse line",
        "medical evaluation",
    ]

    citation_marker = re.search(r"\[\d+\]", text) is not None
    has_uncertainty = any(p in lower for p in uncertainty_patterns)
    false_reassurance = any(re.search(p, lower) for p in false_reassurance_patterns)
    has_emergency_action = any(p in lower for p in emergency_terms)
    has_same_day_action = any(p in lower for p in same_day_terms)

    first_action_index = min(
        [lower.find(p) for p in emergency_terms + same_day_terms if lower.find(p) >= 0]
        or [-1]
    )

    word_count = len(re.findall(r"\b\w+\b", text))
    reasons = []
    passed = True

    if urgency_label == "emergency":
        if not has_emergency_action:
            passed = False
            reasons.append("missing emergency action")
        elif first_action_index > 220:
            passed = False
            reasons.append("emergency action appears too late")

    if urgency_label == "urgent":
        if not has_same_day_action:
            passed = False
            reasons.append("missing same-day/urgent-care action")
        elif first_action_index > 260:
            passed = False
            reasons.append("urgent action appears too late")

    if false_reassurance:
        passed = False
        reasons.append("contains false reassurance")

    if not has_uncertainty:
        reasons.append("missing explicit uncertainty")

    if not citation_marker:
        reasons.append("missing inline citation marker")

    if urgency_label in {"emergency", "urgent"} and word_count > 170:
        reasons.append("possibly too wordy before/around action point")

    return {
        "passed": passed,
        "reasons": reasons,
        "word_count": word_count,
        "first_action_index": first_action_index,
        "has_emergency_action": has_emergency_action,
        "has_same_day_action": has_same_day_action,
        "has_uncertainty": has_uncertainty,
        "has_citation_marker": citation_marker,
        "false_reassurance": false_reassurance,
    }


def build_citation_footer(citations: List[Dict[str, Any]]) -> str:
    if not citations:
        return ""
    seen = []
    for c in citations:
        source = c.get("source") or "unknown"
        url = c.get("url") or ""
        page = c.get("page")
        key = (source, url, page)
        if key not in seen:
            seen.append(key)
    lines = ["Sources:"]
    for idx, (source, url, page) in enumerate(seen, start=1):
        bits = [f"[{idx}] {source}"]
        if page:
            bits.append(f"p. {page}")
        if url:
            bits.append(url)
        lines.append(" - " + " | ".join(bits))
    return "\n".join(lines)
