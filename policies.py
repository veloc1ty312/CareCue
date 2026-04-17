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

    emergency_hits = matched(schema["emergency_patterns"])
    urgent_hits = matched(schema["urgent_patterns"])
    mental_health_hits = matched(schema["mental_health_crisis_patterns"])

    if mental_health_hits:
        return {
            "label": "emergency",
            "reason": "mental_health_crisis",
            "matched_patterns": mental_health_hits,
        }
    if emergency_hits:
        return {
            "label": "emergency",
            "reason": "physical_red_flag",
            "matched_patterns": emergency_hits,
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
            "This could be urgent. If you are having severe chest pain, trouble breathing, "
            "stroke-like symptoms, severe bleeding, loss of consciousness, or you feel unable "
            "to stay safe, call 911 now or go to the nearest emergency department. "
            "If this is a mental health crisis in the U.S. or Canada, call or text 988."
        )
    if label == "urgent":
        return (
            "This does not automatically mean an emergency, but it would be safer to contact a "
            "clinician, urgent care, or a nurse line soon - ideally today - especially if symptoms "
            "are worsening or new red flags appear."
        )
    return (
        "I can help with general information and next-step guidance, but this is not a diagnosis. "
        "If symptoms become severe, rapidly worsen, or new red flags appear, seek urgent medical care."
    )


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
