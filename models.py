from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from policies import audit_text_safety, escalation_copy

load_dotenv()

SYSTEM_COMMON = (
    "You are CareCue, a cautious consumer health information assistant. "
    "Use only the provided retrieved evidence. Do not diagnose. Do not invent facts. "
    "Cite evidence inline using square-bracket references like [1] and [2] that map to the context blocks."
)

CARECUE_STYLE = (
    "Write in a calm but credible tone. Use at most one brief emotional acknowledgment. "
    "Do not say 'don't worry' or imply the user is definitely safe. "
    "Do not use dramatic wording like 'terrifying' or visually alarming language. "
    "Separate common possibilities from warning signs. Put urgent escalation near the top when needed. "
    "End with concrete next steps."
)

NEUTRAL_STYLE = (
    "Write in a neutral, grounded informational tone. "
    "Do not add emotional validation, calming language, empathy-style openings, or reassurance. "
    "Be concise, factual, and clear about uncertainty and next steps. "
    "For emergency or urgent cases, put the action step before background explanation."
)

BANNED_REASSURANCE_REPLACEMENTS = {
    r"\bdon'?t worry\b": "it is understandable to want clarity",
    r"\btry not to panic\b": "take the next step calmly",
    r"\byou'?re probably okay\b": "there are multiple possible causes",
    r"\byou should be fine\b": "it is hard to know the cause from symptoms alone",
    r"\bthis is nothing to worry about\b": "this may not be serious, but it should be watched carefully",
    r"\bdefinitely not serious\b": "not all causes are serious",
    r"\byou are safe\b": "follow the appropriate safety guidance",
}


def _context_block(docs: List[Dict]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.get("meta", {})
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        url = meta.get("url", "")
        blocks.append(f"[{i}] source={src} page={page} url={url}\n{d['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, docs: List[Dict], mode: str, urgency_label: str) -> Tuple[str, str]:
    context = _context_block(docs)
    escalation = escalation_copy(urgency_label)
    if mode == "grounded_neutral":
        style = NEUTRAL_STYLE
        target_words = "90-150 words"
    else:
        style = CARECUE_STYLE
        target_words = "100-170 words"

    if urgency_label == "emergency":
        target_words = "70-130 words"
        response_order = (
            "Response order for this case:\n"
            "1. First sentence: state the emergency action clearly.\n"
            "2. Second sentence: briefly explain why this symptom pattern can be urgent.\n"
            "3. Then mention uncertainty and cite evidence.\n"
            "4. End with the immediate next step.\n"
            "Do not put a long explanation before the urgent action.\n"
        )
    elif urgency_label == "urgent":
        target_words = "90-150 words"
        response_order = (
            "Response order for this case:\n"
            "1. First or second sentence: state the same-day care recommendation.\n"
            "2. Briefly explain what makes the symptoms worth checking.\n"
            "3. Mention uncertainty and cite evidence.\n"
            "4. End with concrete next steps and what to watch for.\n"
            "Do not put a long explanation before the action step.\n"
        )
    elif mode == "carecue":
        response_order = (
            "Response order for this case:\n"
            "1. Start with one brief acknowledgment only if the user expresses worry or fear.\n"
            "2. Give a balanced explanation.\n"
            "3. Mention uncertainty.\n"
            "4. Give watch-outs and next steps.\n"
        )
    else:
        response_order = (
            "Response order for this case:\n"
            "1. Start with a direct factual explanation.\n"
            "2. Mention uncertainty.\n"
            "3. Give watch-outs and next steps.\n"
        )

    instructions = (
        f"{SYSTEM_COMMON} {style} "
        "Return plain text only. No markdown headings. No bullet lists unless needed for safety or next steps. "
        "If the evidence is insufficient, say so plainly."
    )

    prompt = (
        f"RETRIEVED EVIDENCE:\n{context}\n\n"
        f"USER QUESTION: {question}\n\n"
        f"URGENCY LABEL: {urgency_label}\n"
        f"REQUIRED ESCALATION LANGUAGE TO INCORPORATE WHEN RELEVANT: {escalation}\n\n"
        f"{response_order}\n"
        "RESPONSE REQUIREMENTS:\n"
        "1. Use only the retrieved evidence.\n"
        f"2. Keep the answer concise, around {target_words}.\n"
        "3. Include inline citations like [1], [2] for medical claims.\n"
        "4. For emergency or urgent cases, clearly recommend action early in the answer.\n"
        "5. Mention uncertainty.\n"
        "6. Do not diagnose or claim a condition is ruled out.\n"
        "7. Do not say the person is fine, safe, or definitely okay.\n"
        "8. Do not ask the user to judge how urgent the symptoms are. Give guidance based on red flags.\n"
        "9. Do not repeat information the user already gave unless needed for safety."
    )
    return instructions, prompt


def _clean_reassurance(text: str) -> str:
    cleaned = text
    for pattern, replacement in BANNED_REASSURANCE_REPLACEMENTS.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


def _has_emergency_action(text: str) -> bool:
    t = text.lower()
    return any(
        phrase in t
        for phrase in [
            "call 911",
            "call 9-1-1",
            "dial 911",
            "dial 9-1-1",
            "emergency department",
            "emergency room",
            "go to the er",
            "go to er",
            "seek emergency care",
        ]
    )


def _has_same_day_action(text: str) -> bool:
    t = text.lower()
    return any(
        phrase in t
        for phrase in [
            "today",
            "same day",
            "same-day",
            "urgent care",
            "contact a clinician",
            "call a clinician",
            "nurse line",
            "medical evaluation",
        ]
    )


def _frontload_required_action(answer: str, urgency_label: str) -> str:
    text = answer.strip()
    lower = text.lower()

    if urgency_label == "emergency":
        action = (
            "Because these symptoms can be urgent, call 911 now or go to the nearest emergency department "
            "if they are happening now or worsening. "
        )
        action_pos = min(
            [pos for pos in [lower.find("911"), lower.find("emergency department"), lower.find("emergency room"), lower.find("er")] if pos >= 0]
            or [10_000]
        )
        if not _has_emergency_action(text) or action_pos > 220:
            return action + text

    if urgency_label == "urgent":
        action = (
            "The safest next step is to contact a clinician, urgent care, or a nurse line today, "
            "especially if symptoms are worsening. "
        )
        action_pos = min(
            [pos for pos in [lower.find("today"), lower.find("urgent care"), lower.find("clinician"), lower.find("nurse line")] if pos >= 0]
            or [10_000]
        )
        if not _has_same_day_action(text) or action_pos > 260:
            return action + text

    return text


def postprocess_answer(answer: str, urgency_label: str) -> str:
    text = answer.strip()
    text = _clean_reassurance(text)
    text = _frontload_required_action(text, urgency_label)
    return text.strip()


def generate_answer(question: str, docs: List[Dict], mode: str, urgency_label: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    instructions, prompt = build_messages(question, docs, mode, urgency_label)
    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
        temperature=0.2,
        max_output_tokens=450,
    )
    answer = postprocess_answer(resp.output_text.strip(), urgency_label=urgency_label)
    audit_text_safety(answer, urgency_label=urgency_label, mode=mode)
    return answer
