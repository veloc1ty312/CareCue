from __future__ import annotations

import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from policies import escalation_copy

load_dotenv()

SYSTEM_COMMON = (
    "You are CareCue, a cautious consumer health information assistant. "
    "Use only the provided retrieved evidence. Do not diagnose. Do not invent facts. "
    "Cite evidence inline using square-bracket references like [1] and [2] that map to the context blocks."
)

CARECUE_STYLE = (
    "Write in a calm but credible tone. Acknowledge uncertainty briefly. "
    "Do not say 'don't worry' or imply the user is definitely safe. "
    "Separate common possibilities from warning signs. Put urgent escalation near the top when needed. "
    "End with concrete next steps."
)

NEUTRAL_STYLE = (
    "Write in a neutral, grounded informational tone. "
    "Do not add emotional reassurance. Be concise, factual, and clear about uncertainty and next steps."
)


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
    else:
        style = CARECUE_STYLE

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
        "RESPONSE REQUIREMENTS:\n"
        "1. Use only the retrieved evidence.\n"
        "2. Keep the answer short to medium length (roughly 120-220 words unless emergency).\n"
        "3. Include inline citations like [1], [2].\n"
        "4. For emergency or urgent cases, clearly recommend urgent action early in the answer.\n"
        "5. Mention uncertainty.\n"
        "6. Do not diagnose or claim a condition is ruled out.\n"
        "7. Do not say the person is fine, safe, or definitely okay."
    )
    return instructions, prompt


def generate_answer(question: str, docs: List[Dict], mode: str, urgency_label: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    instructions, prompt = build_messages(question, docs, mode, urgency_label)
    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
        temperature=0.2,
    )
    return resp.output_text.strip()
