from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from models import generate_answer
from policies import audit_text_safety, classify_urgency
from retriever import Retriever

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(BASE_DIR / "ui"), static_url_path="/ui")
retr = Retriever(k=int(os.getenv("TOP_K", "5")))


@app.get("/")
def root():
    return app.send_static_file("index.html")


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/chat")
def chat():
    data = request.get_json(force=True, silent=True) or {}
    q = data.get("q", "").strip()
    mode = data.get("mode", "carecue")
    k = int(data.get("k", os.getenv("TOP_K", "5")))
    if not q:
        return jsonify({"error": "Missing 'q'"}), 400
    if mode not in {"carecue", "grounded_neutral"}:
        return jsonify({"error": "mode must be 'carecue' or 'grounded_neutral'"}), 400

    urgency = classify_urgency(q)
    docs = retr.search(q, k=k)
    answer = generate_answer(q, docs, mode=mode, urgency_label=urgency["label"])
    quality = audit_text_safety(answer, urgency_label=urgency["label"], mode=mode)
    citations = [
        {
            "idx": i + 1,
            "source": d["meta"].get("source"),
            "publisher": d["meta"].get("publisher"),
            "page": d["meta"].get("page"),
            "url": d["meta"].get("url"),
            "score": d.get("score"),
        }
        for i, d in enumerate(docs)
    ]
    return jsonify(
        {
            "answer": answer,
            "mode": mode,
            "urgency": urgency,
            "quality": quality,
            "citations": citations,
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
