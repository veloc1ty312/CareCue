from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
SEEDS_PATH = BASE_DIR / "health_source_seeds.jsonl"
RAW_DIR = BASE_DIR / "data" / "raw"
USER_AGENT = "CareCuePrototype/0.1 (+research prototype)"


def load_seeds() -> Iterable[Dict]:
    with open(SEEDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def safe_name(seed: Dict) -> str:
    text = f"{seed['publisher']}_{seed['label']}"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_").lower()


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    chunks = []
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if title:
        chunks.append(title)
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    chunks.append(text)
    return "\n\n".join(c for c in chunks if c).strip()


def fetch(seed: Dict) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = seed["url"]
    name = safe_name(seed)
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    meta_block = {
        "publisher": seed.get("publisher"),
        "label": seed.get("label"),
        "url": url,
        "tags": seed.get("tags", []),
        "content_type": content_type,
    }
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        out_path = RAW_DIR / f"{name}.pdf"
        out_path.write_bytes(resp.content)
        sidecar = RAW_DIR / f"{name}.json"
        sidecar.write_text(json.dumps(meta_block, indent=2), encoding="utf-8")
        return out_path

    text = html_to_text(resp.text)
    out_path = RAW_DIR / f"{name}.txt"
    out_path.write_text(text, encoding="utf-8")
    sidecar = RAW_DIR / f"{name}.json"
    sidecar.write_text(json.dumps(meta_block, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    saved = []
    for seed in load_seeds():
        try:
            path = fetch(seed)
            saved.append(str(path.name))
            print(f"saved {path.name}")
        except Exception as exc:
            print(f"failed {seed.get('url')}: {exc}")
    print(f"done: {len(saved)} files saved to {RAW_DIR}")


if __name__ == "__main__":
    main()
