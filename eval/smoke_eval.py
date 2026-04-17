from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parents[1]
API_URL = "http://localhost:8000/chat"
SEEDS = BASE_DIR / "eval" / "seed_qas.jsonl"


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="carecue", choices=["carecue", "grounded_neutral"])
    args = parser.parse_args()

    for ex in load_jsonl(SEEDS):
        r = requests.post(API_URL, json={"q": ex["q"], "mode": args.mode}, timeout=120)
        print("=" * 80)
        print(ex["id"], args.mode)
        if not r.ok:
            print("ERROR", r.status_code, r.text)
            continue
        js = r.json()
        print("urgency:", js.get("urgency"))
        print("answer:\n", js.get("answer", ""))
        print("citations:", len(js.get("citations", [])))


if __name__ == "__main__":
    main()
