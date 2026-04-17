from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import chromadb
import fitz
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "data" / "raw"
DB_DIR = BASE_DIR / os.getenv("CARECUE_DB_DIR", "indexes/chroma")
COLLECTION_NAME = os.getenv("CARECUE_COLLECTION", "carecue")
EMB_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_sidecar(path: Path) -> Dict:
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def iter_pdf_pages(path: Path) -> Iterator[Tuple[int, str]]:
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        yield i + 1, page.get_text("text")


def iter_text_doc(path: Path) -> Iterator[Tuple[int, str]]:
    yield 1, path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> Iterator[str]:
    i = 0
    n = len(text)
    while i < n:
        yield text[i : i + size]
        i += max(1, size - overlap)


def iter_documents(path: Path) -> Iterator[Tuple[int, str]]:
    if path.suffix.lower() == ".pdf":
        yield from iter_pdf_pages(path)
    elif path.suffix.lower() in {".txt", ".md"}:
        yield from iter_text_doc(path)


def main() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    emb_fn = SentenceTransformerEmbeddingFunction(model_name=EMB_MODEL)

    # reset collection for deterministic rebuilds
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    col = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=emb_fn)

    docs: List[str] = []
    ids: List[str] = []
    metas: List[Dict] = []
    total = 0

    for path in sorted(SRC_DIR.iterdir()):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        sidecar = load_sidecar(path)
        for page_num, text in iter_documents(path):
            text = normalize_text(text)
            if not text:
                continue
            for chunk_idx, chunk in enumerate(chunk_text(text), start=1):
                chunk = chunk.strip()
                if not chunk:
                    continue
                docs.append(chunk)
                ids.append(str(uuid.uuid4()))
                metas.append(
                    {
                        "source": sidecar.get("label", path.name),
                        "publisher": sidecar.get("publisher", "unknown"),
                        "url": sidecar.get("url", ""),
                        "page": page_num,
                        "chunk": chunk_idx,
                        "file_name": path.name,
                        "tags": ",".join(sidecar.get("tags", [])),
                    }
                )
                total += 1
                if len(ids) >= 200:
                    col.add(documents=docs, metadatas=metas, ids=ids)
                    docs, ids, metas = [], [], []

    if ids:
        col.add(documents=docs, metadatas=metas, ids=ids)

    print(f"Ingested {total} chunks into {DB_DIR} / collection={COLLECTION_NAME}")


if __name__ == "__main__":
    main()
