from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / os.getenv("CARECUE_DB_DIR", "indexes/chroma")
COLLECTION_NAME = os.getenv("CARECUE_COLLECTION", "carecue")
EMB_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class Retriever:
    def __init__(self, k: int = 5):
        self.client = chromadb.PersistentClient(path=str(DB_DIR))
        self.col = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=SentenceTransformerEmbeddingFunction(model_name=EMB_MODEL),
        )
        self.k = k

    def search(self, query: str, k: Optional[int] = None, where: Optional[Dict] = None) -> List[Dict]:
        k = k or self.k
        res = self.col.query(
            query_texts=[query],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        hits = []
        for text, meta, dist in zip(docs, metas, dists):
            score = 1.0 / (1.0 + float(dist)) if dist is not None else None
            hits.append({"text": text, "meta": meta, "score": score})
        return hits
