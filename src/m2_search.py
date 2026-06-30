from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
        return str(segmented).replace("_", " ")
    except Exception as exc:
        print(f"  Warning: Vietnamese tokenizer unavailable, using whitespace tokens: {exc}")
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = list(chunks)
        self.corpus_tokens = [segment_vietnamese(chunk["text"].lower()).split() for chunk in self.documents]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []
        query_tokens = segment_vietnamese(query.lower()).split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[index]["text"],
                score=float(scores[index]),
                metadata=dict(self.documents[index].get("metadata", {})),
                method="bm25",
            )
            for index in top_indices
            if scores[index] > 0
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        if not chunks:
            raise ValueError("Cannot build a dense index with no chunks")
        texts = [chunk["text"] for chunk in chunks]
        vectors = self._get_encoder().encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=index,
                vector=vector.tolist(),
                payload={**chunk.get("metadata", {}), "text": chunk["text"]},
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection, points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0 or not self.client.collection_exists(collection):
            return []
        query_vector = self._get_encoder().encode(query, normalize_embeddings=True).tolist()
        response = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
        results: list[SearchResult] = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))
            if text:
                results.append(SearchResult(text=text, score=float(point.score), metadata=payload, method="dense"))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    fused: dict[str, dict] = {}
    for ranked_results in results_list:
        for rank, result in enumerate(ranked_results):
            entry = fused.setdefault(result.text, {"score": 0.0, "result": result, "methods": set()})
            entry["score"] += 1.0 / (k + rank + 1)
            entry["methods"].add(result.method)
    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata={**item["result"].metadata, "fusion_methods": sorted(item["methods"])},
            method="hybrid",
        )
        for item in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        try:
            self.dense.index(chunks)
        except Exception as exc:
            # BM25 remains a valid degraded retrieval path when Qdrant or the
            # embedding model is temporarily unavailable.
            print(f"  Warning: dense index unavailable; BM25-only mode: {exc}")

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
