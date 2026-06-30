from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    OCR_CACHE_DIR,
    OCR_DPI,
    OCR_ENABLED,
    OCR_TESSDATA_DIR,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract the embedded text layer, if a PDF has one."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


@lru_cache(maxsize=1)
def _tesseract_languages() -> set[str]:
    """Return installed Tesseract language packs without assuming Vietnamese exists."""
    if not shutil.which("tesseract"):
        return set()
    command = ["tesseract"]
    if Path(OCR_TESSDATA_DIR).is_dir():
        command.extend(["--tessdata-dir", OCR_TESSDATA_DIR])
    result = subprocess.run(command + ["--list-langs"], capture_output=True, text=True, check=False)
    languages = {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith("List of available")
    }
    return languages


def _ocr_pdf_text(path: str) -> tuple[str, dict]:
    """OCR a scanned PDF page by page and cache the extracted text by file hash."""
    if not OCR_ENABLED:
        return "", {}
    if not shutil.which("pdftoppm") or not shutil.which("tesseract"):
        print(f"  Warning: cannot OCR {os.path.basename(path)}; pdftoppm/tesseract is unavailable")
        return "", {}

    languages = _tesseract_languages()
    language = "vie+eng" if {"vie", "eng"}.issubset(languages) else "eng"
    file_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    cache_path = Path(OCR_CACHE_DIR) / f"{file_hash}-{language.replace('+', '-')}-{OCR_DPI}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8"), {"ocr_cached": True, "ocr_language": language}
    try:
        from pypdf import PdfReader

        page_count = len(PdfReader(path).pages)
    except Exception as exc:
        print(f"  Warning: cannot inspect {os.path.basename(path)} for OCR: {exc}")
        return "", {}

    pages: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="lab18-ocr-") as tmpdir:
            for page_number in range(1, page_count + 1):
                prefix = os.path.join(tmpdir, f"page-{page_number}")
                subprocess.run(
                    [
                        "pdftoppm", "-f", str(page_number), "-l", str(page_number),
                        "-r", str(OCR_DPI), "-png", "-singlefile", path, prefix,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                command = ["tesseract", f"{prefix}.png", "stdout"]
                if Path(OCR_TESSDATA_DIR).is_dir():
                    command.extend(["--tessdata-dir", OCR_TESSDATA_DIR])
                result = subprocess.run(
                    command + ["-l", language, "--psm", "6"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                text = result.stdout.strip()
                if text:
                    pages.append(f"# PDF page {page_number}\n{text}")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown OCR error").strip()
        print(f"  Warning: OCR failed for {os.path.basename(path)}: {detail}")
        return "", {}

    text = "\n\n".join(pages).strip()
    if text:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        print(f"  OCR extracted {page_count} pages from {os.path.basename(path)} ({language})")
    return text, {"ocr_cached": False, "ocr_language": language, "page_count": page_count}


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load markdown, text PDFs, and scanned PDFs through cached OCR."""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        metadata = {"source": os.path.basename(fp), "document_type": "pdf"}
        if len(text) >= 80:
            docs.append({"text": text, "metadata": {**metadata, "extraction": "text-layer"}})
            continue
        text, ocr_metadata = _ocr_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {**metadata, "extraction": "ocr", **ocr_metadata}})
        else:
            print(f"  Warning: skipped {os.path.basename(fp)}; no PDF text layer or OCR output")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\s*\n", text) if s.strip()]
    if not sentences:
        return []

    try:
        import numpy as np

        embeddings = _semantic_encoder().encode(sentences, convert_to_numpy=True, show_progress_bar=False)
        similarities = [
            float(np.dot(embeddings[index - 1], embeddings[index]) /
                  (np.linalg.norm(embeddings[index - 1]) * np.linalg.norm(embeddings[index]) + 1e-9))
            for index in range(1, len(sentences))
        ]
    except Exception as exc:
        # A deterministic fallback keeps the pipeline useful on machines without
        # the semantic model, while making the degraded mode visible.
        print(f"  Warning: semantic encoder unavailable, using paragraph fallback: {exc}")
        return [
            Chunk(chunk.text, {**chunk.metadata, **metadata, "strategy": "semantic-fallback"})
            for chunk in chunk_basic(text, metadata=metadata)
        ]

    groups: list[list[str]] = [[sentences[0]]]
    for sentence, similarity in zip(sentences[1:], similarities):
        current_length = len(" ".join(groups[-1]))
        if similarity < threshold and current_length >= 80:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)

    return [
        Chunk(" ".join(group), {**metadata, "strategy": "semantic", "chunk_index": index})
        for index, group in enumerate(groups)
    ]


@lru_cache(maxsize=1)
def _semantic_encoder():
    """Load the lightweight semantic chunking model only once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return [], []

    parent_texts = _pack_units(paragraphs, parent_size)
    source = str(metadata.get("source", "document")).replace(" ", "_")
    parents: list[Chunk] = []
    children: list[Chunk] = []

    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"{source}:parent_{parent_index}"
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": parent_id, "chunk_index": parent_index},
        ))
        child_units = _split_to_limit(parent_text, child_size)
        for child_index, child_text in enumerate(child_units):
            children.append(Chunk(
                text=child_text,
                metadata={
                    **metadata,
                    "chunk_type": "child",
                    "parent_id": parent_id,
                    "child_index": child_index,
                    "strategy": "hierarchical",
                },
                parent_id=parent_id,
            ))
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    chunks: list[Chunk] = []
    section_path: list[str] = []
    content: list[str] = []

    def flush() -> None:
        if not content:
            return
        body = "\n".join(content).strip()
        if not body:
            return
        section = " > ".join(section_path) if section_path else "Document"
        prefix = "\n".join(f"# {header}" for header in section_path)
        chunks.append(Chunk(
            text=f"{prefix}\n\n{body}".strip(),
            metadata={**metadata, "section": section, "strategy": "structure", "chunk_index": len(chunks)},
        ))

    for line in text.splitlines():
        match = re.match(r"^(#{1,3})\s+(.+?)\s*$", line)
        if not match:
            content.append(line)
            continue
        flush()
        content = []
        level = len(match.group(1))
        title = match.group(2)
        section_path = section_path[:level - 1]
        section_path.append(title)
    flush()
    return chunks or [Chunk(text.strip(), {**metadata, "section": "Document", "strategy": "structure"})]


def _pack_units(units: list[str], limit: int) -> list[str]:
    """Pack paragraphs into bounded chunks without splitting a paragraph first."""
    packed: list[str] = []
    current: list[str] = []
    current_length = 0
    for unit in units:
        if len(unit) > limit:
            if current:
                packed.append("\n\n".join(current))
                current, current_length = [], 0
            packed.extend(_split_to_limit(unit, limit))
            continue
        separator = 2 if current else 0
        if current and current_length + separator + len(unit) > limit:
            packed.append("\n\n".join(current))
            current, current_length = [], 0
        current.append(unit)
        current_length += separator + len(unit)
    if current:
        packed.append("\n\n".join(current))
    return packed


def _split_to_limit(text: str, limit: int) -> list[str]:
    """Split long text at word boundaries, preserving all source text."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    remaining = text.strip()
    while len(remaining) > limit:
        cut = remaining.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
