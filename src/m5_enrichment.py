from __future__ import annotations

"""Chunk enrichment for recall and traceability before vector indexing."""

import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LLM_API_KEY
from src.llm import chat_completion, chat_json


@dataclass
class EnrichedChunk:
    """A chunk plus retrieval-oriented information derived from it."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def summarize_chunk(text: str, use_llm: bool = False) -> str:
    """Create a short summary while retaining numbers and conditions."""
    if use_llm and LLM_API_KEY:
        result = chat_completion(
            "Tóm tắt đoạn văn trong tối đa hai câu ngắn. Giữ nguyên số liệu, phiên bản và điều kiện quan trọng.",
            text,
            max_tokens=150,
        )
        if result:
            return result
    sentences = _sentences(text)
    return " ".join(sentences[:2]) if sentences else text.strip()


def generate_hypothesis_questions(text: str, n_questions: int = 3, use_llm: bool = False) -> list[str]:
    """Generate questions whose wording can bridge query and document vocabulary."""
    if n_questions <= 0:
        return []
    if use_llm and LLM_API_KEY:
        result = chat_completion(
            f"Tạo đúng {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời. Mỗi câu hỏi một dòng, không đánh số.",
            text,
            max_tokens=200,
        )
        if result:
            questions = [_clean_question(line) for line in result.splitlines()]
            return [question for question in questions if question][:n_questions]
    return [
        f"Thông tin nào được nêu về: {sentence.rstrip('.!?')}?"
        for sentence in _sentences(text)[:n_questions]
    ]


def contextual_prepend(text: str, document_title: str = "", use_llm: bool = False) -> str:
    """Prepend a compact statement of a chunk's position and topic."""
    context = ""
    if use_llm and LLM_API_KEY:
        context = chat_completion(
            "Viết đúng một câu mô tả chủ đề và vai trò của đoạn trích trong tài liệu. Không thêm facts ngoài đoạn trích.",
            f"Tài liệu: {document_title or 'không rõ'}\n\nĐoạn trích:\n{text}",
            max_tokens=80,
        ) or ""
    if not context:
        context = f"Đoạn trích từ {document_title or 'tài liệu nội bộ'}, thuộc chủ đề {_topic_for(text)}."
    return f"{context}\n\n{text}"


def extract_metadata(text: str, use_llm: bool = False) -> dict:
    """Extract constrained metadata for filtering and failure analysis."""
    if use_llm and LLM_API_KEY:
        result = chat_json(
            'Trích xuất JSON đúng schema: {"topic": string, "entities": [string], "category": "policy|hr|it|finance|safety|general", "language": "vi|en"}. Không thêm key khác.',
            text,
            max_tokens=150,
        )
        if result:
            return _normalise_metadata(result, text)

    entities = re.findall(r"\b(?:[A-ZĐ][\wÀ-ỹ]+(?:\s+[A-ZĐ][\wÀ-ỹ]+)*)\b", text)
    return {
        "topic": _topic_for(text),
        "entities": list(dict.fromkeys(entities))[:8],
        "category": _category_for(text),
        "language": "vi" if re.search(r"[à-ỹĐđ]", text) else "en",
    }


def _enrich_single_call(text: str, source: str, use_llm: bool = False) -> dict:
    """Use one structured LLM call for all enrichment fields when enabled."""
    if use_llm and LLM_API_KEY:
        result = chat_json(
            """Phân tích đoạn trích và trả về JSON đúng schema:
{"summary":"tối đa 2 câu", "questions":["3 câu hỏi"], "context":"một câu vị trí/chủ đề", "metadata":{"topic":"...", "entities":["..."], "category":"policy|hr|it|finance|safety|general", "language":"vi|en"}}.
Giữ nguyên số liệu, phiên bản và phủ định; không suy đoán facts ngoài đoạn trích.""",
            f"Tài liệu: {source or 'không rõ'}\n\nĐoạn trích:\n{text}",
            max_tokens=400,
        )
        if result:
            questions = result.get("questions", [])
            return {
                "summary": str(result.get("summary", "")),
                "questions": [str(question) for question in questions if str(question).strip()][:3]
                if isinstance(questions, list) else [],
                "context": str(result.get("context", "")),
                "metadata": _normalise_metadata(result.get("metadata", {}), text),
            }

    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": f"Đoạn trích từ {source or 'tài liệu nội bộ'}, thuộc chủ đề {_topic_for(text)}.",
        "metadata": extract_metadata(text),
    }


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
    use_llm: bool = False,
) -> list[EnrichedChunk]:
    """Enrich chunks with either the single-call or individually selectable modes.

    ``use_llm`` is opt-in so unit tests and local inspection do not silently
    incur API costs. The production pipeline enables it from configuration.
    """
    methods = methods or ["combined"]
    use_combined = "combined" in methods
    enriched: list[EnrichedChunk] = []

    for index, chunk in enumerate(chunks):
        text = str(chunk["text"])
        metadata = dict(chunk.get("metadata", {}))
        source = str(metadata.get("source", ""))
        if use_combined:
            result = _enrich_single_call(text, source, use_llm=use_llm)
            summary = result["summary"]
            questions = result["questions"]
            context = result["context"]
            enriched_text = f"{context}\n\n{text}" if context else text
            auto_metadata = result["metadata"]
        else:
            summary = summarize_chunk(text, use_llm=use_llm) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text, use_llm=use_llm) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source, use_llm=use_llm) if "contextual" in methods else text
            auto_metadata = extract_metadata(text, use_llm=use_llm) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**metadata, **auto_metadata},
            method="+".join(methods),
        ))
        if (index + 1) % 10 == 0 or index + 1 == len(chunks):
            print(f"  Enriched {index + 1}/{len(chunks)} chunks...", flush=True)
    return enriched


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", text) if sentence.strip()]


def _clean_question(question: str) -> str:
    question = re.sub(r"^\s*\d+[.)\-:]?\s*", "", question).strip()
    if question and not question.endswith("?"):
        question += "?"
    return question


def _topic_for(text: str) -> str:
    lowered = text.lower()
    topics = {
        "nghỉ phép": ("nghỉ", "phép"),
        "mật khẩu và truy cập": ("mật khẩu", "vpn", "mfa"),
        "lương và phúc lợi": ("lương", "phụ cấp", "bảo hiểm"),
        "mua sắm và chi phí": ("mua", "chi phí", "tạm ứng", "thanh toán"),
        "an toàn và bảo mật": ("malware", "an toàn", "bảo mật", "sự cố"),
    }
    for topic, keywords in topics.items():
        if any(keyword in lowered for keyword in keywords):
            return topic
    return "chính sách nội bộ"


def _category_for(text: str) -> str:
    topic = _topic_for(text)
    if topic == "mật khẩu và truy cập":
        return "it"
    if topic in {"lương và phúc lợi", "mua sắm và chi phí"}:
        return "finance"
    if topic == "an toàn và bảo mật":
        return "safety"
    return "hr" if topic == "nghỉ phép" else "policy"


def _normalise_metadata(value: object, text: str) -> dict:
    metadata = value if isinstance(value, dict) else {}
    entities = metadata.get("entities", [])
    if not isinstance(entities, list):
        entities = []
    return {
        "topic": str(metadata.get("topic") or _topic_for(text)),
        "entities": [str(entity) for entity in entities][:8],
        "category": str(metadata.get("category") or _category_for(text)),
        "language": str(metadata.get("language") or ("vi" if re.search(r"[à-ỹĐđ]", text) else "en")),
    }


if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm."
    print(enrich_chunks([{"text": sample, "metadata": {"source": "demo.md"}}]))
