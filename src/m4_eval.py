from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import asyncio
import math
import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAGAS_LLM_API_KEY,
    RAGAS_LLM_BASE_URL,
    RAGAS_LLM_MODEL,
    RAGAS_EMBEDDING_PROVIDER,
    RAGAS_EMBEDDING_API_KEY,
    RAGAS_EMBEDDING_BASE_URL,
    RAGAS_EMBEDDING_MODEL,
    TEST_SET_PATH,
)


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str],
                   enabled: bool = False) -> dict:
    """Run RAGAS evaluation."""
    empty = _empty_evaluation()
    if len({len(questions), len(answers), len(contexts), len(ground_truths)}) != 1:
        raise ValueError("questions, answers, contexts, and ground_truths must have equal lengths")
    if not questions or not enabled:
        return empty
    openai_embeddings = RAGAS_EMBEDDING_PROVIDER != "huggingface"
    if not RAGAS_LLM_API_KEY or (openai_embeddings and not RAGAS_EMBEDDING_API_KEY):
        print("  Warning: RAGAS skipped because no LLM API key is configured")
        return empty

    try:
        from langchain_openai import ChatOpenAI
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from ragas.run_config import RunConfig

        llm_kwargs = {"api_key": RAGAS_LLM_API_KEY}
        if RAGAS_LLM_BASE_URL:
            llm_kwargs["base_url"] = RAGAS_LLM_BASE_URL
        run_config = RunConfig(max_workers=1, max_retries=0, timeout=45)
        llm = LangchainLLMWrapper(
            ChatOpenAI(model=RAGAS_LLM_MODEL, temperature=0, **llm_kwargs),
            run_config=run_config,
        )
        # Only answer_relevancy uses embeddings; bge-m3 measures Vietnamese
        # relevancy faithfully where text-embedding-3-large compresses cosines.
        if openai_embeddings:
            from langchain_openai import OpenAIEmbeddings
            embedding_kwargs = {"api_key": RAGAS_EMBEDDING_API_KEY}
            if RAGAS_EMBEDDING_BASE_URL:
                embedding_kwargs["base_url"] = RAGAS_EMBEDDING_BASE_URL
            embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(model=RAGAS_EMBEDDING_MODEL, **embedding_kwargs)
            )
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            # Pin to CPU: retrieval (bge-m3) and the reranker already occupy the
            # GPU, and this only embeds ~60 short questions, so CPU avoids OOM.
            embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(
                    model_name=RAGAS_EMBEDDING_MODEL,
                    model_kwargs={"device": os.getenv("RAGAS_EMBEDDING_DEVICE", "cpu")},
                    encode_kwargs={"normalize_embeddings": True},
                )
            )
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        for metric in metrics:
            if hasattr(metric, "llm"):
                metric.llm = llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = embeddings
            metric.init(run_config)

        # ragas.evaluate() in 0.1.22 can leave an executor worker alive in
        # this environment. Calling the same RAGAS metric objects sequentially
        # gives deterministic resource use and an explicit per-metric timeout.
        per_question = []
        for question, answer, context, ground_truth in zip(questions, answers, contexts, ground_truths):
            row = {
                "question": question,
                "answer": answer,
                "contexts": [str(item) for item in context],
                "ground_truth": ground_truth,
            }
            values = asyncio.run(_score_row(metrics, row, timeout=run_config.timeout))
            per_question.append(EvalResult(
                question=question,
                answer=answer,
                contexts=row["contexts"],
                ground_truth=ground_truth,
                faithfulness=values["faithfulness"],
                answer_relevancy=values["answer_relevancy"],
                context_precision=values["context_precision"],
                context_recall=values["context_recall"],
            ))
        return {
            metric: _mean([getattr(item, metric) for item in per_question])
            for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        } | {"per_question": per_question}
    except Exception as exc:
        print(f"  Warning: RAGAS evaluation failed: {exc}")
        return empty


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": (
            "Câu trả lời có claim không được context hỗ trợ.",
            "Siết prompt grounded, giảm context nhiễu và chỉ sinh khi có bằng chứng.",
        ),
        "context_recall": (
            "Retriever không lấy được evidence cần thiết.",
            "Điều chỉnh chunking/query, giữ hybrid BM25+dense và bổ sung tài liệu thiếu.",
        ),
        "context_precision": (
            "Top contexts chứa quá nhiều đoạn không liên quan.",
            "Giảm top-k, tăng chất lượng reranking hoặc lọc theo metadata/version.",
        ),
        "answer_relevancy": (
            "Câu trả lời không trực tiếp giải quyết câu hỏi.",
            "Dùng prompt trả lời ngắn, yêu cầu trả lời đúng trọng tâm và kiểm tra intent.",
        ),
    }
    analyses = []
    for result in eval_results:
        values = {metric: _metric_value(getattr(result, metric)) for metric in diagnostic_tree}
        worst_metric = min(values, key=values.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyses.append({
            "question": result.question,
            "answer": result.answer,
            "ground_truth": result.ground_truth,
            "worst_metric": worst_metric,
            "score": round(_mean(list(values.values())), 4),
            "metrics": {metric: round(score, 4) for metric, score in values.items()},
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return sorted(analyses, key=lambda item: item["score"])[:max(bottom_n, 0)]


def _empty_evaluation() -> dict:
    return {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "per_question": [],
    }


def _metric_value(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


async def _score_row(metrics: list, row: dict, timeout: float) -> dict[str, float]:
    scores: dict[str, float] = {}
    for metric in metrics:
        try:
            scores[metric.name] = _metric_value(await metric.ascore(row, timeout=timeout))
        except Exception as exc:
            print(f"  Warning: RAGAS {metric.name} failed: {exc}")
            scores[metric.name] = 0.0
    return scores


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
