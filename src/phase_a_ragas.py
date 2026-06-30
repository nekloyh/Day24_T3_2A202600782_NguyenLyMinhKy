from __future__ import annotations

"""Phase A: RAGAS Production Evaluation — 50q, 3 distributions, cluster analysis."""

import json
import math
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH

Distribution = str  # "factual" | "multi_hop" | "adversarial"

DIAGNOSTIC_TREE = {
    "faithfulness":      ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall":    ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy":  ("Answer doesn't match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }
        return min(scores, key=scores.get)


# ─── Đã implement sẵn ────────────────────────────────────────────────────────

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set với 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers từ setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json không tìm thấy tại {path}\n"
            "→ Chạy trước: python setup_answers.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_phase_a_report(results: list[RagasResult], clusters: dict,
                         path: str = "reports/ragas_50q.json") -> None:
    """Save Phase A report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness":      sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy":  sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall":    sum(r.context_recall for r in subset) / len(subset),
                "avg_score":         sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": [
            {"rank": i + 1, "question_id": r.question_id, "distribution": r.distribution,
             "question": r.question, "avg_score": round(r.avg_score, 4),
             "worst_metric": r.worst_metric}
            for i, r in enumerate(sorted(results, key=lambda x: x.avg_score)[:10])
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase A report saved → {path}")


# ─── Tasks 1-4: Sinh viên implement ──────────────────────────────────────────

def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Nhóm 50 câu hỏi theo 3 distributions.

    Returns:
        {"factual": [...], "multi_hop": [...], "adversarial": [...]}
    """
    groups = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        distribution = item.get("distribution")
        if distribution not in groups:
            raise ValueError(f"Unknown distribution: {distribution!r}")
        groups[distribution].append(item)
    return groups


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    """Task 2: Chạy RAGAS 4 metrics trên toàn bộ 50 câu hỏi.

    Gợi ý — import từ Day 18 của bạn:
        from src.m4_eval import evaluate_ragas

    Steps:
        1. Extract questions, answers, contexts, ground_truths từ answers list
        2. Gọi evaluate_ragas() từ m4_eval.py
        3. Kết hợp kết quả với distribution info từ answers list
        4. Return list[RagasResult]
    """
    if not answers:
        return []

    questions = [a.get("question", "") for a in answers]
    ans_texts = [a.get("answer", "") for a in answers]
    contexts = [a.get("contexts", []) for a in answers]
    ground_truths = [a.get("ground_truth", "") for a in answers]

    per_q = []
    try:
        from config import ENABLE_RAGAS
        from src.m4_eval import evaluate_ragas
        raw = evaluate_ragas(questions, ans_texts, contexts, ground_truths, enabled=ENABLE_RAGAS)
        per_q = raw.get("per_question", [])
    except Exception as exc:
        print(f"⚠️  RAGAS unavailable, using lexical fallback scores: {exc}")

    results: list[RagasResult] = []
    for index, answer_item in enumerate(answers):
        pq = per_q[index] if index < len(per_q) else None
        if pq is None:
            scores = _fallback_scores(
                answer_item.get("question", ""),
                answer_item.get("answer", ""),
                answer_item.get("contexts", []),
                answer_item.get("ground_truth", ""),
            )
        else:
            scores = {
                "faithfulness": _clean_score(getattr(pq, "faithfulness", 0.0)),
                "answer_relevancy": _clean_score(getattr(pq, "answer_relevancy", 0.0)),
                "context_precision": _clean_score(getattr(pq, "context_precision", 0.0)),
                "context_recall": _clean_score(getattr(pq, "context_recall", 0.0)),
            }
        results.append(RagasResult(
            question_id=answer_item.get("id", index + 1),
            distribution=answer_item.get("distribution", ""),
            question=answer_item.get("question", ""),
            answer=answer_item.get("answer", ""),
            contexts=answer_item.get("contexts", []),
            ground_truth=answer_item.get("ground_truth", ""),
            faithfulness=scores["faithfulness"],
            answer_relevancy=scores["answer_relevancy"],
            context_precision=scores["context_precision"],
            context_recall=scores["context_recall"],
        ))
    return results


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Lấy 10 câu hỏi có avg_score thấp nhất.

    Returns:
        [{"rank": 1, "question_id": ..., "distribution": ...,
          "question": ..., "avg_score": ..., "worst_metric": ...,
          "diagnosis": ..., "suggested_fix": ...}, ...]
    """
    output = []
    for rank, result in enumerate(sorted(results, key=lambda r: r.avg_score)[:10], start=1):
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[result.worst_metric]
        output.append({
            "rank": rank,
            "question_id": result.question_id,
            "distribution": result.distribution,
            "question": result.question,
            "avg_score": round(result.avg_score, 4),
            "worst_metric": result.worst_metric,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Task 4: Phân tích failure clusters theo (worst_metric × distribution).

    Mục tiêu: tìm ra distribution nào hay bị failure nhất và metric nào yếu nhất.

    Returns:
        {
          "matrix": {
            "faithfulness":      {"factual": 3, "multi_hop": 5, "adversarial": 2},
            "answer_relevancy":  {...},
            "context_precision": {...},
            "context_recall":    {...},
          },
          "dominant_failure_distribution": "multi_hop",
          "dominant_failure_metric": "context_recall",
          "insight": "..."
        }
    """
    matrix = {
        metric: {"factual": 0, "multi_hop": 0, "adversarial": 0}
        for metric in DIAGNOSTIC_TREE
    }
    for result in results:
        if result.distribution in matrix[result.worst_metric]:
            matrix[result.worst_metric][result.distribution] += 1

    distributions = ["factual", "multi_hop", "adversarial"]
    dominant_dist = max(distributions, key=lambda d: sum(matrix[m][d] for m in matrix))
    dominant_metric = max(matrix, key=lambda m: sum(matrix[m].values()))
    insight = (
        f"Distribution '{dominant_dist}' có nhiều failure nhất; "
        f"metric '{dominant_metric}' là điểm yếu chủ đạo. "
        f"Gợi ý ưu tiên: {DIAGNOSTIC_TREE[dominant_metric][1]}."
    )
    return {
        "matrix": matrix,
        "dominant_failure_distribution": dominant_dist,
        "dominant_failure_metric": dominant_metric,
        "insight": insight,
    }


def _fallback_scores(question: str, answer: str, contexts: list[str], ground_truth: str) -> dict[str, float]:
    answer_tokens = _tokens(answer)
    question_tokens = _tokens(question)
    ground_truth_tokens = _tokens(ground_truth)
    context_tokens = _tokens("\n".join(str(c) for c in contexts))

    answer_relevancy = _overlap(answer_tokens, question_tokens | ground_truth_tokens)
    context_recall = _overlap(ground_truth_tokens, context_tokens)
    context_precision = _overlap(context_tokens, question_tokens | ground_truth_tokens)
    faithfulness = _overlap(answer_tokens, context_tokens | ground_truth_tokens)
    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\wÀ-ỹ]+", str(text).lower())
        if len(token) > 1
    }


def _overlap(source: set[str], target: set[str]) -> float:
    if not source:
        return 0.0
    return round(len(source & target) / len(source), 4)


def _clean_score(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return min(1.0, max(0.0, numeric))


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_set = load_test_set_50q()
    print(f"Loaded {len(test_set)} questions")

    groups = group_by_distribution(test_set)
    for dist, qs in groups.items():
        print(f"  {dist}: {len(qs)} questions")

    answers = load_answers()
    results = run_ragas_50q(answers)

    if results:
        b10 = bottom_10(results)
        clusters = cluster_analysis(results)
        save_phase_a_report(results, clusters)
        print("\nBottom 10 worst questions:")
        for item in b10:
            print(f"  #{item['rank']} [{item['distribution']}] {item['question'][:50]}... "
                  f"avg={item['avg_score']:.3f} worst={item['worst_metric']}")
        print(f"\nDominant failure: {clusters.get('dominant_failure_distribution')} / "
              f"{clusters.get('dominant_failure_metric')}")
    else:
        print("⚠️  No results — implement run_ragas_50q() first.")
