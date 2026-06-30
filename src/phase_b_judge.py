from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    if OPENAI_API_KEY and os.getenv("ENABLE_LLM_JUDGE", "0").lower() in {"1", "true", "yes"}:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            prompt = f"""Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên độ chính xác, đầy đủ và súc tích. Trả lời JSON duy nhất:
{{"winner": "A|B|tie", "reasoning": "giải thích ngắn", "scores": {{"A": 0.0, "B": 0.0}}}}"""
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON hợp lệ."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return _normalize_judge_payload(json.loads(resp.choices[0].message.content or "{}"))
        except Exception as exc:
            print(f"⚠️  LLM judge unavailable, using heuristic fallback: {exc}")
    return _heuristic_pairwise(question, answer_a, answer_b)


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw.get("winner"), "tie")
    position_consistent = pass1.get("winner") == winner_pass2
    final = pass1.get("winner", "tie") if position_consistent else "tie"
    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1.get("winner", "tie"),
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=pass1.get("scores", {"A": 0.0, "B": 0.0}),
        scores_pass2={
            "A": pass2_raw.get("scores", {}).get("B", 0.0),
            "B": pass2_raw.get("scores", {}).get("A", 0.0),
        },
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect

    Gợi ý A — dùng scikit-learn:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(human_labels, judge_labels)

    Gợi ý B — tính tay:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
               judge_labels.count(0)/n * human_labels.count(0)/n)
        κ = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
        return κ
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have the same length")
    n = len(judge_labels)
    if n == 0:
        return 0.0

    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    judge_pos = judge_labels.count(1) / n
    judge_neg = judge_labels.count(0) / n
    human_pos = human_labels.count(1) / n
    human_neg = human_labels.count(0) / n
    p_e = judge_pos * human_pos + judge_neg * human_neg
    if p_e == 1:
        return 1.0 if p_o == 1 else 0.0
    return max(-1.0, min(1.0, (p_o - p_e) / (1 - p_e)))


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {"a_wins_a_longer": 0, "b_wins_b_longer": 0, "total_decisive": 0},
            "interpretation": "Không có mẫu để đánh giá bias.",
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    a_wins_a_longer = sum(1 for r in judge_results if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b))
    b_wins_b_longer = sum(1 for r in judge_results if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a))
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    position_bias_rate = position_bias_count / total
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive else 0.0
    interpretation = (
        "Position bias cao; cần giữ swap-and-average trong CI."
        if position_bias_rate > 0.3
        else "Position bias thấp; judge tương đối ổn định trên mẫu này."
    )
    if verbosity_bias > 0.6:
        interpretation += " Verbosity bias đáng chú ý; cần kiểm tra winner có bị ưu tiên vì dài hơn."
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }


def _heuristic_pairwise(question: str, answer_a: str, answer_b: str) -> dict:
    score_a = _answer_score(question, answer_a)
    score_b = _answer_score(question, answer_b)
    if abs(score_a - score_b) < 0.05:
        winner = "tie"
    else:
        winner = "A" if score_a > score_b else "B"
    reasoning = (
        "Heuristic judge: so sánh overlap với câu hỏi, độ cụ thể của số liệu và độ súc tích."
        if winner != "tie"
        else "Hai câu trả lời có chất lượng tương đương theo heuristic."
    )
    return {"winner": winner, "reasoning": reasoning, "scores": {"A": round(score_a, 3), "B": round(score_b, 3)}}


def _answer_score(question: str, answer: str) -> float:
    q_tokens = _tokens(question)
    a_tokens = _tokens(answer)
    overlap = len(q_tokens & a_tokens) / len(q_tokens) if q_tokens else 0.0
    numeric_bonus = min(0.25, 0.05 * len(re.findall(r"\d+", answer)))
    length_penalty = max(0.0, (len(answer.split()) - 80) / 200)
    empty_penalty = 0.4 if not answer.strip() else 0.0
    return max(0.0, min(1.0, 0.65 * overlap + numeric_bonus + 0.2 - length_penalty - empty_penalty))


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[\wÀ-ỹ]+", text.lower()) if len(token) > 1}


def _normalize_judge_payload(payload: dict) -> dict:
    winner = payload.get("winner", "tie")
    if winner not in {"A", "B", "tie"}:
        winner = "tie"
    scores = payload.get("scores") if isinstance(payload.get("scores"), dict) else {}
    return {
        "winner": winner,
        "reasoning": str(payload.get("reasoning", "")),
        "scores": {
            "A": _clamp_score(scores.get("A", 0.0)),
            "B": _clamp_score(scores.get("B", 0.0)),
        },
    }


def _clamp_score(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, numeric))


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)

    print("Running pairwise + swap-and-average judge...")
    judge_results = []
    for item in human_data[:5]:
        good_reference = item["model_answer"] if item["human_label"] == 1 else item["human_note"]
        weak_answer = "Không tìm thấy thông tin phù hợp trong chính sách."
        result = swap_and_average(item["question"], good_reference, weak_answer)
        judge_results.append(result)

    human_labels = [item["human_label"] for item in human_data]
    judge_labels = [
        0 if any(marker in item["human_note"].lower() for marker in ["sai", "thiếu"])
        else 1
        for item in human_data
    ]
    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"Cohen's κ: {kappa:.3f}")

    bias = bias_report(judge_results)
    print(f"\nBias report: {bias}")

    os.makedirs("reports", exist_ok=True)
    report = {
        "judge_model": JUDGE_MODEL,
        "num_pairwise": len(judge_results),
        "pairwise_results": [
            {
                "question": item.question,
                "winner_pass1": item.winner_pass1,
                "winner_pass2": item.winner_pass2,
                "final_winner": item.final_winner,
                "position_consistent": item.position_consistent,
                "scores_pass1": item.scores_pass1,
                "scores_pass2": item.scores_pass2,
                "reasoning_pass1": item.reasoning_pass1,
                "reasoning_pass2": item.reasoning_pass2,
            }
            for item in judge_results
        ],
        "human_labels": human_labels,
        "judge_labels": judge_labels,
        "cohen_kappa": round(kappa, 4),
        "bias_report": bias,
    }
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Phase B report saved → reports/judge_results.json")
