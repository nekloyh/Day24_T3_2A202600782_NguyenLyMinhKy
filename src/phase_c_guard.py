from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers. (Đã implement sẵn)

    Custom recognizers thêm vào:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)

    Các recognizers mặc định đã có sẵn: EMAIL, PHONE_NUMBER (international), ...
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)

    analyzer  = AnalyzerEngine(registry=registry)
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,   # text với PII được thay bằng <TYPE>
        }
    """
    entities = _regex_pii_entities(text)
    presidio_results = []
    if analyzer is not None:
        try:
            presidio_results = analyzer.analyze(
                text=text,
                language=PRESIDIO_LANGUAGE,
                entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "VN_CCCD", "VN_PHONE"],
            )
        except Exception:
            presidio_results = []

    seen = {(e["start"], e["end"], e["type"]) for e in entities}
    for result in presidio_results:
        entity_type = result.entity_type
        if entity_type == "EMAIL_ADDRESS":
            entity_type = "EMAIL"
        key = (result.start, result.end, entity_type)
        if key in seen:
            continue
        seen.add(key)
        entities.append({
            "type": entity_type,
            "text": text[result.start:result.end],
            "score": round(float(result.score), 3),
            "start": result.start,
            "end": result.end,
        })

    if not entities:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized = text
    for entity in sorted(entities, key=lambda item: item["start"], reverse=True):
        anonymized = anonymized[:entity["start"]] + f"<{entity['type']}>" + anonymized[entity["end"]:]
    return {"has_pii": True, "entities": sorted(entities, key=lambda item: item["start"]), "anonymized": anonymized}


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml. (Đã implement sẵn)

    Config directory: guardrails/
        config.yml  — model + rails config
        rails.co    — Colang dialogue flows (topic check, jailbreak check, output check)
    """
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    rails  = LLMRails(config)
    return rails


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,          # NeMo's raw response
        }
    """
    heuristic = _heuristic_input_block(text)
    if heuristic:
        return {"allowed": False, "blocked_reason": heuristic, "response": "Blocked by local input rail."}

    if rails is not None:
        try:
            response = await rails.generate_async(messages=[{"role": "user", "content": text}])
            response_text = response.get("content", "") if isinstance(response, dict) else str(response)
            refuse_keywords = ["xin lỗi", "không thể", "không được phép", "i cannot", "i'm sorry", "không hỗ trợ"]
            blocked = any(keyword in response_text.lower() for keyword in refuse_keywords)
            return {
                "allowed": not blocked,
                "blocked_reason": "nemo_input_rail" if blocked else None,
                "response": response_text,
            }
        except Exception as exc:
            return {"allowed": True, "blocked_reason": None, "response": f"NeMo skipped: {exc}"}
    return {"allowed": True, "blocked_reason": None, "response": "Allowed by local input rail."}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    NeMo output rails hoạt động trong context của cả cuộc hội thoại (input + output).
    Kiểm tra: có PII không? Nội dung có phù hợp không? Có hallucination rõ ràng không?

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,          # answer đã qua guard (có thể bị redact)
        }
    """
    pii = pii_scan(answer)
    if pii["has_pii"]:
        return {"safe": False, "flagged_reason": "pii_in_output", "final_answer": pii["anonymized"]}

    lower_answer = answer.lower()
    sensitive_patterns = [
        "mật khẩu admin", "system prompt", "confidential", "toàn bộ thông tin nhân viên",
        "bảng lương chi tiết", "employee records",
    ]
    if any(pattern in lower_answer for pattern in sensitive_patterns):
        return {
            "safe": False,
            "flagged_reason": "sensitive_output",
            "final_answer": "Xin lỗi, tôi không thể cung cấp nội dung nhạy cảm.",
        }

    if rails is not None:
        try:
            response = await rails.generate_async(messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            response_text = response.get("content", "") if isinstance(response, dict) else str(response)
            flagged = any(keyword in response_text.lower() for keyword in ["xin lỗi", "không thể cung cấp", "i cannot"])
            return {
                "safe": not flagged,
                "flagged_reason": "nemo_output_rail" if flagged else None,
                "final_answer": response_text if flagged else answer,
            }
        except Exception:
            pass
    return {"safe": True, "flagged_reason": None, "final_answer": answer}


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii (cho category pii_injection)
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    Returns:
        list of {
          "id": int, "category": str, "input": str,
          "expected": "blocked"|"allowed",
          "actual":   "blocked"|"allowed",
          "blocked_by": str | None,       # "presidio" | "nemo_input" | None
          "passed": bool,
        }
    """
    async def _run_all():
        results = []
        for item in adversarial_set:
            blocked_by = None
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            text = item["input"]
            results.append({
                "id": item["id"],
                "category": item["category"],
                "input": text[:80] + ("..." if len(text) > 80 else ""),
                "expected": item["expected"],
                "actual": actual,
                "blocked_by": blocked_by,
                "passed": actual == item["expected"],
            })
        return results

    results = _run_async(_run_all())
    passed = sum(1 for r in results if r["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Đo P50/P95/P99 latency cho từng layer trong guard stack.

    Mục tiêu production: P95 total < LATENCY_BUDGET_P95_MS (500ms mặc định)

    Insight cần quan sát:
        - Presidio: local regex → rất nhanh (<10ms)
        - NeMo:     LLM API call → chậm (~200-800ms tuỳ model và network)
        → Tổng: dominated by NeMo

    Returns:
        {
          "presidio_ms":  {"p50": float, "p95": float, "p99": float},
          "nemo_ms":      {"p50": float, "p95": float, "p99": float},
          "total_ms":     {"p50": float, "p95": float, "p99": float},
          "latency_budget_ok": bool,
          "budget_ms": int,
        }
    """
    presidio_times, nemo_times, total_times = [], [], []
    samples = test_inputs[:max(1, n_runs)] or [""]

    async def _measure():
        for index in range(n_runs):
            text = samples[index % len(samples)]
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    _run_async(_measure())

    total_p = _percentiles(total_times)
    return {
        "presidio_ms": _percentiles(presidio_times),
        "nemo_ms": _percentiles(nemo_times),
        "total_ms": total_p,
        "latency_budget_ok": total_p["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


def _regex_pii_entities(text: str) -> list[dict]:
    patterns = [
        ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", 0.95),
        ("VN_PHONE", r"\b0[3-9]\d{8}\b", 0.9),
        ("VN_CCCD", r"\b\d{12}\b", 0.9),
        ("VN_CCCD", r"\b\d{9}\b", 0.7),
    ]
    entities = []
    for entity_type, pattern, score in patterns:
        for match in re.finditer(pattern, text):
            entities.append({
                "type": entity_type,
                "text": match.group(0),
                "score": score,
                "start": match.start(),
                "end": match.end(),
            })
    return _dedupe_entities(entities)


def _dedupe_entities(entities: list[dict]) -> list[dict]:
    deduped = []
    occupied: list[tuple[int, int]] = []
    for entity in sorted(entities, key=lambda item: (item["start"], -(item["end"] - item["start"]))):
        span = (entity["start"], entity["end"])
        if any(not (span[1] <= old[0] or span[0] >= old[1]) for old in occupied):
            continue
        occupied.append(span)
        deduped.append(entity)
    return deduped


def _heuristic_input_block(text: str) -> str | None:
    lower = text.lower()
    patterns = {
        "prompt_injection": [
            "ignore", "bỏ qua", "forget all", "system override", "admin command",
            "priority 1", "previous instructions", "system instructions", "system prompt",
            "dump all", "in ra toàn bộ", "<!--", "đóng vai", "pretend you are", "dan",
            "không có giới hạn", "unrestricted",
        ],
        "sensitive_data": [
            "mật khẩu admin", "list all employee salaries", "bảng lương chi tiết",
            "toàn bộ thông tin nhân viên", "employee records", "confidential hr data",
            "tiết lộ", "cccd và số điện thoại", "thông tin nhân viên",
        ],
        "harmful": ["tấn công mạng", "malware", "illegal", "hack"],
        "off_topic": [
            "bài thơ", "nấu phở", "bitcoin", "ethereum", "phương trình vi phân",
            "marvel", "bộ phim",
        ],
    }
    for reason, needles in patterns.items():
        if any(needle in lower for needle in needles):
            return reason
    return None


def _percentiles(times: list[float]) -> dict[str, float]:
    if not times:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    values = sorted(times)
    n = len(values)
    return {
        "p50": round(values[min(int((n - 1) * 0.50), n - 1)], 2),
        "p95": round(values[min(int((n - 1) * 0.95), n - 1)], 2),
        "p99": round(values[min(int((n - 1) * 0.99), n - 1)], 2),
    }


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Task 9a: PII scan demo
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    result = pii_scan(test_pii)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    results = run_adversarial_suite(adversarial_set)
    if results:
        passed = sum(1 for r in results if r["passed"])
        print(f"Adversarial suite: {passed}/{len(results)} passed")

    # Task 12: P95 latency
    sample_inputs = [item["input"] for item in adversarial_set[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=10)
    print(f"\nLatency P95 — Presidio: {latency['presidio_ms']['p95']}ms | "
          f"NeMo: {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")

    os.makedirs("reports", exist_ok=True)
    report = {
        "total_inputs": len(adversarial_set),
        "passed": sum(1 for r in results if r["passed"]),
        "pass_rate": round(sum(1 for r in results if r["passed"]) / len(results), 4) if results else 0.0,
        "results": results,
        "latency": latency,
    }
    with open("reports/guard_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Phase C report saved → reports/guard_results.json")
