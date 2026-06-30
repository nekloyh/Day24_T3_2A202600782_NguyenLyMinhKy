"""Deterministic tools used before answer generation for high-risk intents."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable


@dataclass(frozen=True)
class AnswerTool:
    name: str
    description: str
    handler: Callable[[str, list[str]], str | None]


def plan_retrieval_queries(question: str) -> list[str]:
    """Decompose known multi-hop policy questions into coverage-oriented queries."""
    lowered = question.lower()
    plan = [question]
    if "senior" in lowered and "lương" in lowered and ("phép" in lowered or "thâm niên" in lowered):
        plan.extend([
            "chính sách nghỉ phép năm thâm niên",
            "dải lương Senior P3 P4",
        ])
    elif any(token in lowered for token in ("laptop", "thiết bị", "mua sắm")) and "phê duyệt" in lowered:
        plan.extend([
            "quy trình mua sắm mức phê duyệt theo giá trị",
            "mua sắm thiết bị CNTT xác nhận cấu hình kỹ thuật",
            "mua sắm trên 10 triệu tối thiểu 3 báo giá",
        ])
    elif "mentor" in lowered and "buddy" in lowered:
        plan.extend([
            "mentor và buddy nhân viên mới phải khác người",
            "quản lý trực tiếp làm mentor hoặc buddy",
        ])
    return list(dict.fromkeys(plan))


def calculate_advance_penalty(question: str, contexts: list[str]) -> str | None:
    """Calculate late advance settlement fees from policy values in retrieved context."""
    lowered = question.lower()
    if "tạm ứng" not in lowered or not any(token in lowered for token in ("phạt", "phí", "thanh toán")):
        return None

    source = re.sub(r"[*_`#]", "", "\n".join(contexts))
    amount_match = re.search(r"tạm ứng\s+([\d.,]+)\s*(triệu|trieu|vnđ|vnd)?", question, re.IGNORECASE)
    days_match = re.search(r"sau\s+(\d+)\s+ngày", question, re.IGNORECASE)
    deadline_match = re.search(
        r"(?:thời hạn thanh toán|trong vòng)[^\d]{0,100}(\d+)\s+ngày",
        source,
        re.IGNORECASE | re.DOTALL,
    )
    rate_match = re.search(r"(\d+(?:[,.]\d+)?)\s*%\s*/\s*tháng", source, re.IGNORECASE)
    if not all((amount_match, days_match, deadline_match, rate_match)):
        return None

    amount = _money_to_vnd(amount_match.group(1), amount_match.group(2) or "")
    days_elapsed = int(days_match.group(1))
    deadline = int(deadline_match.group(1))
    monthly_rate = float(rate_match.group(1).replace(",", ".")) / 100
    if amount is None or days_elapsed <= deadline:
        return None

    days_late = days_elapsed - deadline
    monthly_fee = round(amount * monthly_rate)
    prorated_fee = round(monthly_fee * days_late / 30)
    return (
        f"Thời hạn thanh toán là {deadline} ngày. Bạn thanh toán sau {days_elapsed} ngày, "
        f"nên quá hạn {days_late} ngày. Phí là {_format_vnd(monthly_fee)}/tháng "
        f"({rate_match.group(1)}% trên {_format_vnd(amount)}); tính pro-rata 30 ngày, "
        f"phải trả khoảng {_format_vnd(prorated_fee)} cho {days_late} ngày quá hạn."
    )


ANSWER_TOOLS = (
    AnswerTool(
        name="calculate_advance_penalty",
        description="Compute a late advance-settlement fee from retrieved policy values.",
        handler=calculate_advance_penalty,
    ),
)


def invoke_answer_tools(question: str, contexts: list[str]) -> str | None:
    """Invoke the first applicable deterministic tool before calling the LLM."""
    for tool in ANSWER_TOOLS:
        result = tool.handler(question, contexts)
        if result:
            print(f"  Tool invoked: {tool.name}")
            return result
    return None


def _money_to_vnd(raw: str, unit: str) -> int | None:
    normalized = raw.replace(" ", "")
    if unit.lower() in {"triệu", "trieu"}:
        try:
            return round(float(normalized.replace(",", ".")) * 1_000_000)
        except ValueError:
            return None
    digits = re.sub(r"\D", "", normalized)
    return int(digits) if digits else None


def _format_vnd(value: int) -> str:
    return f"{value:,}".replace(",", ".") + " VNĐ"
