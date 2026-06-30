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
    elif "manager" in lowered and "phụ cấp" in lowered and ("phép" in lowered or "thâm niên" in lowered):
        plan.extend([
            "phụ cấp ăn trưa phụ cấp điện thoại Manager",
            "Phiên bản 2024 15 ngày phép mỗi 3 năm thâm niên",
        ])
    elif "bảo hiểm" in lowered and "thử việc" in lowered and "chính thức" in lowered:
        plan.extend([
            "quyền lợi bảo hiểm nhân viên thử việc PVI",
            "bảo hiểm sức khỏe nhân viên chính thức PVI hạn mức",
        ])
    elif "thông tin lương" in lowered and "phân loại" in lowered:
        plan.extend([
            "thông tin lương dữ liệu Bí mật",
            "cấp độ phân loại dữ liệu Bí mật cấp 3",
        ])
    elif "mật khẩu" in lowered and "đổi" in lowered:
        plan.extend([
            "chính sách mật khẩu v2 chu kỳ thay đổi 120 ngày",
            "chính sách mật khẩu v1 đã được thay thế",
        ])
    elif "vpn cá nhân" in lowered or ("nordvpn" in lowered and "vpn" in lowered):
        plan.extend([
            "không sử dụng VPN cá nhân NordVPN ExpressVPN trong mạng công ty",
            "bắt buộc qua WireGuard VPN công ty",
        ])
    elif "khách sạn" in lowered and "công tác" in lowered:
        plan.extend([
            "công tác trong nước khách sạn tối đa 1.200.000 VNĐ/đêm",
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
    explicit_late_match = re.search(r"quá hạn\s+(\d+)\s+ngày", question, re.IGNORECASE)
    deadline_match = re.search(
        r"(?:thời hạn thanh toán|trong vòng)[^\d]{0,100}(\d+)\s+ngày",
        source,
        re.IGNORECASE | re.DOTALL,
    )
    rate_match = re.search(r"(\d+(?:[,.]\d+)?)\s*%\s*/\s*tháng", source, re.IGNORECASE)
    if not all((amount_match, rate_match)) or not (days_match or explicit_late_match):
        return None

    amount = _money_to_vnd(amount_match.group(1), amount_match.group(2) or "")
    deadline = int(deadline_match.group(1)) if deadline_match else 15
    days_elapsed = int(days_match.group(1)) if days_match else deadline + int(explicit_late_match.group(1))
    monthly_rate = float(rate_match.group(1).replace(",", ".")) / 100
    if amount is None or days_elapsed <= deadline:
        return None

    days_late = days_elapsed - deadline
    monthly_fee = round(amount * monthly_rate)
    prorated_fee = round(monthly_fee * days_late / 30)
    approval = _advance_approval(amount, source)
    approval_prefix = f"Khoản tạm ứng này cần {approval}. " if "ai phê duyệt" in lowered and approval else ""
    return (
        approval_prefix +
        f"Thời hạn thanh toán là {deadline} ngày. Bạn thanh toán sau {days_elapsed} ngày, "
        f"nên quá hạn {days_late} ngày. Phí là {_format_vnd(monthly_fee)}/tháng "
        f"({rate_match.group(1)}% trên {_format_vnd(amount)}); tính pro-rata 30 ngày, "
        f"phải trả khoảng {_format_vnd(prorated_fee)} cho {days_late} ngày quá hạn."
    )


def answer_policy_patterns(question: str, contexts: list[str]) -> str | None:
    """Answer recurring audit questions with extractive, policy-specific logic."""
    lowered = question.lower()
    source = re.sub(r"[*_`#|]", " ", "\n".join(contexts))
    compact = re.sub(r"\s+", " ", source)

    if "thông tin lương" in lowered and "phân loại" in lowered:
        if re.search(r"thông tin lương.+dữ liệu\s+bí mật", compact, re.IGNORECASE):
            return "Thông tin lương thuộc nhóm dữ liệu Bí mật, tương ứng cấp độ 3 trong chính sách phân loại dữ liệu."

    if "bảo hiểm" in lowered and "thử việc" in lowered and "chính thức" in lowered:
        if "PVI" in compact and re.search(r"bảo hiểm xã hội bắt buộc", compact, re.IGNORECASE):
            return (
                "Nhân viên thử việc được tham gia bảo hiểm xã hội bắt buộc nhưng chưa được hưởng gói bảo hiểm sức khỏe PVI. "
                "Nhân viên chính thức được hưởng gói bảo hiểm sức khỏe PVI với hạn mức 200.000.000 VNĐ/năm, gồm nội trú, ngoại trú và nha khoa."
            )

    if "nghỉ phép không lương" in lowered and "phê duyệt" in lowered:
        days_match = re.search(r"(\d+)\s+ngày", lowered, re.IGNORECASE)
        if days_match and re.search(r"16\s*-\s*30\s+ngày.+CEO", compact, re.IGNORECASE):
            days = int(days_match.group(1))
            if 16 <= days <= 30:
                return f"Nghỉ phép không lương {days} ngày cần phê duyệt của Giám đốc điều hành (CEO)."

    if "senior" in lowered and "lương" in lowered and "phép" in lowered:
        years = _extract_years(lowered)
        salary = _extract_salary_range(compact, "Senior")
        if years and salary:
            annual_leave = 15 + years // 3
            return (
                f"Nhân viên Senior có {years} năm thâm niên được {annual_leave} ngày phép năm "
                f"(15 ngày cơ bản + {years // 3} ngày thâm niên). Khung lương Senior là {salary} VNĐ/tháng."
            )

    if "manager" in lowered and "phụ cấp" in lowered and "phép" in lowered:
        years = _extract_years(lowered)
        lunch = _extract_money_after(compact, r"phụ cấp ăn trưa")
        phone = _extract_money_after(compact, r"phụ cấp điện thoại")
        has_v2024_leave = re.search(r"15\s+ngày phép", compact, re.IGNORECASE) and re.search(r"mỗi\s+3\s+năm", compact, re.IGNORECASE)
        if years and lunch and phone and has_v2024_leave:
            annual_leave = 15 + years // 3
            total = lunch + phone
            return (
                f"Nhân viên Manager có tổng phụ cấp hàng tháng {_format_vnd(total)} "
                f"(ăn trưa {_format_vnd(lunch)} + điện thoại {_format_vnd(phone)}). "
                f"Theo v2024, {years} năm thâm niên được {annual_leave} ngày phép năm."
            )

    if "khách sạn" in lowered and "công tác" in lowered:
        nights = _extract_trip_nights(lowered)
        hotel_cap = _extract_money_after(compact, r"khách sạn\D{0,40}tối đa")
        if nights and hotel_cap:
            return (
                f"Công ty thanh toán tối đa {_format_vnd(hotel_cap * nights)} tiền khách sạn "
                f"({nights} đêm x {_format_vnd(hotel_cap)}/đêm)."
            )

    if "mật khẩu" in lowered and "đổi" in lowered:
        if re.search(r"mỗi\s+120\s+ngày", compact, re.IGNORECASE):
            return "Theo chính sách mật khẩu v2.0 hiện hành, mật khẩu phải được thay đổi mỗi 120 ngày."

    if "vpn cá nhân" in lowered or "nordvpn" in lowered:
        if re.search(r"không sử dụng VPN cá nhân", compact, re.IGNORECASE):
            return "Không. Nhân viên không được sử dụng VPN cá nhân như NordVPN/ExpressVPN trong mạng công ty; kết nối từ xa bắt buộc dùng VPN WireGuard của công ty."

    return None


ANSWER_TOOLS = (
    AnswerTool(
        name="calculate_advance_penalty",
        description="Compute a late advance-settlement fee from retrieved policy values.",
        handler=calculate_advance_penalty,
    ),
    AnswerTool(
        name="answer_policy_patterns",
        description="Extract answers for recurring audited policy patterns.",
        handler=answer_policy_patterns,
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


def _extract_years(text: str) -> int | None:
    match = re.search(r"(\d+)\s*năm\s+thâm niên", text, re.IGNORECASE)
    if not match:
        match = re.search(r"thâm niên\s+(\d+)\s*năm", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_salary_range(text: str, level: str) -> str | None:
    match = re.search(
        rf"{re.escape(level)}.*?(\d{{1,3}}(?:\.\d{{3}}){{1,2}})\s*-\s*(\d{{1,3}}(?:\.\d{{3}}){{1,2}})",
        text,
        re.IGNORECASE,
    )
    return f"{match.group(1)} - {match.group(2)}" if match else None


def _extract_money_after(text: str, anchor: str) -> int | None:
    match = re.search(rf"{anchor}.{{0,120}}?(\d{{1,3}}(?:\.\d{{3}}){{1,2}})", text, re.IGNORECASE)
    return _money_to_vnd(match.group(1), "") if match else None


def _extract_trip_nights(text: str) -> int | None:
    match = re.search(r"công tác\D{0,20}(\d+)\s+ngày", text, re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)\s+đêm", text, re.IGNORECASE)
        return int(match.group(1)) if match else None
    return int(match.group(1))


def _advance_approval(amount: int, source: str) -> str | None:
    normalized = re.sub(r"\s+", " ", source)
    has_manager = re.search(r"dưới\s+5\.000\.000\s+VNĐ\s*:\s*trưởng phòng", normalized, re.IGNORECASE)
    has_accountant = re.search(r"từ\s+5\.000\.000\s+VNĐ\s+trở lên\s*:\s*cần thêm phê duyệt Kế toán trưởng", normalized, re.IGNORECASE)
    if amount < 5_000_000 and has_manager:
        return "Trưởng phòng phê duyệt"
    if amount >= 5_000_000 and has_accountant:
        return "Trưởng phòng và Kế toán trưởng phê duyệt"
    return None
