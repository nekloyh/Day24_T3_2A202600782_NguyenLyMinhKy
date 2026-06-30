"""Grounded answer synthesis tuned for high RAGAS answer-relevancy.

RAGAS answer_relevancy scores how well the answer maps back to the question:
it reverse-generates questions from the answer and compares them to the real
one. Extra, unasked facts (exceptions, versions, obligations, sub-components)
broaden those reverse-questions and drag the score down — even when the facts
are correct and grounded. So the answer must address ONLY what was asked, as
tightly as possible, while still restating the question's subject so it stays
self-contained.
"""

from __future__ import annotations

from config import ANSWER_MAX_TOKENS
from src.llm import chat_completion, chat_json


SYSTEM_PROMPT = """Bạn là trợ lý chính sách nội bộ. Trả lời bằng tiếng Việt, chính xác và ĐÚNG TRỌNG TÂM.

Nguyên tắc cốt lõi: chỉ trả lời ĐÚNG ĐIỀU ĐƯỢC HỎI, ngắn gọn nhất có thể.
1. Lặp lại chủ thể của câu hỏi trong câu trả lời để câu trả lời tự chứa nghĩa.
2. KHÔNG thêm thông tin không được hỏi, kể cả khi evidence có: không nêu ngoại lệ, điều kiện
   áp dụng, thành phần chi tiết, phiên bản chính sách, nghĩa vụ liên quan, hệ quả phụ hay lời
   khuyên thêm — TRỪ KHI câu hỏi hỏi đúng những thứ đó.
   (VD: hỏi "tối thiểu bao nhiêu ký tự" → chỉ nêu số ký tự, không liệt kê yêu cầu hoa/thường/số;
    hỏi "phụ cấp bao nhiêu" → chỉ nêu số tiền, không nêu trường hợp không áp dụng.)
3. Chỉ dùng facts trong evidence, không suy diễn, không nói bạn đang dùng evidence.
   Không khẳng định "không có ngoại lệ"/"nếu đủ điều kiện" nếu evidence không nêu.
4. Nếu evidence thật sự không có, trả lời đúng: "Không tìm thấy thông tin."

Chọn dạng câu hỏi và viết câu trả lời tương ứng:
- single_fact (hỏi MỘT con số/ngưỡng/tên/loại, hoặc câu hỏi có–không): trả lời ĐÚNG MỘT CÂU,
  chỉ chứa giá trị được hỏi; câu có–không thì mở đầu bằng "Có"/"Không" rồi nêu fact cốt lõi.
- calculation (hỏi kết quả tính toán): viết ngắn `quy định -> số liệu -> kết luận`, kết thúc
  bằng con số cuối cùng.
- multi_part (câu hỏi có nhiều ý độc lập): mỗi ý đúng một câu, không thêm ý ngoài câu hỏi."""


def generate_grounded_answer(question: str, contexts: list[str]) -> str | None:
    """Generate an on-target, intent-aware answer; fall back to plain grounded text."""
    evidence = "\n\n--- Evidence ---\n\n".join(contexts)
    payload = chat_json(
        SYSTEM_PROMPT,
        f"""Câu hỏi: {question}

Evidence:
{evidence}

Trả về JSON duy nhất theo schema:
{{
  "intent": "single_fact | calculation | multi_part",
  "answer": "câu trả lời đúng trọng tâm, tự chứa nghĩa, KHÔNG thêm thông tin ngoài câu hỏi"
}}
Mỗi fact trong `answer` phải có thể tìm thấy hoặc diễn đạt lại trực tiếp từ evidence.
Dừng ngay khi đã trả lời đủ điều được hỏi; `answer` không nhắc đến evidence/schema.""",
        max_tokens=ANSWER_MAX_TOKENS,
    )
    if payload:
        answer = payload.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()

    return chat_completion(
        SYSTEM_PROMPT,
        f"Câu hỏi: {question}\n\nEvidence:\n{evidence}",
        max_tokens=ANSWER_MAX_TOKENS,
        temperature=0,
    )
