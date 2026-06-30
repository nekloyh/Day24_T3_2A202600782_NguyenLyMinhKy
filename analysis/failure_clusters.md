# Failure Cluster Analysis - Phase A

**Sinh viên:** N/A  
**Ngày:** 2026-06-30

## 1. Aggregate RAGAS Scores Theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.896 | 0.523 | 0.767 |
| answer_relevancy | 0.498 | 0.419 | 0.446 |
| context_precision | 0.946 | 0.992 | 0.942 |
| context_recall | 0.925 | 0.821 | 0.683 |
| **avg_score** | **0.816** | **0.688** | **0.709** |

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---:|---|---|---:|---|
| 1 | factual | Thông tin lương thuộc cấp độ phân loại dữ liệu nào? | 0.333 | faithfulness |
| 2 | multi_hop | So sánh quyền lợi bảo hiểm giữa nhân viên thử việc và nhân viên chính thức. | 0.375 | faithfulness |
| 3 | multi_hop | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp hàng tháng và số ngày phép năm theo v2024 là bao nhiêu? | 0.375 | faithfulness |
| 4 | adversarial | Bao lâu phải đổi mật khẩu một lần? | 0.396 | faithfulness |
| 5 | adversarial | Nhân viên Manager có thể dùng VPN cá nhân khi WFH để tăng bảo mật thêm không? | 0.417 | faithfulness |
| 6 | multi_hop | Nhân viên tạm ứng 8 triệu, chưa thanh toán sau 30 ngày. Ai phê duyệt và phí phạt là bao nhiêu? | 0.500 | faithfulness |
| 7 | factual | Nghỉ phép không lương 20 ngày cần ai phê duyệt? | 0.527 | answer_relevancy |
| 8 | multi_hop | Nhân viên đi công tác trong nước 2 ngày, ở khách sạn giá 1.500.000 VNĐ/đêm. Công ty thanh toán tối đa bao nhiêu? | 0.540 | faithfulness |
| 9 | multi_hop | Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào? | 0.583 | answer_relevancy |
| 10 | multi_hop | Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu? | 0.586 | faithfulness |

## 3. Failure Cluster Matrix

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 1 | 9 | 2 | 12 |
| answer_relevancy | 18 | 11 | 5 | 34 |
| context_precision | 0 | 0 | 0 | 0 |
| context_recall | 1 | 0 | 3 | 4 |

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** answer_relevancy

The real RAGAS run shows retrieval is generally strong (`context_precision` around 0.94-0.99), but answer shaping is weaker. `answer_relevancy` dominates the worst-metric matrix, especially on factual questions, which suggests generated answers include wording or extra context that RAGAS does not map cleanly back to the original question. Multi-hop has the lowest average score, mainly from faithfulness drops on calculation/comparison questions.

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating or unsupported claims | Tighten grounded prompt and require citation-backed claims |
| context_recall | Missing relevant chunks | Improve chunking, hybrid search, and query expansion |
| context_precision | Too many irrelevant chunks | Use reranking and metadata/version filters |
| answer_relevancy | Answer does not match user intent | Improve prompt template and answer-shape classification |

## 6. Adversarial Distribution Notes

Adversarial avg_score is 0.709, lower than factual 0.816 but slightly higher than multi-hop 0.688. The adversarial bottom-10 cases include password rotation and personal VPN policy, which are version/policy-conflict traps. The pipeline handles many adversarial retrieval cases, but context recall is weakest in this distribution at 0.683, so version-aware metadata filtering remains a priority.
