# Failure Cluster Analysis - Phase A

**Sinh viên:** Nguyễn Lý Minh Kỳ 
**MSSV:** 2A202600782
**Ngày:** 2026-06-30

## 1. Aggregate RAGAS Scores Theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.958 | 0.590 | 0.900 |
| answer_relevancy | 0.530 | 0.492 | 0.604 |
| context_precision | 0.962 | 1.000 | 0.983 |
| context_recall | 0.950 | 0.921 | 0.683 |
| **avg_score** | **0.850** | **0.751** | **0.793** |

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---:|---|---|---:|---|
| 1 | multi_hop | Nhân viên đi công tác trong nước 2 ngày, ở khách sạn giá 1.500.000 VNĐ/đêm. Công ty thanh toán tối đa bao nhiêu cho tiền khách sạn? | 0.541 | faithfulness |
| 2 | multi_hop | Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu? | 0.620 | faithfulness |
| 3 | multi_hop | Nhân viên đi công tác nước ngoài 4 ngày tại thành phố lớn, ở khách sạn 200 USD/đêm (3 đêm). Công ty thanh toán bao nhiêu tiền khách sạn? | 0.628 | faithfulness |
| 4 | multi_hop | Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu? | 0.634 | faithfulness |
| 5 | multi_hop | Nhân viên tạm ứng 8 triệu, chưa thanh toán sau 30 ngày (quá hạn 15 ngày). Ai phê duyệt khoản này và phí phạt là bao nhiêu? | 0.645 | faithfulness |
| 6 | adversarial | Nhân viên thử việc có được nghỉ phép năm không? | 0.667 | faithfulness |
| 7 | adversarial | Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không? | 0.667 | answer_relevancy |
| 8 | multi_hop | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp hàng tháng và số ngày phép năm theo v2024 là bao nhiêu? | 0.684 | faithfulness |
| 9 | factual | Cơ cấu điểm đánh giá hiệu suất gồm những thành phần nào và tỷ lệ ra sao? | 0.704 | answer_relevancy |
| 10 | multi_hop | Nhân viên tạm ứng 4 triệu và một nhân viên khác tạm ứng 7 triệu: quy trình phê duyệt khác nhau thế nào? | 0.714 | answer_relevancy |

## 3. Failure Cluster Matrix

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 0 | 9 | 1 | 10 |
| answer_relevancy | 19 | 11 | 6 | 36 |
| context_precision | 0 | 0 | 0 | 0 |
| context_recall | 1 | 0 | 3 | 4 |

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** answer_relevancy

The latest full RAGAS run shows retrieval is strong (`context_precision` around 0.96-1.00) and improved after the bottom-10 answer audit. `answer_relevancy` still dominates the worst-metric matrix, especially on factual questions, which suggests generated answers can still include wording that RAGAS does not map cleanly back to the original question. Multi-hop remains the weakest distribution, mainly from faithfulness drops on calculation/comparison questions.

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating or unsupported claims | Tighten grounded prompt and require citation-backed claims |
| context_recall | Missing relevant chunks | Improve chunking, hybrid search, and query expansion |
| context_precision | Too many irrelevant chunks | Use reranking and metadata/version filters |
| answer_relevancy | Answer does not match user intent | Improve prompt template and answer-shape classification |

## 6. Adversarial Distribution Notes

Adversarial avg_score is 0.793, lower than factual 0.850 but higher than multi-hop 0.751. The previous password rotation and personal VPN traps improved after version-aware/policy-specific answer rules. Context recall is still weakest in adversarial at 0.683, so version-aware metadata filtering remains a priority.
