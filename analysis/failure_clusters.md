# Failure Cluster Analysis - Phase A

**Sinh viên:** Nguyễn Lý Minh Kỳ 
**MSSV:** 2A202600782
**Ngày:** 2026-06-30

## 1. Aggregate RAGAS Scores Theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.983 | 0.681 | 0.967 |
| answer_relevancy | 0.515 | 0.488 | 0.665 |
| context_precision | 0.962 | 0.950 | 0.967 |
| context_recall | 0.975 | 0.925 | 0.683 |
| **avg_score** | **0.859** | **0.761** | **0.820** |

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---:|---|---|---:|---|
| 1 | multi_hop | Nhân viên đi công tác trong nước 2 ngày, ở khách sạn giá 1.500.000 VNĐ/đêm. Công ty thanh toán tối đa bao nhiêu cho tiền khách sạn? | 0.535 | faithfulness |
| 2 | multi_hop | Nhân viên đi công tác nước ngoài 4 ngày tại thành phố lớn, ở khách sạn 200 USD/đêm (3 đêm). Công ty thanh toán bao nhiêu tiền khách sạn? | 0.630 | faithfulness |
| 3 | multi_hop | So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0 về độ dài tối thiểu, thời hạn đổi và MFA. | 0.655 | context_precision |
| 4 | multi_hop | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp hàng tháng và số ngày phép năm theo v2024 là bao nhiêu? | 0.684 | faithfulness |
| 5 | multi_hop | Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu? | 0.685 | answer_relevancy |
| 6 | multi_hop | Nhân viên tạm ứng 8 triệu, chưa thanh toán sau 30 ngày (quá hạn 15 ngày). Ai phê duyệt khoản này và phí phạt là bao nhiêu? | 0.696 | answer_relevancy |
| 7 | multi_hop | Nhân viên tự ý xóa malware khỏi máy và chia sẻ thông tin sự cố này trên Slack nội bộ. Vi phạm những chính sách nào cụ thể? | 0.720 | faithfulness |
| 8 | multi_hop | Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu? | 0.731 | answer_relevancy |
| 9 | adversarial | Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không? | 0.735 | answer_relevancy |
| 10 | factual | Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt? | 0.735 | answer_relevancy |

## 3. Failure Cluster Matrix

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 0 | 6 | 0 | 6 |
| answer_relevancy | 19 | 12 | 6 | 37 |
| context_precision | 0 | 1 | 0 | 1 |
| context_recall | 1 | 1 | 4 | 6 |

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** answer_relevancy

The latest full RAGAS run shows retrieval remains strong and answer grounding improved after the second bottom-10 audit. Multi-hop faithfulness increased, but `answer_relevancy` still dominates the worst-metric matrix, especially on factual questions. Remaining issues are mostly answer-shape sensitivity and a few multi-hop calculation/comparison cases.

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating or unsupported claims | Tighten grounded prompt and require citation-backed claims |
| context_recall | Missing relevant chunks | Improve chunking, hybrid search, and query expansion |
| context_precision | Too many irrelevant chunks | Use reranking and metadata/version filters |
| answer_relevancy | Answer does not match user intent | Improve prompt template and answer-shape classification |

## 6. Adversarial Distribution Notes

Adversarial avg_score is 0.820, lower than factual 0.859 but higher than multi-hop 0.761. The previous password rotation, personal VPN, and trial-period traps improved after version-aware/policy-specific answer rules. Context recall is still weakest in adversarial at 0.683, so version-aware metadata filtering remains a priority.
