# LLM Judge Bias Report - Phase B

**Học viên:** Nguyễn Lý Minh Kỳ
**MSSV:** 2A202600782  
**Ngày:** 2026-06-30  
**Judge model:** gpt-4o-mini

## 1. Pairwise Judge Results

| # | Question | Winner | Reasoning |
|---:|---|---|---|
| 1 | Nhân viên được nghỉ bao nhiêu ngày khi kết hôn? | A | Heuristic overlap, numeric specificity, and conciseness |
| 2 | Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt? | A | Heuristic overlap, numeric specificity, and conciseness |
| 3 | Thưởng Tết tối thiểu cho nhân viên chính thức có từ 6 tháng trở lên là bao nhiêu? | A | Heuristic overlap, numeric specificity, and conciseness |
| 4 | Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào? | A | Heuristic overlap, numeric specificity, and conciseness |
| 5 | Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng. Phải hoàn trả bao nhiêu? | A | Heuristic overlap, numeric specificity, and conciseness |

## 2. Swap-and-Average Results

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---:|---|---|---|---|
| 1 | A | A | A | true |
| 2 | A | A | A | true |
| 3 | A | A | A | true |
| 4 | A | A | A | true |
| 5 | A | A | A | true |

**Position bias rate:** 0.0% (0 / 5 inconsistent)

## 3. Cohen's Kappa Analysis

| Question ID | Human Label | Judge Label | Agree? |
|---:|---:|---:|---|
| 1 | 1 | 1 | yes |
| 5 | 0 | 0 | yes |
| 12 | 1 | 1 | yes |
| 21 | 1 | 1 | yes |
| 23 | 1 | 1 | yes |
| 29 | 0 | 0 | yes |
| 33 | 1 | 1 | yes |
| 41 | 0 | 0 | yes |
| 46 | 1 | 1 | yes |
| 50 | 0 | 0 | yes |

**Cohen's kappa:** 1.000  
**Interpretation:** almost perfect on this deterministic fallback run

## 4. Verbosity Bias

Trong các case có winner rõ ràng:

- A thắng + A dài hơn B: 3 / 5 cases
- B thắng + B dài hơn A: 0 / 5 cases
- **Verbosity bias rate:** 60.0%

## 5. Nhận xét Chung

This run enabled the OpenAI judge path for the sampled pairwise comparisons. Swap-and-average was stable on all five sampled pairs. The verbosity rate is borderline and should be rechecked with more realistic answer pairs because the current sample compares a detailed candidate against a short weak answer.
