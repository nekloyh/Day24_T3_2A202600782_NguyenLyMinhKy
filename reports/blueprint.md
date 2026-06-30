# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** N/A  
**Ngày:** 2026-06-30

## Guard Stack Architecture

```
User Input
    |
    v
[PII Scan]
    | block if: VN_CCCD / VN_PHONE / EMAIL detected
    | action: reject + log
    v
[Input Rail]
    | block if: off-topic / jailbreak / prompt injection / PII request
    | action: refuse with reason
    v
[RAG Pipeline (Day 18)]
    | M1 Chunk -> M2 Search -> M3 Rerank -> Answer
    v
[Output Rail]
    | flag if: PII in response / sensitive content
    | action: redact or replace with safe response
    v
User Response
```

## Guard Stack Pipeline

| Layer | Tool | Latency P95 | Failure Action |
|---|---|---:|---|
| PII Detection | Presidio-compatible regex fallback | 0.01ms | Reject + log |
| Topic/Jailbreak | Input rail rules, NeMo-compatible interface | 0.01ms | Block + reason |
| RAG Pipeline | Day 18 modules | Not measured in this run | Fallback answer |
| Output Check | Output rail rules, NeMo-compatible interface | Not measured separately | Redact/block + log |

## CI Gates

- [x] RAGAS report generated on 50q test set
- [x] Adversarial suite pass rate >= 90%: 20/20
- [x] P95 total guard latency < 500ms: 0.02ms
- [x] Unit tests pass: 40/40 phase tests

## Monitoring

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness | < 0.75 | Inspect bottom-10 and retrieval evidence |
| Adversarial pass rate | < 18/20 | Add new rail patterns and regression tests |
| Guard P95 latency | > 500ms | Profile NeMo/API calls and add caching |
| PII detected count | Spike >10/hour | Security review |

## Lab Results

| Metric | Result |
|---|---:|
| RAGAS avg_score (50q) | 0.799 |
| Worst metric | answer_relevancy |
| Dominant failure distribution | factual |
| Cohen's kappa | 1.000 |
| Adversarial pass rate | 20 / 20 |
| Guard P95 latency | 0.02ms |

## Notes

This run used the Day 18 Qdrant pipeline to regenerate `answers_50q.json` and enabled real RAGAS scoring for Phase A. Phase B was run with the OpenAI judge path enabled for pairwise samples; the kappa labels are still derived from the provided human-label fixture. Phase C uses the local deterministic guard implementation behind the NeMo-compatible interface.
