from __future__ import annotations

"""Production RAG Pipeline: chunk, enrich, retrieve, rerank, answer, evaluate."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch, reciprocal_rank_fusion
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from src.answering import generate_grounded_answer
from src.tools import invoke_answer_tools, plan_retrieval_queries
from config import ENABLE_LLM_ENRICHMENT, ENABLE_RAGAS, RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    parent_chunks: dict[str, str] = {}
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parent_chunks.update({parent.metadata["parent_id"]: parent.text for parent in parents})
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    native_chunks = [chunk for chunk in all_chunks if chunk["metadata"].get("extraction") != "ocr"]
    ocr_chunks = [chunk for chunk in all_chunks if chunk["metadata"].get("extraction") == "ocr"]
    enriched = enrich_chunks(native_chunks, use_llm=ENABLE_LLM_ENRICHMENT)
    # OCR text is indexed, but avoids hundreds of costly enrichment calls for
    # long scans; deterministic contextual metadata is sufficient for recall.
    enriched.extend(enrich_chunks(ocr_chunks, methods=["contextual", "metadata"], use_llm=False))
    all_chunks = [{"text": item.enriched_text, "metadata": item.auto_metadata} for item in enriched]
    print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    # Children are indexed for precision; full parents are returned to the LLM.
    search.parent_chunks = parent_chunks
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    query_plan = plan_retrieval_queries(query)
    result_lists = [search.search(planned_query) for planned_query in query_plan]
    results = reciprocal_rank_fusion(result_lists) if len(result_lists) > 1 else result_lists[0]
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    selected = _select_coverage_results(reranked, result_lists)
    parent_lookup = getattr(search, "parent_chunks", {})
    contexts = []
    for result in selected:
        parent_id = result.metadata.get("parent_id")
        context = parent_lookup.get(parent_id, result.text)
        if context not in contexts:
            contexts.append(context)
    tool_answer = invoke_answer_tools(query, contexts)
    if tool_answer:
        return tool_answer, contexts
    answer = generate_grounded_answer(query, contexts) if contexts else None
    answer = answer or (contexts[0] if contexts else "Không tìm thấy thông tin.")
    return answer, contexts


def _select_coverage_results(reranked, result_lists):
    """Keep one result per planned sub-query before filling with reranked hits."""
    selected = []
    for result_list in result_lists:
        if result_list:
            selected.append(result_list[0])
    selected.extend(reranked)
    unique = []
    seen = set()
    for result in selected:
        if result.text not in seen:
            unique.append(result)
            seen.add(result.text)
        if len(unique) == RERANK_TOP_K:
            break
    return unique


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths, enabled=ENABLE_RAGAS)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
