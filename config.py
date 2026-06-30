"""Shared configuration for Lab 24: Eval + Guardrail Stack."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Optional: for HuggingFace models

# --- Day 18 compatibility ---
LLM_API_KEY = os.getenv("LLM_API_KEY", OPENAI_API_KEY)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
ACTIVE_LLM_PROVIDER = os.getenv("ACTIVE_LLM_PROVIDER", "openai")
FALLBACK_LLM_API_KEY = os.getenv("FALLBACK_LLM_API_KEY", OPENAI_API_KEY)
FALLBACK_LLM_BASE_URL = os.getenv("FALLBACK_LLM_BASE_URL", "")
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", "gpt-4o-mini")
NVIDIA_RATE_LIMIT_COOLDOWN_SECONDS = int(os.getenv("NVIDIA_RATE_LIMIT_COOLDOWN_SECONDS", "60"))
ANSWER_TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0"))
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "300"))
ENABLE_LLM_ENRICHMENT = os.getenv("ENABLE_LLM_ENRICHMENT", "0").lower() in {"1", "true", "yes"}
ENABLE_RAGAS = os.getenv("ENABLE_RAGAS", "0").lower() in {"1", "true", "yes"}

# --- Qdrant (same as Day 18) ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab24_production"

# --- Embedding (same as Day 18) ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking (same as Day 18) ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85
OCR_ENABLED = os.getenv("OCR_ENABLED", "0").lower() in {"1", "true", "yes"}
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
OCR_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".ocr_cache")
OCR_TESSDATA_DIR = os.getenv("OCR_TESSDATA_DIR", "")

# --- Search (same as Day 18) ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set_50q.json")
ANSWERS_PATH = os.path.join(os.path.dirname(__file__), "answers_50q.json")
HUMAN_LABELS_PATH = os.path.join(os.path.dirname(__file__), "human_labels_10q.json")
ADVERSARIAL_SET_PATH = os.path.join(os.path.dirname(__file__), "adversarial_set_20.json")
GUARDRAILS_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "guardrails")

# --- LLM Judge ---
JUDGE_MODEL = "gpt-4o-mini"

# --- RAGAS ---
RAGAS_LLM_API_KEY = os.getenv("RAGAS_LLM_API_KEY", OPENAI_API_KEY)
RAGAS_LLM_BASE_URL = os.getenv("RAGAS_LLM_BASE_URL", "")
RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "gpt-4o-mini")
RAGAS_EMBEDDING_PROVIDER = os.getenv("RAGAS_EMBEDDING_PROVIDER", "openai")
RAGAS_EMBEDDING_API_KEY = os.getenv("RAGAS_EMBEDDING_API_KEY", OPENAI_API_KEY)
RAGAS_EMBEDDING_BASE_URL = os.getenv("RAGAS_EMBEDDING_BASE_URL", "")
RAGAS_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")

# --- Guardrail latency budget ---
LATENCY_BUDGET_P95_MS = 500  # target: full guard stack P95 < 500ms
PRESIDIO_LANGUAGE = "en"    # Presidio base language; custom VN recognizers added via PatternRecognizer
