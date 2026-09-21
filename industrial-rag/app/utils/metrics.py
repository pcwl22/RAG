"""Low-cardinality Prometheus metrics for RAG pipeline service-level indicators."""

from prometheus_client import Counter, Histogram

LLM_REQUESTS = Counter(
    "rag_llm_requests_total",
    "LLM requests by provider, operation, and outcome.",
    ("provider", "operation", "outcome"),
)
LLM_DURATION = Histogram(
    "rag_llm_duration_seconds",
    "End-to-end LLM request duration.",
    ("provider", "operation"),
)
EMBEDDING_REQUESTS = Counter(
    "rag_embedding_requests_total",
    "Embedding batches by outcome.",
    ("outcome",),
)
EMBEDDING_DURATION = Histogram(
    "rag_embedding_duration_seconds",
    "Embedding batch duration.",
)
RETRIEVAL_REQUESTS = Counter(
    "rag_retrieval_requests_total",
    "Retrieval requests by outcome.",
    ("outcome",),
)
RETRIEVAL_DURATION = Histogram(
    "rag_retrieval_duration_seconds",
    "End-to-end hybrid retrieval duration.",
)
CACHE_LOOKUPS = Counter(
    "rag_cache_lookups_total",
    "Semantic answer-cache lookups by outcome.",
    ("outcome",),
)
