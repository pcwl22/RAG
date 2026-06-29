"""Search and answer routes."""

from fastapi import APIRouter

from app.api.enhanced_query import EnhancedQueryRequest, EnhancedQueryResponse
from app.api.enhanced_query import router as enhanced_query_router
from app.api.query import AnswerRequest, AnswerResponse, QueryRequest, QueryResponse
from app.api.query import router as query_router
from app.api.query_simple import router_simple

router = APIRouter()
router.include_router(query_router, tags=["query"])
router.include_router(router_simple, tags=["query-simple"])
router.include_router(enhanced_query_router, tags=["enhanced"])

__all__ = [
    "router",
    "QueryRequest",
    "QueryResponse",
    "AnswerRequest",
    "AnswerResponse",
    "EnhancedQueryRequest",
    "EnhancedQueryResponse",
]
