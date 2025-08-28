from typing import List, Optional
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    text: str = Field(..., description="embedding text", min_length=1, max_length=10000)
    model: Optional[str] = Field(
        default="gemini-embedding-001", description="embedding model name"
    )


class BatchEmbeddingRequest(BaseModel):
    texts: List[str] = Field(
        ..., description="embedding text list", min_length=1, max_length=100
    )
    model: Optional[str] = Field(
        default="gemini-embedding-001", description="embedding model name"
    )


class EmbeddingResponse(BaseModel):
    embedding: List[float] = Field(..., description="embedding vector")
    dimension: int = Field(..., description="embedding dimension")
    model: str = Field(..., description="embedding model name")
    text: str = Field(..., description="embedding text")


class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="embedding vector list")
    dimension: int = Field(..., description="embedding dimension")
    model: str = Field(..., description="embedding model name")
    texts: List[str] = Field(..., description="embedding text list")
    count: int = Field(..., description="embedding text count")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="error message")
    detail: Optional[str] = Field(None, description="error detail")
    code: Optional[str] = Field(None, description="error code")


class HealthResponse(BaseModel):
    status: str = Field(..., description="service status")
    version: str = Field(..., description="API version")
    model_available: bool = Field(..., description="model available")
    timestamp: str = Field(..., description="timestamp")
