import os
import logging
from datetime import datetime
from typing import Dict
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from embedding.gemini import GeminiEmbeddingClient
from api.models import (
    EmbeddingRequest,
    BatchEmbeddingRequest,
    EmbeddingResponse,
    BatchEmbeddingResponse,
    ErrorResponse,
    HealthResponse,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.embedding_client = GeminiEmbeddingClient()
        app.state.start_time = datetime.now()
        logger.info("Gemini embedding client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize embedding client: {e}")
        app.state.embedding_client = None

    yield

    runtime = datetime.now() - app.state.start_time
    logger.info(f"Shutting down API server after {runtime}")


app = FastAPI(
    title="RAG Embedding API",
    description="基于Gemini的文本嵌入向量API服务",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_embedding_client(request: Request) -> GeminiEmbeddingClient:
    client = getattr(request.app.state, "embedding_client", None)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service is not available. Please check API key configuration.",
        )
    return client


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error", detail=str(exc)
        ).model_dump(),
    )


@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "RAG Embedding API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    client = getattr(request.app.state, "embedding_client", None)
    model_available = client is not None

    try:
        if model_available:
            test_embedding = client.get_embedding("test")
            model_available = len(test_embedding) > 0
    except Exception as e:
        logger.warning(f"Health check model test failed: {e}")
        model_available = False

    return HealthResponse(
        status="healthy" if model_available else "degraded",
        version="1.0.0",
        model_available=model_available,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/embedding", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    client: GeminiEmbeddingClient = Depends(get_embedding_client),
):
    """
    create single text embedding

    - **text**: embedding text
    - **model**: embedding model name (optional, default: gemini-embedding-001)
    """
    try:
        logger.info(f"Creating embedding for text: {request.text[:50]}...")

        embedding = client.get_embedding(request.text, request.model)

        logger.info(f"Successfully created embedding with dimension: {len(embedding)}")

        return EmbeddingResponse(
            embedding=embedding,
            dimension=len(embedding),
            model=request.model,
            text=request.text,
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create embedding: {str(e)}",
        )


@app.post("/embedding/batch", response_model=BatchEmbeddingResponse)
async def create_batch_embeddings(
    request: BatchEmbeddingRequest,
    client: GeminiEmbeddingClient = Depends(get_embedding_client),
):
    """
    create batch embeddings

    - **texts**: embedding text list
    - **model**: embedding model name (optional, default: gemini-embedding-001)
    """
    try:
        logger.info(f"Creating batch embeddings for {len(request.texts)} texts")

        if not request.texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text list cannot be empty",
            )

        if len(request.texts) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Too many texts. Maximum 100 texts per batch.",
            )

        embeddings = client.get_embeddings_batch(request.texts, request.model)

        dimension = len(embeddings[0]) if embeddings else 0

        logger.info(
            f"Successfully created {len(embeddings)} embeddings with dimension: {dimension}"
        )

        return BatchEmbeddingResponse(
            embeddings=embeddings,
            dimension=dimension,
            model=request.model,
            texts=request.texts,
            count=len(embeddings),
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating batch embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create batch embeddings: {str(e)}",
        )


@app.get("/models")
async def list_models():
    """list available embedding models"""
    return {
        "models": [
            {
                "id": "gemini-embedding-001",
                "name": "Gemini Embedding 001",
                "description": "Google Gemini embedding model",
                "max_input_length": 10000,
                "dimension": 768,
            }
        ]
    }


def main():
    """main function, start API server"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set!")
        logger.error("Please set your Gemini API key:")
        logger.error("  export GEMINI_API_KEY='your_api_key'")
        logger.error("  or add it to .env file")
        return

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    logger.info(f"Starting RAG Embedding API server on {host}:{port}")
    logger.info("API documentation will be available at http://localhost:8000/docs")

    uvicorn.run("api.server:app", host=host, port=port, reload=True, log_level="info")


if __name__ == "__main__":
    main()
