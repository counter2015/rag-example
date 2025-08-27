"""Gemini embedding client wrapper for better testability."""

import os
from typing import List, Optional
from google import genai
from google.genai import types

from embedding import normalize_embedding


class GeminiEmbeddingClient:
    """Wrapper for Gemini embedding API with better error handling and testability."""

    def __init__(self, api_key: Optional[str] = None, dimension: Optional[int] = None):
        """Initialize the Gemini client

        Args:
            api_key: API key for Gemini. If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter."
            )

        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-embedding-001"
        self.dimension = dimension or 3072

    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Get embedding for a single text.

        Args:
            text: Input text to embed
            model: Model name (default: gemini-embedding-001)

        Returns:
            List of embedding values

        Raises:
            ValueError: If text is empty
            Exception: If API call fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        model_name = model or self.model

        try:
            result = self.client.models.embed_content(
                model=model_name,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=self.dimension),
            )

            if not result.embeddings or len(result.embeddings) == 0:
                raise Exception("No embeddings returned from API")

            raw_embedding = result.embeddings[0].values
            normalized_embedding = normalize_embedding(raw_embedding)
            return normalized_embedding

        except Exception as e:
            raise Exception(f"Failed to get embedding: {str(e)}")

    def get_embeddings_batch(
        self, texts: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        """Get embeddings for multiple texts.

        Args:
            texts: List of input texts to embed
            model: Model name (default: gemini-embedding-001)

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If texts list is empty
            Exception: If API call fails
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text, model)
            embeddings.append(embedding)

        return embeddings

    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get the dimension of embeddings for the model.

        Args:
            model: Model name (default: gemini-embedding-001)

        Returns:
            Embedding dimension
        """
        # Get a sample embedding to determine dimension
        sample_embedding = self.get_embedding("sample", model)
        return len(sample_embedding)
