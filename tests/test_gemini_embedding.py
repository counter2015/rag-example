import unittest

from embedding.gemini import GeminiEmbeddingClient
from misc.proxy import setup_proxy


class TestGeminiEmbedding(unittest.TestCase):
    setup_proxy()

    def test_get_embedding(self):
        client = GeminiEmbeddingClient()
        response = client.get_embedding("Hello, world!")
        self.assertIsNotNone(response)
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 3072)

    def test_get_embeddings_batch(self):
        client = GeminiEmbeddingClient()
        response = client.get_embeddings_batch(["Hello, world!", "Hello, world!"])
        self.assertIsNotNone(response)
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 2)
        self.assertEqual(len(response[0]), 3072)
        self.assertEqual(len(response[1]), 3072)

    def test_get_embedding_dimension(self):
        client = GeminiEmbeddingClient(dimension=768)
        response = client.get_embedding("Hello, world!")
        self.assertIsNotNone(response)
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 768)
