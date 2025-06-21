import pytest
import torch
import numpy as np
import tempfile
import pathlib
from unittest.mock import patch, MagicMock

from openembeddings.models.dense_embedder import DenseEmbedder
from openembeddings.models.hybrid_retriever import HybridRetriever
from openembeddings.models.sparse_embedder import SparseEmbedder
from openembeddings.models.reranker import ReRanker


class TestDenseEmbedder:
    @pytest.fixture
    def embedder(self):
        # Use hashing backend for fast, offline, deterministic tests
        return DenseEmbedder(model_name="hashing-encoder")

    def test_initialization(self, embedder):
        assert embedder is not None
        assert embedder.dimension == 384
        assert embedder.backend == "hash"

    def test_forward_pass(self, embedder):
        texts = ["Hello world", "This is a test"]
        embeddings, _ = embedder(texts)
        assert embeddings.shape == (2, 384)
        norms = torch.norm(embeddings, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_encode_batch(self, embedder):
        texts = [f"Text {i}" for i in range(100)]
        embeddings = embedder.encode(texts, batch_size=32)
        assert embeddings.shape == (100, 384)
        assert isinstance(embeddings, np.ndarray)

    def test_save_load_pretrained(self, embedder):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir)
            embedder.save_pretrained(str(path))
            loaded_embedder = DenseEmbedder.from_pretrained(str(path))
            assert loaded_embedder.backend == "hash"
            assert loaded_embedder.dimension == embedder.dimension

    @pytest.mark.parametrize("strategy", ["mean", "cls", "max"])
    def test_different_pooling_strategies(self, strategy):
        embedder = DenseEmbedder(pooling_strategy=strategy)
        embeddings, _ = embedder(["Test text"])
        assert embeddings.shape == (1, 384)


class TestSparseEmbedder:
    @pytest.fixture
    def embedder(self):
        # Mock NLTK to avoid download errors in CI
        with patch("openembeddings.models.sparse_embedder._NLTK_AVAILABLE", False):
            yield SparseEmbedder(model_type="bm25")

    def test_initialization(self, embedder):
        assert embedder is not None
        assert embedder.model_type == "bm25"

    def test_fit_transform(self, embedder):
        documents = [
            "The quick brown fox",
            "Jumps over the lazy dog",
            "Machine learning algorithms",
        ]
        embedder.fit(documents)
        scores = embedder.compute_scores(["quick fox"], documents)
        assert len(scores) == 1
        assert len(scores[0]) == 3
        assert scores[0][0] > scores[0][2]


class TestHybridRetriever:
    @pytest.fixture
    def documents(self):
        return [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "Python is a programming language",
            "Transformers have revolutionized NLP",
        ]

    @pytest.fixture
    def linear_retriever(self, documents):
        # Retriever for 'linear' fusion strategy (no ANN)
        retriever = HybridRetriever(
            dense_model="hashing-encoder",
            fusion_strategy="linear",
            dense_weight=0.5,
            sparse_weight=0.5,
            use_ann=False,
        )
        retriever.index(documents)
        return retriever

    @pytest.fixture
    def rrf_retriever(self, documents):
        # Retriever for 'rrf' fusion strategy (no ANN for deterministic tests)
        retriever = HybridRetriever(
            dense_model="hashing-encoder",
            fusion_strategy="rrf",
            use_ann=False,
        )
        retriever.index(documents)
        return retriever

    def test_initialization_and_indexing(self, linear_retriever, documents):
        assert linear_retriever is not None
        assert linear_retriever.is_indexed
        assert linear_retriever.fusion_strategy == "linear"
        assert len(linear_retriever.corpus) == len(documents)
        assert linear_retriever.doc_dense_embeddings is not None
        assert linear_retriever.doc_dense_embeddings.shape == (len(documents), 384)

    def test_retrieval_linear(self, linear_retriever):
        query = "What is machine learning?"
        results = linear_retriever.retrieve(query, top_k=2)
        assert len(results) == 2
        # Check scores are descending
        assert results[0][1] >= results[1][1]
        # Check that the top result is the most relevant one
        assert "machine learning" in results[0][2].lower()

    def test_retrieval_rrf(self, rrf_retriever):
        query = "What is machine learning?"
        results = rrf_retriever.retrieve(query, top_k=2)
        assert len(results) == 2
        assert results[0][1] >= results[1][1]
        # Check that we get some relevant results (relaxed assertion for hashing backend)
        assert any("machine" in doc.lower() or "learning" in doc.lower() for _, _, doc in results)

    def test_forward_pass(self, linear_retriever):
        queries = ["test query"]
        results = linear_retriever.forward(queries)
        assert "scores" in results
        assert "dense_scores" in results
        assert "sparse_scores" in results
        assert results["scores"].shape == (1, len(linear_retriever.corpus))

    def test_save_load_indexed_retriever(self, linear_retriever, documents):
        query = "What is python?"
        original_results = linear_retriever.retrieve(query, top_k=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(tmpdir)
            linear_retriever.save_pretrained(path)
            loaded_retriever = HybridRetriever.from_pretrained(path)

        assert loaded_retriever.is_indexed
        assert len(loaded_retriever.corpus) == len(documents)
        assert loaded_retriever.doc_dense_embeddings is not None

        loaded_results = loaded_retriever.retrieve(query, top_k=2)
        assert len(loaded_results) == 2
        assert original_results[0][2] == loaded_results[0][2]  # Same top doc
        assert np.isclose(original_results[0][1], loaded_results[0][1])  # Same score

    @pytest.mark.skipif(
        not hasattr(__import__("openembeddings.models.hybrid_retriever"), "faiss"),
        reason="faiss not installed",
    )
    def test_save_load_faiss_index(self, documents):
        retriever = HybridRetriever(
            dense_model="hashing-encoder", fusion_strategy="rrf", use_ann=True
        )
        retriever.index(documents)
        assert retriever.faiss_index is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(tmpdir)
            retriever.save_pretrained(path)
            loaded_retriever = HybridRetriever.from_pretrained(path)

        assert loaded_retriever.is_indexed
        assert loaded_retriever.faiss_index is not None
        assert loaded_retriever.faiss_index.ntotal == len(documents)


class TestReRanker:
    @patch("openembeddings.models.reranker.CrossEncoder")
    def test_reranking(self, MockCrossEncoder):
        # Mock the CrossEncoder predict method
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])
        MockCrossEncoder.return_value = mock_model

        reranker = ReRanker()
        query = "test query"
        documents = [
            (0, 0.8, "doc A"),
            (1, 0.7, "doc B"),
            (2, 0.6, "doc C"),
        ]
        reranked = reranker.rerank(query, documents)

        # Check that it was called with the right pairs
        # The enhanced ReRanker now includes batch_size parameter
        mock_model.predict.assert_called_once_with(
            [("test query", "doc A"), ("test query", "doc B"), ("test query", "doc C")],
            batch_size=32,
            show_progress_bar=False,
        )

        # Check the new order and scores
        assert len(reranked) == 3
        assert reranked[0][2] == "doc B"  # Highest new score 0.9
        assert reranked[0][1] == 0.9
        assert reranked[1][2] == "doc C"  # Second highest 0.5
        assert reranked[1][1] == 0.5


class TestIntegration:
    @pytest.fixture
    def rrf_retriever(self):
        documents = [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "Python is a programming language",
            "Transformers have revolutionized NLP",
        ]
        retriever = HybridRetriever(
            dense_model="hashing-encoder",
            fusion_strategy="rrf",
            use_ann=False,
        )
        retriever.index(documents)
        return retriever

    def test_end_to_end_retrieval(self, rrf_retriever):
        documents = [
            "Transformers revolutionized NLP",
            "BERT uses bidirectional training",
            "GPT is an autoregressive model",
            "Attention is all you need",
            "Word embeddings capture semantics",
        ]
        query = "What are transformer models?"

        # 1. Index
        rrf_retriever.index(documents)

        # 2. Retrieve
        results = rrf_retriever.retrieve(query, top_k=3)

        assert len(results) == 3
        # Check that we get some relevant results (relaxed for hashing backend)
        assert any("transformer" in doc.lower() or "bert" in doc.lower() or "gpt" in doc.lower() for _, _, doc in results)
        scores = [score for _, score, _ in results]
        assert scores == sorted(scores, reverse=True)
