"""Tests for hybrid retriever module."""
import pytest
import numpy as np
from openembeddings.models.dense_embedder import DenseEmbedder
from openembeddings.models.sparse_embedder import SparseEmbedder
from openembeddings.models.hybrid_retriever import HybridRetriever

class MockDenseEmbedder:
    def retrieve(self, query, document_ids=None, top_k=10):
        return [("d1", 0.9), ("d2", 0.8), ("d3", 0.7)]

class MockSparseEmbedder:
    def retrieve(self, query, document_ids=None, top_k=10):
        return [("s1", 0.95), ("s2", 0.85), ("s3", 0.75)]

@pytest.fixture
def hybrid_retriever():
    dense = MockDenseEmbedder()
    sparse = MockSparseEmbedder()
    return HybridRetriever(dense, sparse, k=3)

def test_reciprocal_rank_fusion(hybrid_retriever):
    dense_results = [("d1", 0.9), ("d2", 0.8), ("d3", 0.7)]
    sparse_results = [("s1", 0.95), ("s2", 0.85), ("s3", 0.75)]
    
    rrf_results = hybrid_retriever.reciprocal_rank_fusion(dense_results, sparse_results)
    
    # Should return 3 results
    assert len(rrf_results) == 3
    
    # Should contain documents from both retrievers
    doc_ids = [doc_id for doc_id, _ in rrf_results]
    assert "d1" in doc_ids
    assert "s1" in doc_ids

def test_retrieve(hybrid_retriever):
    results = hybrid_retriever.retrieve("test query")
    
    # Should return 3 results
    assert len(results) == 3
    
    # Should have blended scores
    for _, score in results:
        assert 0 < score < 1

def test_parameter_weights():
    dense = MockDenseEmbedder()
    sparse = MockSparseEmbedder()
    hybrid = HybridRetriever(dense, sparse, k=3)
    
    # Test dense weight dominance
    results_dense = hybrid.retrieve("test", dense_weight=1.0, sparse_weight=0.0)
    doc_ids_dense = [doc_id for doc_id, _ in results_dense]
    assert "d1" in doc_ids_dense
    assert "s1" not in doc_ids_dense
    
    # Test sparse weight dominance
    results_sparse = hybrid.retrieve("test", dense_weight=0.0, sparse_weight=1.0)
    doc_ids_sparse = [doc_id for doc_id, _ in results_sparse]
    assert "d1" not in doc_ids_sparse
    assert "s1" in doc_ids_sparse

def test_rrf_constant_effect():
    dense = MockDenseEmbedder()
    sparse = MockSparseEmbedder()
    
    # With high RRF constant (less rank fusion)
    hybrid_high = HybridRetriever(dense, sparse, k=3, rrf_k=1000)
    results_high = hybrid_high.retrieve("test")
    
    # With low RRF constant (more rank fusion)
    hybrid_low = HybridRetriever(dense, sparse, k=3, rrf_k=10)
    results_low = hybrid_low.retrieve("test")
    
    # Should have different rankings
    assert results_high != results_low 