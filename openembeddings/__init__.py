"""OpenEmbeddings: Advanced Embedding and Retrieval Research Framework

A comprehensive, production-ready framework for embedding and retrieval research,
designed for both academic research and real-world applications.

Key Features:
- Production-ready embeddings with multiple backends
- Scalable search with FAISS integration
- Advanced fusion strategies (Linear, RRF, Learned)
- Cross-encoder re-ranking
- Comprehensive benchmarking with BEIR/MTEB
- Hyperparameter optimization
- Performance profiling and analysis
- Multi-language support
- Intelligent caching and persistence

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from .models.dense_embedder import DenseEmbedder
from .models.sparse_embedder import SparseEmbedder
from .models.hybrid_retriever import HybridRetriever
from .models.reranker import ReRanker

# Import advanced modules
try:
    from .benchmarks import BenchmarkSuite, DatasetManager
    from .experiments import ExperimentRunner, ExperimentConfig, AutoMLExperiment
except ImportError:
    # Optional dependencies not available
    BenchmarkSuite = None
    DatasetManager = None
    ExperimentRunner = None
    ExperimentConfig = None
    AutoMLExperiment = None

# Version information
__version__ = "0.2.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"

# Main exports
__all__ = [
    # Core models
    "DenseEmbedder",
    "SparseEmbedder", 
    "HybridRetriever",
    "ReRanker",
    
    # Advanced features (when available)
    "BenchmarkSuite",
    "DatasetManager",
    "ExperimentRunner",
    "ExperimentConfig",
    "AutoMLExperiment",
    
    # Utility functions
    "create_retrieval_pipeline",
    "benchmark_models",
    "optimize_hyperparameters",
]

# Convenience functions
def create_retrieval_pipeline(
    documents=None,
    dense_model="all-MiniLM-L6-v2",
    fusion_strategy="rrf",
    use_reranker=False,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    **kwargs
):
    """Create a complete retrieval pipeline with sensible defaults.
    
    Args:
        documents: List of documents to index
        dense_model: Dense embedding model name
        fusion_strategy: Fusion strategy ('linear', 'rrf')
        use_reranker: Whether to include re-ranking
        reranker_model: Re-ranker model name
        **kwargs: Additional arguments for components
        
    Returns:
        Configured retrieval pipeline
    """
    # Create hybrid retriever
    retriever = HybridRetriever(
        dense_model=dense_model,
        fusion_strategy=fusion_strategy,
        **kwargs
    )
    
    # Index documents if provided
    if documents:
        retriever.index(documents, show_progress=True)
    
    # Create re-ranker if requested
    reranker = None
    if use_reranker:
        reranker = ReRanker(model_name=reranker_model)
    
    return retriever, reranker

def benchmark_models(
    models,
    datasets=None,
    output_dir="benchmark_results",
    **kwargs
):
    """Benchmark multiple models on standard datasets.
    
    Args:
        models: List of model configurations or names
        datasets: List of dataset names (default: common BEIR datasets)
        output_dir: Output directory for results
        **kwargs: Additional benchmarking options
        
    Returns:
        Benchmark results dictionary
    """
    if BenchmarkSuite is None:
        raise ImportError("Benchmarking requires optional dependencies. Install with: pip install openembeddings[research]")
    
    if datasets is None:
        datasets = ["scifact", "nfcorpus", "arguana"]
    
    benchmark_suite = BenchmarkSuite(output_dir)
    results = {}
    
    for model_config in models:
        if isinstance(model_config, str):
            model_config = {"dense_model": model_config}
            
        retriever = HybridRetriever(**model_config)
        model_results = benchmark_suite.run_comprehensive_benchmark(
            retriever=retriever,
            beir_datasets=datasets,
            **kwargs
        )
        results[model_config.get("name", str(model_config))] = model_results
    
    return results

def optimize_hyperparameters(
    dataset,
    parameter_space=None,
    n_trials=50,
    output_dir="optimization_results",
    **kwargs
):
    """Optimize hyperparameters for a retrieval system.
    
    Args:
        dataset: Dataset for optimization
        parameter_space: Parameter space to search
        n_trials: Number of optimization trials
        output_dir: Output directory for results
        **kwargs: Additional optimization options
        
    Returns:
        Optimization results
    """
    if ExperimentRunner is None:
        raise ImportError("Optimization requires optional dependencies. Install with: pip install openembeddings[research]")
    
    if parameter_space is None:
        parameter_space = {
            "dense_weight": [0.3, 0.5, 0.7, 0.9],
            "fusion_strategy": ["linear", "rrf"],
            "use_ann": [True, False]
        }
    
    from .experiments import ExperimentConfig
    
    config = ExperimentConfig(
        name="hyperparameter_optimization",
        description="Automated hyperparameter optimization",
        model_configs=[],
        datasets=[dataset],
        metrics=["accuracy", "f1_score"],
        hyperparameters=parameter_space,
        output_dir=output_dir
    )
    
    runner = ExperimentRunner(config)
    results = runner.run_hyperparameter_optimization(
        model_class=HybridRetriever,
        dataset=dataset,
        n_trials=n_trials,
        **kwargs
    )
    
    return results

import sys

# Expose faiss at top-level package if available so static checks (e.g., tests) can
# detect that the optional dependency has been installed without importing the
# whole sub-module implementation. This avoids unnecessary test skips based on
# `hasattr(__import__("openembeddings"), "faiss")` style checks.
if "faiss" in sys.modules:
    import faiss as _faiss  # type: ignore
    # Attach as attribute for external introspection
    faiss = _faiss  # noqa: F401
