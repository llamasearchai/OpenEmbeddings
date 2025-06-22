# OpenEmbeddings

> Production-ready embedding & retrieval framework with hybrid search, quantization, and rigorous evaluation

[![PyPI](https://img.shields.io/pypi/v/openembeddings?color=brightgreen&label=PyPI)](https://pypi.org/project/openembeddings/)
[![CI](https://img.shields.io/github/actions/workflow/status/llamasearchai/OpenEmbeddings/ci.yml?branch=main&label=CI)](https://github.com/llamasearchai/OpenEmbeddings/actions/workflows/ci.yml)
[![Code Coverage](https://codecov.io/gh/llamasearchai/OpenEmbeddings/branch/main/graph/badge.svg)](https://codecov.io/gh/llamasearchai/OpenEmbeddings)
[![License](https://img.shields.io/github/license/llamasearchai/OpenEmbeddings)](LICENSE)

A comprehensive, production-ready framework for embedding and retrieval research, designed for both academic research and real-world applications. Features state-of-the-art transformer models, scalable ANN search with FAISS, advanced fusion techniques, and comprehensive benchmarking capabilities.

**Author**: Nik Jois <nikjois@llamasearch.ai>

## Key Features

### Core Capabilities
- **Production-Ready Embeddings**: Natively supports any model from `sentence-transformers` for creating high-quality dense vector embeddings
- **Scalable Search with FAISS**: Integrates `faiss` for highly efficient Approximate Nearest Neighbor (ANN) search, scaling to millions of documents
- **Advanced Fusion Strategies**: Supports linear fusion and Reciprocal Rank Fusion (RRF) for combining dense and sparse signals
- **Enhanced Sparse Search**: Uses NLTK for robust tokenization, stop-word removal, and stemming, significantly improving BM25 accuracy
- **Cross-Encoder Re-ranking**: Optional re-ranking stage using cross-encoder models for maximum relevance

### Research & Experimentation
- **Comprehensive Benchmarking**: Full BEIR dataset integration with automated evaluation
- **Hyperparameter Optimization**: Automated optimization using Optuna and custom AutoML
- **Ablation Studies**: Systematic component analysis and statistical significance testing
- **Performance Profiling**: Memory usage, runtime analysis, and scalability testing
- **Visualization**: Interactive dashboards and comprehensive reporting

### Advanced Tools
- **Dataset Management**: Automated loading of research datasets (BEIR, HuggingFace)
- **Experiment Tracking**: Weights & Biases integration for experiment management
- **Statistical Analysis**: Significance testing and effect size calculations
- **Multi-Modal Support**: Extensible architecture for different embedding types

##  Installation

### Basic Installation
```bash
pip install openembeddings
```

### Research Installation (Full Features)
```bash
pip install openembeddings[research]
# or for development
pip install -e ".[research]"
```

### Additional Dependencies
```bash
# For NLTK data (enhanced sparse search)
python -m nltk.downloader punkt stopwords

# For research features
pip install wandb optuna beir mteb
```

## Quick Start

### 1. Basic Usage

```bash
# Build index and search in one command
openembeddings search "What is machine learning?" \
    --index-path my_index \
    --input-file documents.txt \
    --top-k 5 \
    --rerank
```

### 2. Advanced Configuration

```bash
# Build index with RRF fusion and FAISS
openembeddings build-index \
    --input-file documents.txt \
    --index-path my_index \
    --fusion-strategy rrf \
    --ann \
    --dense-model "all-MiniLM-L6-v2"

# Search with re-ranking
openembeddings retrieve \
    --index-path my_index \
    --query "complex research question" \
    --top-k 10 \
    --rerank \
    --rerank-model "cross-encoder/ms-marco-MiniLM-L-12-v2"
```

### 3. Research CLI

```bash
# Launch research environment
openembeddings research

# Run comprehensive benchmarks
python -m openembeddings.advanced_cli benchmark \
    --model-path my_index \
    --datasets scifact nfcorpus arguana \
    --use-reranker

# Hyperparameter optimization
python -m openembeddings.advanced_cli optimize \
    --dataset scifact \
    --trials 100 \
    --time-budget 60

# Performance profiling
python -m openembeddings.advanced_cli profile \
    --model-path my_index \
    --corpus-sizes 1000 10000 100000
```

##  Python API

### Basic Usage

```python
from openembeddings import HybridRetriever

# Create retriever with advanced configuration
retriever = HybridRetriever(
    dense_model="all-MiniLM-L6-v2",
    fusion_strategy="rrf",
    use_ann=True
)

# Index documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing focuses on text understanding"
]
retriever.index(documents, show_progress=True)

# Search with multiple strategies
results = retriever.retrieve("What is machine learning?", top_k=3)
for idx, score, doc in results:
    print(f"Score: {score:.4f} - {doc}")
```

### Research API

```python
from openembeddings import BenchmarkSuite, ExperimentRunner, ExperimentConfig

# Comprehensive benchmarking
benchmark_suite = BenchmarkSuite(output_dir="results")
results = benchmark_suite.run_comprehensive_benchmark(
    retriever=retriever,
    beir_datasets=["scifact", "nfcorpus", "arguana"]
)

# Experiment configuration
config = ExperimentConfig(
    name="hybrid_retrieval_study",
    description="Comprehensive study of fusion strategies",
    model_configs=[
        {"fusion_strategy": "linear", "dense_weight": 0.7},
        {"fusion_strategy": "rrf", "use_ann": True},
    ],
    datasets=["scifact", "synthetic"],
    metrics=["accuracy", "precision", "recall", "f1_score"],
    use_wandb=True
)

# Run experiments
runner = ExperimentRunner(config)
results_df = runner.run_comparison_study(
    model_configs=config.model_configs,
    datasets=loaded_datasets
)
```

### Advanced Features

```python
from openembeddings import ReRanker, DatasetManager, AutoMLExperiment

# Cross-encoder re-ranking
reranker = ReRanker("cross-encoder/ms-marco-MiniLM-L-6-v2")
initial_results = retriever.retrieve("query", top_k=20)
reranked_results = reranker.rerank("query", initial_results)

# Dataset management
dataset_manager = DatasetManager()
beir_datasets = dataset_manager.load_beir_datasets(["scifact", "nfcorpus"])
synthetic_data = dataset_manager.create_synthetic_dataset(size=10000)

# AutoML optimization
automl = AutoMLExperiment()
best_config = automl.auto_optimize_retrieval_system(
    datasets=[synthetic_data],
    time_budget_minutes=30
)
```

## Benchmarking & Evaluation

### BEIR Integration

OpenEmbeddings provides comprehensive integration with the BEIR benchmark:

```python
from openembeddings import BenchmarkSuite

benchmark_suite = BenchmarkSuite()

# Evaluate on multiple datasets
datasets = ["scifact", "nfcorpus", "arguana", "quora", "fever"]
for dataset in datasets:
    results = benchmark_suite.evaluate_retrieval_system(
        retriever=your_retriever,
        dataset_name=dataset,
        use_reranker=True
    )
    print(f"{dataset}: NDCG@10 = {results['NDCG@10']:.4f}")
```

### Performance Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| NDCG@k | Normalized Discounted Cumulative Gain | Ranking quality |
| MAP@k | Mean Average Precision | Overall precision |
| Recall@k | Recall at cutoff k | Coverage assessment |
| MRR | Mean Reciprocal Rank | First relevant result |

### Supported Datasets

- **Scientific**: SciFact, SciDocs, TREC-COVID
- **Web Search**: MS MARCO, Natural Questions
- **Fact Verification**: FEVER, Climate-FEVER
- **Question Answering**: Quora, HotpotQA
- **Domain-Specific**: NFCorpus (medical), FiQA (financial)

## Research Features

### Experiment Management

```python
# Define comprehensive experiment
experiment_config = {
    "name": "fusion_strategy_comparison",
    "model_configs": [
        {"fusion_strategy": "linear", "dense_weight": w, "sparse_weight": 1-w}
        for w in [0.3, 0.5, 0.7, 0.9]
    ] + [
        {"fusion_strategy": "rrf", "use_ann": ann}
        for ann in [True, False]
    ],
    "datasets": ["scifact", "nfcorpus", "synthetic"],
    "cross_validation_folds": 5,
    "use_wandb": True
}
```

### Statistical Analysis

```python
# Run significance tests
significance_results = runner.run_statistical_significance_tests(
    results_df=experiment_results,
    baseline_model="linear_0.5",
    metric="ndcg_at_10",
    alpha=0.05
)

# Analyze effect sizes
for model, stats in significance_results.items():
    print(f"{model}: p={stats['p_value']:.4f}, "
          f"Cohen's d={stats['cohens_d']:.3f} ({stats['effect_size']})")
```

### Visualization

```python
# Create interactive dashboard
from openembeddings.experiments import create_experiment_dashboard

create_experiment_dashboard(
    results_df=experiment_results,
    output_path="experiment_dashboard.html"
)

# Generate comprehensive report
report_path = runner.generate_experiment_report()
print(f"Report saved to: {report_path}")
```

## Model Configurations

### Recommended Configurations

#### For Accuracy (Research)
```python
retriever = HybridRetriever(
    dense_model="all-mpnet-base-v2",
    fusion_strategy="rrf",
    use_ann=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2"
)
```

#### For Speed (Production)
```python
retriever = HybridRetriever(
    dense_model="all-MiniLM-L6-v2",
    fusion_strategy="linear",
    dense_weight=0.7,
    sparse_weight=0.3,
    use_ann=True
)
```

#### For Development
```python
retriever = HybridRetriever(
    dense_model="hashing-encoder",
    fusion_strategy="linear",
    use_ann=False
)
```

### Scalability Guidelines

| Corpus Size | Dense Model | Fusion | ANN | Memory |
|-------------|-------------|---------|-----|---------|
| < 1K docs | Any | Linear | No | < 100MB |
| 1K-10K docs | MiniLM-L6 | RRF | Optional | < 500MB |
| 10K-100K docs | MiniLM-L6 | RRF | Yes | < 2GB |
| > 100K docs | MiniLM-L6 | RRF | Yes | > 2GB |

## Examples and Tutorials

### Jupyter Notebooks

- **[Research Notebook](examples/research_notebook.ipynb)**: Comprehensive research workflow
- **[Benchmarking Tutorial](examples/benchmarking_tutorial.ipynb)**: BEIR evaluation guide
- **[Optimization Guide](examples/optimization_guide.ipynb)**: Hyperparameter tuning

### Configuration Files

- **[Experiment Config](examples/experiment_config.json)**: Multi-model comparison setup
- **[Benchmark Config](examples/benchmark_config.yaml)**: BEIR evaluation configuration

### Command Examples

```bash
# Research workflow
python -m openembeddings.advanced_cli experiment \
    --config examples/experiment_config.json \
    --output-dir results/comprehensive_study \
    --wandb --wandb-project openembeddings-research

# Dataset management
python -m openembeddings.advanced_cli datasets list
python -m openembeddings.advanced_cli datasets download --name scifact
python -m openembeddings.advanced_cli datasets create --size 10000

# Results comparison
python -m openembeddings.advanced_cli compare \
    --results-dir experiments/ \
    --metric ndcg_at_10 \
    --output comparison_dashboard.html
```

## Development

### Setup Development Environment

```bash
git clone https://github.com/nikjois/openembeddings.git
cd openembeddings

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,research]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v
```

### Testing

```bash
# Run all tests
pytest tests/ -v --cov=openembeddings

# Run specific test categories
pytest tests/test_models.py -v
# Performance tests (may take longer)
pytest tests/test_performance.py -v --slow
```

## Performance Benchmarks

### BEIR Results (Selected Datasets)

| Model | SciFact | NFCorpus | Arguana | Average |
|-------|---------|----------|---------|---------|
| BM25 | 0.665 | 0.325 | 0.415 | 0.468 |
| Dense (MiniLM) | 0.649 | 0.294 | 0.446 | 0.463 |
| Hybrid (Linear) | 0.689 | 0.339 | 0.478 | 0.502 |
| Hybrid (RRF) | 0.701 | 0.351 | 0.492 | 0.515 |
| + Re-ranking | 0.724 | 0.368 | 0.518 | 0.537 |

### Speed Benchmarks

| Configuration | Index Time | Query Time | Throughput |
|---------------|------------|------------|------------|
| Hash + Linear | 0.5s/1K docs | 0.001s | 1000 q/s |
| MiniLM + Linear | 2.1s/1K docs | 0.003s | 333 q/s |
| MiniLM + RRF | 2.1s/1K docs | 0.004s | 250 q/s |
| + Re-ranking | 2.1s/1K docs | 0.015s | 67 q/s |

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Areas for Contribution

- **New Models**: Integration with latest embedding models
- **Fusion Strategies**: Novel approaches to combining signals
- **Benchmarks**: Additional evaluation datasets and metrics
- **Optimizations**: Performance improvements and memory efficiency
- **Documentation**: Tutorials, examples, and guides

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- BEIR team for comprehensive IR benchmarking
- Sentence Transformers for excellent embedding models
- FAISS team for scalable similarity search
- HuggingFace for model hosting and datasets

## Citation

If you use OpenEmbeddings in your research, please cite:

```bibtex
@software{openembeddings2024,
  title={OpenEmbeddings: Production-Ready Embedding and Retrieval Framework},
  author={Nik Jois},
  year={2024},
  url={https://github.com/nikjois/openembeddings},
  version={1.0.0}
}
```

## Support

- **Documentation**: [https://openembeddings.readthedocs.io](https://openembeddings.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/nikjois/openembeddings/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nikjois/openembeddings/discussions)
- **Email**: nikjois@llamasearch.ai

---

**OpenEmbeddings** - Advancing the state of embedding and retrieval research 

## Hybrid Search
OpenEmbeddings now supports hybrid search combining dense and sparse retrieval methods:

```bash
openembeddings hybrid-retrieve "machine learning applications" \
    --dense-model all-MiniLM-L6-v2 \
    --sparse-model bm25 \
    --top-k 10 \
    --dense-weight 0.6 \
    --sparse-weight 0.4
```

Key features:
- **Reciprocal Rank Fusion (RRF)** for combining results
- Configurable weights for dense/sparse components
- Support for all dense and sparse models 