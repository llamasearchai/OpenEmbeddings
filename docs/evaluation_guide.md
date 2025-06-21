# Model Evaluation Guide

OpenEmbeddings provides comprehensive evaluation capabilities through integration with the Massive Text Embedding Benchmark (MTEB).

## Key Features
- **Standardized Benchmarking**: Evaluate models on 50+ tasks across 8 categories
- **Visual Reporting**: Generate radar charts for easy performance comparison
- **Multi-language Support**: Evaluate on 112+ languages
- **Reproducible Results**: Detailed output formats for sharing

## Running Evaluation

### Command Line
```bash
openembeddings evaluate-mteb \
    --model-name all-MiniLM-L6-v2 \
    --task-langs en \
    --output-dir my_results \
    --visualize
```

### Python API
```python
from openembeddings.evaluation_harness import evaluate_with_mteb
from openembeddings.models.dense_embedder import DenseEmbedder

model = DenseEmbedder(model_name="all-MiniLM-L6-v2")
results = evaluate_with_mteb(
    model,
    task_langs=["en"],
    output_folder="my_results"
)
```

## Available Tasks
MTEB includes over 50 tasks across:
- Bitext Mining
- Classification
- Clustering
- Pair Classification
- Reranking
- Retrieval
- STS (Semantic Textual Similarity)
- Summarization

See full list: [MTEB GitHub](https://github.com/embeddings-benchmark/mteb) 