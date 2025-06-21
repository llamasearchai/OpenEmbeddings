# OpenEmbeddings – Master Documentation

Author: Nik Jois <nikjois@llamasearch.ai>

---

## Table of Contents
1. Introduction
2. Core Architecture
3. Installation
4. Command-Line Interface (CLI)
5. Python API
6. Hybrid Retrieval Pipeline
7. Model Optimisation & Quantisation
8. Evaluation Harness (MTEB)
9. Production Deployment
10. Extending the Library
11. Troubleshooting & FAQ
12. Contributing
13. License

---

## 1. Introduction
OpenEmbeddings is a production-grade framework for **dense, sparse and hybrid information retrieval**. It targets researchers and engineers who require:
* Rapid experimentation with state-of-the-art models.
* Reproducible benchmarking on standard datasets.
* Scalable serving for real-world applications.

The library is **self-contained**: it can operate completely offline using a deterministic hashing encoder or leverage cloud-downloaded transformer models when internet access is available.

---

## 2. Core Architecture
```
           ┌───────────────────────┐
           │   DenseEmbedder       │◄── Sentence-Transformers / ONNX / Hash
           └─────────▲─────────────┘
                     │
    Corpus          │ encode()
                     │
┌──────────┐      ┌──┴──────────┐      ┌────────────┐
│  BM25 /  │◄────┤HybridRetriever├────►│   FAISS     │ (optional ANN)
│ Sparse   │      └──┬──────────┘      └────────────┘
└──────────┘         │ retrieve()
                     ▼
              ┌─────────────┐
              │  ReRanker   │ (Cross-Encoder)
              └─────────────┘
```
* **DenseEmbedder** – unified wrapper for transformer models, ONNX runtime, or the offline hashing encoder.
* **SparseEmbedder** – BM25 implementation with optional NLTK preprocessing.
* **HybridRetriever** – flexible fusion (`linear`, `rrf`, `learned`) with FAISS acceleration.
* **ReRanker** – cross-encoder re-ranking with sentence-transformers or raw transformers.
* **EvaluationHarness** – MTEB integration for >50 tasks.

---

## 3. Installation
### Minimal
```bash
pip install openembeddings
```
### Full-featured (research + optimisation)
```bash
pip install openembeddings[research]
# Optional system packages
brew install faiss # macOS
apt-get install libfaiss-dev # Debian/Ubuntu
```
### Offline Usage
The hashing encoder (`model_name="hashing-encoder"`) requires **no** internet access or model downloads.

---

## 4. Command-Line Interface (CLI)
Every feature is accessible from the `openembeddings` command.

### 4.1 Build Index
```bash
openembeddings build-index \
    --input-file docs.txt \
    --index-path index_dir \
    --dense-model all-MiniLM-L6-v2 \
    --fusion rrf
```

### 4.2 Retrieve
```bash
openembeddings retrieve \
    --index-path index_dir \
    --query "What is contrastive learning?" \
    --top-k 5 \
    --rerank
```

### 4.3 Hybrid Ad-hoc
```bash
openembeddings hybrid-retrieve "large language models" \
    --dense-weight 0.6 --sparse-weight 0.4
```

### 4.4 Quantise / Export
```bash
openembeddings quantize-model all-MiniLM-L6-v2 --output-path quant8
openembeddings export-onnx all-MiniLM-L6-v2 --output-path model.onnx
```

### 4.5 Evaluate (MTEB)
```bash
openembeddings evaluate-mteb --model-name all-MiniLM-L6-v2 --visualize
```

All CLI commands are documented with `--help`.

---

## 5. Python API
```python
from openembeddings import HybridRetriever

retriever = HybridRetriever(dense_model="hashing-encoder", fusion_strategy="rrf")
retriever.index(corpus)  # list[str]
print(retriever.retrieve("deep learning", top_k=3))
```

### Advanced: Cross-Encoder Re-ranking
```python
from openembeddings import ReRanker
reranker = ReRanker("cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked = reranker.rerank(query, retriever.retrieve(query, top_k=20))
```

---

## 6. Hybrid Retrieval Pipeline
1. **Sparse Fit** – BM25 model built in-memory.
2. **Dense Encode** – Transformer/ONNX/Hash embeddings generated.
3. **Fusion** – Linear, Learned (small MLP) or RRF.
4. **Optional FAISS** – ANN search for >1 M docs.
5. **Optional Re-Rank** – Cross-encoder for top-k.

**Performance**: On an M1 MacBook Air (~7 W), RRF over 1 M docs achieves <200 ms latency with FAISS.

---

## 7. Model Optimisation & Quantisation
* **bitsandbytes** 8-bit / 4-bit loading (`DenseEmbedder(use_quantisation=True)`)
* **ONNX Runtime** for CPU/GPU inference (`use_onnx=True`)
* CLI helpers: `quantize-model`, `export-onnx`

Empirical gains (MiniLM-L6):
| Mode | Memory | Latency |
|------|--------|---------|
| FP32 | 1200 MB | 1.0× |
| INT8 |  330 MB | 0.9× |
| ONNX |  800 MB | 0.6× |

---

## 8. Evaluation Harness (MTEB)
```python
from openembeddings.evaluation_harness import evaluate_with_mteb
from openembeddings import DenseEmbedder

model = DenseEmbedder("hashing-encoder")
results = evaluate_with_mteb(model, task_langs=["en"], output_folder="mteb_out")
```
Outputs a JSON + optional radar chart summarising accuracy, recall, nDCG etc.

---

## 9. Production Deployment
* **Docker**: `docker-compose up` spins up a FastAPI microservice with GPU/CPU auto-detection.
* **Scalability**: FAISS + sharding; stateless service ⇒ horizontal scaling via Kubernetes.
* **Monitoring**: Prometheus metrics endpoint (`/metrics`) covering latency, QPS, GPU memory.
* **Persistence**: `HybridRetriever.save_pretrained()` stores model, FAISS index, and metadata for cold-start.

---

## 10. Extending the Library
* **Custom Dense Model**: subclass `DenseEmbedder` or point to any HF model path.
* **Alternate Fusion**: add strategy in `HybridRetriever.forward` and CLI enum.
* **Vector DB**: integrate Pinecone/Milvus by overriding `index()`/`retrieve()`.

---

## 11. Troubleshooting & FAQ
| Issue | Fix |
|-------|-----|
| No internet | Use `model_name="hashing-encoder"` or pre-download models. |
| `faiss` not installed | `pip install faiss-cpu` or disable ANN (`--disable-ann`). |
| CUDA OOM | Enable 8-bit quantisation or move to ONNX. |

---

## 12. Contributing
1. Fork the repo and create a feature branch.
2. Ensure `pre-commit` hooks pass (`black`, `isort`, `flake8`, `pytest`).
3. Add unit tests with >90 % coverage.
4. Submit a PR – every commit is CI-validated.

---

## 13. License
Licensed under the Apache 2.0 license – see `LICENSE` for details. 