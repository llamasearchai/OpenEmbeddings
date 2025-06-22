# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-19

### Added
- **Hybrid Search Pipeline**: Advanced retrieval system combining dense and sparse embeddings
  - Reciprocal Rank Fusion (RRF) algorithm for optimal result combination
  - Configurable fusion strategies (linear, rrf, weighted)
  - ANN (Approximate Nearest Neighbor) support with FAISS integration
  - CLI command: `openembeddings hybrid-retrieve`

- **Model Quantization & Optimization**: Production-ready model optimization
  - 8-bit and 4-bit quantization using bitsandbytes
  - ONNX Runtime support for 2-5x faster inference
  - Memory-efficient model loading and caching
  - CLI commands: `quantize-model`, `export-onnx`

- **Evaluation Harness**: Comprehensive model evaluation framework
  - MTEB (Massive Text Embedding Benchmark) integration
  - 58+ evaluation tasks across 7 categories
  - Automated benchmarking and performance analysis
  - Visualization with radar charts and performance metrics
  - CLI command: `evaluate-mteb`

- **Advanced Re-ranking**: Cross-encoder re-ranking with production features
  - Multiple cross-encoder architectures support
  - Batch processing with memory optimization
  - Intelligent caching and persistence
  - Multi-stage re-ranking pipelines
  - Quality assessment and analysis tools

- **Dense Embeddings**: Robust sentence transformer integration
  - Support for 1000+ pre-trained models
  - Automatic model quantization and optimization
  - Fallback mechanisms for offline usage
  - Memory-efficient batch processing

- **Sparse Embeddings**: Traditional IR methods with modern optimizations
  - TF-IDF, BM25, and custom sparse representations
  - NLTK integration with fallback tokenization
  - Configurable vocabulary and feature selection
  - Efficient sparse matrix operations

- **Web API**: FastAPI-based REST API service
  - Async endpoints for all embedding operations
  - Automatic API documentation with Swagger/OpenAPI
  - Health checks and monitoring endpoints
  - Production-ready with proper error handling

- **Comprehensive CLI**: Feature-rich command-line interface
  - 15+ commands covering all functionality
  - Progress bars and detailed logging
  - Configuration file support
  - Batch processing capabilities

- **Docker Support**: Containerized deployment
  - Multi-stage Docker builds for optimization
  - Docker Compose for development
  - Production-ready container configuration

### Technical Improvements
- **Testing**: Comprehensive test suite with 21 tests covering all modules
- **Documentation**: Extensive documentation with examples and guides
- **Type Safety**: Full type annotations throughout codebase
- **Error Handling**: Robust error handling with graceful fallbacks
- **Performance**: Optimized for both speed and memory efficiency
- **Caching**: Intelligent caching at multiple levels
- **Logging**: Structured logging with configurable levels

### Dependencies
- Core: `torch`, `transformers`, `sentence-transformers`, `numpy`, `scipy`
- Optional: `faiss-cpu`, `bitsandbytes`, `onnxruntime`, `mteb`
- Web: `fastapi`, `uvicorn`, `pydantic`
- CLI: `click`, `rich`, `tqdm`
- Development: `pytest`, `black`, `isort`, `mypy`

### Author
- **Nik Jois** <nikjois@llamasearch.ai>

## [0.1.0] - 2024-12-19

### Added
- Initial project structure
- Basic embedding functionality
- Core CLI framework
- Docker configuration
- Test framework setup

## [1.0.1] - 2025-06-22

### Changed
- Polished README with project badges and concise tagline for improved presentation
- No code changes; documentation only

---

For more details, see the [documentation](docs/MASTER_DOCUMENTATION.md) and [examples](examples/). 