"""Comprehensive benchmarking and evaluation module for OpenEmbeddings.

This module provides state-of-the-art evaluation capabilities including:
- MTEB (Massive Text Embedding Benchmark) integration
- BEIR (Benchmarking IR) support
- Custom evaluation metrics
- Performance profiling and analysis
- Dataset loading and preprocessing
- Visualization and reporting

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    import psutil
    import memory_profiler
    _PROFILING_AVAILABLE = True
except ImportError:
    _PROFILING_AVAILABLE = False
    warnings.warn("Profiling libraries not available. Install psutil and memory_profiler for performance analysis.")

try:
    from datasets import load_dataset
    from evaluate import load as load_metric
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False
    warnings.warn("Datasets library not available. Install datasets for automatic dataset loading.")

try:
    import beir
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    _BEIR_AVAILABLE = True
except ImportError:
    _BEIR_AVAILABLE = False
    warnings.warn("BEIR library not available. Install beir for IR benchmarking.")

try:
    import mteb
    _MTEB_AVAILABLE = True
except ImportError:
    _MTEB_AVAILABLE = False
    # Only warn when MTEB functionality is actually needed
    # warnings.warn("MTEB library not available. Install mteb for embedding benchmarking.")

from .models.hybrid_retriever import HybridRetriever
from .models.dense_embedder import DenseEmbedder
from .models.sparse_embedder import SparseEmbedder
from .models.reranker import ReRanker


class BenchmarkSuite:
    """Comprehensive benchmarking suite for embedding and retrieval systems."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def evaluate_retrieval_system(
        self,
        retriever: HybridRetriever,
        dataset_name: str = "scifact",
        top_k: int = 100,
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> Dict[str, float]:
        """Evaluate a retrieval system on BEIR datasets."""
        if not _BEIR_AVAILABLE:
            raise ImportError("BEIR library not available. Install with: pip install beir")
            
        # Load dataset
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = GenericDataLoader(data_folder="datasets").download_and_unzip(url, "datasets")
        
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        
        # Index corpus
        documents = [corpus[doc_id]["text"] for doc_id in corpus]
        retriever.index(documents, show_progress=True)
        
        # Create document ID mapping
        doc_ids = list(corpus.keys())
        
        # Retrieve for all queries
        results = {}
        reranker = ReRanker(reranker_model) if use_reranker else None
        
        for query_id, query_text in queries.items():
            retrieved = retriever.retrieve(query_text, top_k=top_k)
            
            if use_reranker and reranker:
                retrieved = reranker.rerank(query_text, retrieved)
            
            # Convert to BEIR format
            results[query_id] = {
                doc_ids[doc_idx]: score 
                for doc_idx, score, _ in retrieved
            }
        
        # Evaluate
        evaluator = EvaluateRetrieval()
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [1, 3, 5, 10, 100])
        
        metrics = {
            "NDCG@1": ndcg["NDCG@1"],
            "NDCG@3": ndcg["NDCG@3"],
            "NDCG@5": ndcg["NDCG@5"],
            "NDCG@10": ndcg["NDCG@10"],
            "MAP@100": _map["MAP@100"],
            "Recall@1": recall["Recall@1"],
            "Recall@3": recall["Recall@3"],
            "Recall@5": recall["Recall@5"],
            "Recall@10": recall["Recall@10"],
            "Precision@1": precision["P@1"],
            "Precision@3": precision["P@3"],
            "Precision@5": precision["P@5"],
            "Precision@10": precision["P@10"],
        }
        
        return metrics
    
    def benchmark_embedding_models(
        self,
        models: List[str],
        tasks: List[str] = None,
        languages: List[str] = None
    ) -> pd.DataFrame:
        """Benchmark embedding models using MTEB."""
        if not _MTEB_AVAILABLE:
            raise ImportError("MTEB library not available. Install with: pip install mteb")
            
        if tasks is None:
            tasks = ["Clustering", "Retrieval", "STS", "Classification"]
            
        if languages is None:
            languages = ["en"]
            
        results = []
        
        for model_name in models:
            embedder = DenseEmbedder(model_name=model_name)
            
            # Create MTEB wrapper
            class MTEBWrapper:
                def __init__(self, embedder):
                    self.embedder = embedder
                    
                def encode(self, sentences, **kwargs):
                    return self.embedder.encode(sentences)
            
            model = MTEBWrapper(embedder)
            evaluation = mteb.MTEB(tasks=tasks, languages=languages)
            
            try:
                model_results = evaluation.run(model, output_folder=str(self.output_dir / f"mteb_{model_name}"))
                
                for task_result in model_results:
                    results.append({
                        "model": model_name,
                        "task": task_result["task_name"],
                        "score": task_result["score"],
                        "language": task_result.get("language", "en")
                    })
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def profile_performance(
        self,
        retriever: HybridRetriever,
        queries: List[str],
        corpus_sizes: List[int] = None,
        fusion_strategies: List[str] = None
    ) -> Dict[str, Any]:
        """Profile performance across different configurations."""
        if not _PROFILING_AVAILABLE:
            warnings.warn("Profiling libraries not available. Results may be incomplete.")
            
        if corpus_sizes is None:
            corpus_sizes = [100, 1000, 10000]
            
        if fusion_strategies is None:
            fusion_strategies = ["linear", "rrf"]
            
        results = {
            "indexing_time": {},
            "retrieval_time": {},
            "memory_usage": {},
            "throughput": {}
        }
        
        # Generate synthetic corpus
        base_docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing focuses on text understanding.",
            "Computer vision deals with image and video analysis.",
            "Reinforcement learning learns through trial and error."
        ]
        
        for corpus_size in corpus_sizes:
            # Create corpus of specified size
            corpus = (base_docs * (corpus_size // len(base_docs) + 1))[:corpus_size]
            
            for strategy in fusion_strategies:
                config_name = f"{strategy}_{corpus_size}"
                
                # Configure retriever
                retriever.fusion_strategy = strategy
                
                # Profile indexing
                start_time = time.time()
                if _PROFILING_AVAILABLE:
                    mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                retriever.index(corpus)
                
                indexing_time = time.time() - start_time
                
                if _PROFILING_AVAILABLE:
                    mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    memory_usage = mem_after - mem_before
                else:
                    memory_usage = 0
                
                # Profile retrieval
                retrieval_times = []
                # Determine safe top_k based on corpus size
                safe_top_k = min(10, len(retriever._documents) if hasattr(retriever, '_documents') else 10)
                for query in queries:
                    start_time = time.time()
                    retriever.retrieve(query, top_k=safe_top_k)
                    retrieval_times.append(time.time() - start_time)
                
                avg_retrieval_time = np.mean(retrieval_times)
                throughput = len(queries) / sum(retrieval_times)
                
                results["indexing_time"][config_name] = indexing_time
                results["retrieval_time"][config_name] = avg_retrieval_time
                results["memory_usage"][config_name] = memory_usage
                results["throughput"][config_name] = throughput
        
        return results
    
    def load_research_datasets(self) -> Dict[str, Any]:
        """Load high-quality research datasets from HuggingFace."""
        if not _DATASETS_AVAILABLE:
            raise ImportError("Datasets library not available. Install with: pip install datasets")
            
        datasets = {}
        
        # Text classification datasets
        try:
            datasets["imdb"] = load_dataset("imdb")
            datasets["ag_news"] = load_dataset("ag_news")
            datasets["yelp_polarity"] = load_dataset("yelp_polarity")
        except Exception as e:
            print(f"Error loading classification datasets: {e}")
        
        # Question answering datasets
        try:
            datasets["squad"] = load_dataset("squad")
            datasets["natural_questions"] = load_dataset("natural_questions")
            datasets["ms_marco"] = load_dataset("ms_marco", "v1.1")
        except Exception as e:
            print(f"Error loading QA datasets: {e}")
        
        # Information retrieval datasets
        try:
            datasets["scifact"] = load_dataset("scifact")
            datasets["fever"] = load_dataset("fever", "v1.0")
        except Exception as e:
            print(f"Error loading IR datasets: {e}")
        
        # Semantic similarity datasets
        try:
            datasets["stsb"] = load_dataset("stsb_multi_mt")
            datasets["sick"] = load_dataset("sick")
        except Exception as e:
            print(f"Error loading similarity datasets: {e}")
        
        return datasets
    
    def create_visualization_dashboard(self, results: Dict[str, Any]) -> None:
        """Create comprehensive visualization dashboard."""
        
        # Performance comparison plot
        if "indexing_time" in results:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Indexing Time", "Retrieval Time", "Memory Usage", "Throughput"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            configs = list(results["indexing_time"].keys())
            
            # Indexing time
            fig.add_trace(
                go.Bar(x=configs, y=list(results["indexing_time"].values()), name="Indexing Time"),
                row=1, col=1
            )
            
            # Retrieval time
            fig.add_trace(
                go.Bar(x=configs, y=list(results["retrieval_time"].values()), name="Retrieval Time"),
                row=1, col=2
            )
            
            # Memory usage
            fig.add_trace(
                go.Bar(x=configs, y=list(results["memory_usage"].values()), name="Memory Usage (MB)"),
                row=2, col=1
            )
            
            # Throughput
            fig.add_trace(
                go.Bar(x=configs, y=list(results["throughput"].values()), name="Throughput (queries/sec)"),
                row=2, col=2
            )
            
            fig.update_layout(
                title="OpenEmbeddings Performance Benchmark",
                showlegend=False,
                height=800
            )
            
            fig.write_html(str(self.output_dir / "performance_dashboard.html"))
            
        # Save results as JSON
        with open(self.output_dir / "benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    
    def run_comprehensive_benchmark(
        self,
        retriever: HybridRetriever,
        test_queries: List[str] = None,
        beir_datasets: List[str] = None
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        
        if test_queries is None:
            test_queries = [
                "What is machine learning?",
                "How do neural networks work?",
                "Explain natural language processing",
                "What are the applications of AI?",
                "How does deep learning differ from traditional ML?"
            ]
            
        if beir_datasets is None:
            beir_datasets = ["scifact", "nfcorpus", "arguana"]
            
        results = {}
        
        # Performance profiling
        print("Running performance profiling...")
        results["performance"] = self.profile_performance(retriever, test_queries)
        
        # BEIR evaluation
        if _BEIR_AVAILABLE:
            print("Running BEIR evaluation...")
            results["beir"] = {}
            for dataset in beir_datasets:
                try:
                    results["beir"][dataset] = self.evaluate_retrieval_system(
                        retriever, dataset_name=dataset
                    )
                except Exception as e:
                    print(f"Error evaluating on {dataset}: {e}")
                    continue
        
        # Create visualizations
        print("Creating visualization dashboard...")
        self.create_visualization_dashboard(results)
        
        # Save comprehensive report
        self.save_benchmark_report(results)
        
        return results
    
    def save_benchmark_report(self, results: Dict[str, Any]) -> None:
        """Save comprehensive benchmark report."""
        
        report_path = self.output_dir / "benchmark_report.md"
        
        with open(report_path, "w") as f:
            f.write("# OpenEmbeddings Benchmark Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
            
            # Performance summary
            if "performance" in results:
                f.write("## Performance Summary\n\n")
                perf = results["performance"]
                
                if "indexing_time" in perf:
                    f.write("### Indexing Performance\n")
                    for config, time_taken in perf["indexing_time"].items():
                        f.write(f"- {config}: {time_taken:.2f}s\n")
                    f.write("\n")
                
                if "retrieval_time" in perf:
                    f.write("### Retrieval Performance\n")
                    for config, time_taken in perf["retrieval_time"].items():
                        f.write(f"- {config}: {time_taken:.4f}s per query\n")
                    f.write("\n")
                
                if "throughput" in perf:
                    f.write("### Throughput\n")
                    for config, tps in perf["throughput"].items():
                        f.write(f"- {config}: {tps:.2f} queries/sec\n")
                    f.write("\n")
            
            # BEIR results
            if "beir" in results:
                f.write("## BEIR Evaluation Results\n\n")
                for dataset, metrics in results["beir"].items():
                    f.write(f"### {dataset.upper()}\n")
                    for metric, value in metrics.items():
                        f.write(f"- {metric}: {value:.4f}\n")
                    f.write("\n")
            
            f.write("## Recommendations\n\n")
            f.write("Based on the benchmark results:\n\n")
            
            # Add recommendations based on results
            if "performance" in results and "throughput" in results["performance"]:
                best_config = max(results["performance"]["throughput"], 
                                key=results["performance"]["throughput"].get)
                f.write(f"- Best throughput configuration: {best_config}\n")
            
            if "beir" in results:
                best_dataset_performance = {}
                for dataset, metrics in results["beir"].items():
                    best_dataset_performance[dataset] = metrics.get("NDCG@10", 0)
                
                if best_dataset_performance:
                    best_dataset = max(best_dataset_performance, key=best_dataset_performance.get)
                    f.write(f"- Best performing dataset: {best_dataset}\n")
        
        print(f"Benchmark report saved to: {report_path}")


class DatasetManager:
    """Manager for loading and preprocessing research datasets."""
    
    def __init__(self, cache_dir: str = "dataset_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def load_beir_datasets(self, datasets: List[str] = None) -> Dict[str, Any]:
        """Load BEIR datasets for information retrieval evaluation."""
        if not _BEIR_AVAILABLE:
            raise ImportError("BEIR library not available. Install with: pip install beir")
            
        if datasets is None:
            datasets = [
                "scifact", "nfcorpus", "arguana", "quora", "scidocs",
                "fever", "climate-fever", "dbpedia-entity", "hotpotqa"
            ]
        
        loaded_datasets = {}
        
        for dataset_name in datasets:
            try:
                url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
                data_path = GenericDataLoader(data_folder=str(self.cache_dir)).download_and_unzip(url, str(self.cache_dir))
                
                corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
                
                loaded_datasets[dataset_name] = {
                    "corpus": corpus,
                    "queries": queries,
                    "qrels": qrels
                }
                
                print(f"Loaded {dataset_name}: {len(corpus)} docs, {len(queries)} queries")
                
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
                continue
        
        return loaded_datasets
    
    def create_synthetic_dataset(
        self,
        size: int = 10000,
        domains: List[str] = None
    ) -> Dict[str, List[str]]:
        """Create synthetic dataset for testing and development."""
        
        if domains is None:
            domains = ["technology", "science", "health", "education", "business"]
        
        # Template sentences for each domain
        templates = {
            "technology": [
                "Artificial intelligence is revolutionizing {field}.",
                "Machine learning algorithms can {action} with high accuracy.",
                "Deep learning models are used for {application}.",
                "Neural networks excel at {task} in {domain}.",
                "Computer vision systems can {capability} in real-time."
            ],
            "science": [
                "Research in {field} has shown that {finding}.",
                "Scientists have discovered {discovery} in {area}.",
                "The study of {subject} reveals {insight}.",
                "Experiments demonstrate {result} under {conditions}.",
                "Analysis of {data} indicates {conclusion}."
            ],
            "health": [
                "Medical research shows {treatment} is effective for {condition}.",
                "Healthcare professionals recommend {intervention} for {disease}.",
                "Clinical trials demonstrate {therapy} improves {outcome}.",
                "Patients with {diagnosis} benefit from {approach}.",
                "Prevention strategies include {method} to reduce {risk}."
            ],
            "education": [
                "Students learn {subject} through {method}.",
                "Educational research shows {finding} improves {outcome}.",
                "Teaching {topic} requires {approach} and {resources}.",
                "Learning {skill} helps students {benefit}.",
                "Academic performance improves with {strategy}."
            ],
            "business": [
                "Companies use {technology} to {advantage}.",
                "Business strategies focus on {goal} through {method}.",
                "Market analysis shows {trend} in {sector}.",
                "Organizations implement {solution} to {objective}.",
                "Economic factors influence {decision} in {industry}."
            ]
        }
        
        # Fill-in options
        fill_ins = {
            "field": ["healthcare", "finance", "manufacturing", "retail", "transportation"],
            "action": ["classify", "predict", "optimize", "automate", "analyze"],
            "application": ["image recognition", "natural language processing", "recommendation systems"],
            "task": ["pattern recognition", "data analysis", "decision making"],
            "domain": ["medical imaging", "financial markets", "social media"],
            "capability": ["detect objects", "recognize faces", "track movement"],
            "finding": ["new mechanisms", "improved methods", "significant correlations"],
            "discovery": ["novel compounds", "genetic markers", "cellular processes"],
            "area": ["molecular biology", "quantum physics", "environmental science"],
            "subject": ["climate change", "human behavior", "material properties"],
            "insight": ["complex interactions", "underlying patterns", "causal relationships"],
            "result": ["positive outcomes", "significant improvements", "measurable changes"],
            "conditions": ["controlled environments", "specific parameters", "optimal settings"],
            "data": ["genomic sequences", "atmospheric measurements", "behavioral patterns"],
            "conclusion": ["strong evidence", "clear trends", "significant associations"],
            # Add more fill-ins as needed...
        }
        
        import random
        
        documents = []
        queries = []
        
        # Generate documents
        for _ in range(size):
            domain = random.choice(domains)
            template = random.choice(templates[domain])
            
            # Fill in template
            filled_template = template
            for placeholder in fill_ins:
                if f"{{{placeholder}}}" in filled_template:
                    filled_template = filled_template.replace(
                        f"{{{placeholder}}}", 
                        random.choice(fill_ins[placeholder])
                    )
            
            documents.append(filled_template)
        
        # Generate queries (shorter, question-like)
        query_templates = [
            "What is {topic}?",
            "How does {concept} work?",
            "Explain {subject}",
            "What are the benefits of {technology}?",
            "How to {action} using {method}?"
        ]
        
        topics = ["machine learning", "artificial intelligence", "data science", "neural networks"]
        concepts = ["deep learning", "natural language processing", "computer vision"]
        subjects = ["reinforcement learning", "supervised learning", "unsupervised learning"]
        technologies = ["blockchain", "cloud computing", "edge computing"]
        actions = ["optimize", "implement", "deploy", "analyze"]
        methods = ["algorithms", "frameworks", "tools", "techniques"]
        
        for _ in range(size // 10):  # Generate 10% as many queries as documents
            template = random.choice(query_templates)
            
            # Fill in query template
            filled_query = template
            if "{topic}" in filled_query:
                filled_query = filled_query.replace("{topic}", random.choice(topics))
            if "{concept}" in filled_query:
                filled_query = filled_query.replace("{concept}", random.choice(concepts))
            if "{subject}" in filled_query:
                filled_query = filled_query.replace("{subject}", random.choice(subjects))
            if "{technology}" in filled_query:
                filled_query = filled_query.replace("{technology}", random.choice(technologies))
            if "{action}" in filled_query:
                filled_query = filled_query.replace("{action}", random.choice(actions))
            if "{method}" in filled_query:
                filled_query = filled_query.replace("{method}", random.choice(methods))
            
            queries.append(filled_query)
        
        return {
            "documents": documents,
            "queries": queries
        }


# Utility functions for advanced analysis
def analyze_embedding_quality(embeddings: np.ndarray, labels: List[str] = None) -> Dict[str, float]:
    """Analyze the quality of embeddings using various metrics."""
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    
    metrics = {}
    
    # Dimensionality
    metrics["dimensions"] = embeddings.shape[1]
    metrics["num_samples"] = embeddings.shape[0]
    
    # Clustering quality (if labels provided)
    if labels is not None:
        unique_labels = list(set(labels))
        if len(unique_labels) > 1:
            kmeans = KMeans(n_clusters=len(unique_labels), random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            metrics["silhouette_score"] = silhouette_score(embeddings, cluster_labels)
    
    # Embedding statistics
    metrics["mean_norm"] = np.mean(np.linalg.norm(embeddings, axis=1))
    metrics["std_norm"] = np.std(np.linalg.norm(embeddings, axis=1))
    
    # Cosine similarity statistics
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    
    # Remove diagonal (self-similarity)
    sim_matrix_no_diag = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]
    
    metrics["mean_cosine_similarity"] = np.mean(sim_matrix_no_diag)
    metrics["std_cosine_similarity"] = np.std(sim_matrix_no_diag)
    
    return metrics


def create_embedding_visualization(
    embeddings: np.ndarray,
    labels: List[str] = None,
    method: str = "tsne",
    output_path: str = "embedding_visualization.html"
) -> None:
    """Create interactive visualization of embeddings."""
    
    # Dimensionality reduction
    if method.lower() == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    elif method.lower() == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    elif method.lower() == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        except ImportError:
            print("UMAP not available, falling back to t-SNE")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Reduce dimensions
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create interactive plot
    if labels is not None:
        fig = px.scatter(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            color=labels,
            title=f"Embedding Visualization ({method.upper()})",
            labels={"x": "Component 1", "y": "Component 2"},
            hover_data={"index": list(range(len(embeddings)))}
        )
    else:
        fig = px.scatter(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            title=f"Embedding Visualization ({method.upper()})",
            labels={"x": "Component 1", "y": "Component 2"},
            hover_data={"index": list(range(len(embeddings)))}
        )
    
    fig.write_html(output_path)
    print(f"Visualization saved to: {output_path}")


def benchmark_quantization():
    """Benchmark different quantization settings."""
    from openembeddings.models.dense_embedder import DenseEmbedder
    
    # Test configurations
    configs = [
        {"name": "Baseline", "use_quantization": False, "use_onnx": False},
        {"name": "8-bit Quant", "use_quantization": True, "quantization_config": {"quant_type": "8bit"}},
        {"name": "4-bit Quant", "use_quantization": True, "quantization_config": {"quant_type": "4bit"}},
        {"name": "ONNX", "use_onnx": True}
    ]
    
    results = []
    for config in configs:
        embedder = DenseEmbedder(model_name="all-MiniLM-L6-v2", **config)
        
        # Benchmark embedding speed
        start = time.time()
        embeddings = embedder.embed(TEXTS)
        duration = time.time() - start
        
        # Memory usage
        mem_usage = get_memory_usage()
        
        results.append({
            "config": config['name'],
            "time": duration,
            "memory": mem_usage,
            "embedding_shape": embeddings.shape
        })
    
    return results 