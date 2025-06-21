"""Comprehensive utilities for OpenEmbeddings framework.

This module provides utility functions for:
- Data preprocessing and cleaning
- Text processing and normalization
- Evaluation metrics and analysis
- Visualization and reporting
- System monitoring and optimization
- Configuration management

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import re
import json
import time
import hashlib
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False

try:
    import psutil
    _SYSTEM_MONITORING_AVAILABLE = True
except ImportError:
    _SYSTEM_MONITORING_AVAILABLE = False

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# Text Processing Utilities
def clean_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = False,
    remove_numbers: bool = False,
    remove_extra_whitespace: bool = True,
    min_length: int = 1
) -> str:
    """Clean and normalize text with various options.
    
    Args:
        text: Input text to clean
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation marks
        remove_numbers: Remove numeric characters
        remove_extra_whitespace: Normalize whitespace
        min_length: Minimum length filter
        
    Returns:
        Cleaned text string
    """
    if not text or len(text) < min_length:
        return ""
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove punctuation
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    if remove_extra_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text if len(text) >= min_length else ""

def preprocess_documents(
    documents: List[str],
    min_length: int = 10,
    max_length: Optional[int] = None,
    remove_duplicates: bool = True,
    clean_config: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Preprocess a collection of documents.
    
    Args:
        documents: List of document strings
        min_length: Minimum document length
        max_length: Maximum document length (None for no limit)
        remove_duplicates: Remove duplicate documents
        clean_config: Configuration for text cleaning
        
    Returns:
        Preprocessed list of documents
    """
    if clean_config is None:
        clean_config = {
            "lowercase": True,
            "remove_extra_whitespace": True
        }
    
    processed_docs = []
    seen_docs = set() if remove_duplicates else None
    
    for doc in documents:
        # Clean text
        cleaned_doc = clean_text(doc, **clean_config)
        
        # Length filtering
        if len(cleaned_doc) < min_length:
            continue
        if max_length and len(cleaned_doc) > max_length:
            cleaned_doc = cleaned_doc[:max_length]
        
        # Duplicate removal
        if remove_duplicates:
            doc_hash = hashlib.md5(cleaned_doc.encode()).hexdigest()
            if doc_hash in seen_docs:
                continue
            seen_docs.add(doc_hash)
        
        processed_docs.append(cleaned_doc)
    
    return processed_docs

def extract_keywords(
    text: str,
    method: str = "frequency",
    top_k: int = 10,
    min_length: int = 3
) -> List[Tuple[str, float]]:
    """Extract keywords from text using various methods.
    
    Args:
        text: Input text
        method: Extraction method ('frequency', 'tfidf')
        top_k: Number of keywords to return
        min_length: Minimum keyword length
        
    Returns:
        List of (keyword, score) tuples
    """
    # Simple tokenization
    words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + ',}\b', text.lower())
    
    if method == "frequency":
        word_counts = Counter(words)
        total_words = len(words)
        keywords = [(word, count/total_words) for word, count in word_counts.most_common(top_k)]
        
    elif method == "tfidf":
        # Simple TF-IDF approximation
        word_counts = Counter(words)
        doc_length = len(words)
        
        # Assume single document for simplicity
        keywords = []
        for word, count in word_counts.most_common(top_k):
            tf = count / doc_length
            # Simple IDF approximation (assume corpus of 1000 docs)
            idf = np.log(1000 / (1 + 1))  # +1 for smoothing
            tfidf_score = tf * idf
            keywords.append((word, tfidf_score))
    
    else:
        raise ValueError(f"Unknown keyword extraction method: {method}")
    
    return keywords


# Evaluation Utilities
def compute_retrieval_metrics(
    retrieved_docs: List[int],
    relevant_docs: List[int],
    k_values: List[int] = None
) -> Dict[str, float]:
    """Compute standard retrieval metrics.
    
    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs
        k_values: Values of k for precision@k, recall@k
        
    Returns:
        Dictionary of computed metrics
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]
    
    relevant_set = set(relevant_docs)
    retrieved_set = set(retrieved_docs)
    
    # Basic metrics
    metrics = {
        "num_retrieved": len(retrieved_docs),
        "num_relevant": len(relevant_docs),
        "num_relevant_retrieved": len(relevant_set & retrieved_set)
    }
    
    # Precision and Recall
    if len(retrieved_docs) > 0:
        metrics["precision"] = len(relevant_set & retrieved_set) / len(retrieved_docs)
    else:
        metrics["precision"] = 0.0
        
    if len(relevant_docs) > 0:
        metrics["recall"] = len(relevant_set & retrieved_set) / len(relevant_docs)
    else:
        metrics["recall"] = 0.0
    
    # F1 Score
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
    else:
        metrics["f1"] = 0.0
    
    # Precision@k and Recall@k
    for k in k_values:
        retrieved_at_k = set(retrieved_docs[:k])
        relevant_retrieved_at_k = len(relevant_set & retrieved_at_k)
        
        metrics[f"precision@{k}"] = relevant_retrieved_at_k / min(k, len(retrieved_docs)) if len(retrieved_docs) > 0 else 0.0
        metrics[f"recall@{k}"] = relevant_retrieved_at_k / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
    
    # Mean Reciprocal Rank (MRR)
    reciprocal_rank = 0.0
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_set:
            reciprocal_rank = 1.0 / (i + 1)
            break
    metrics["mrr"] = reciprocal_rank
    
    return metrics

def compute_ndcg(
    retrieved_docs: List[int],
    relevance_scores: Dict[int, float],
    k: int = 10
) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG).
    
    Args:
        retrieved_docs: List of retrieved document IDs
        relevance_scores: Dictionary mapping doc IDs to relevance scores
        k: Cutoff value for NDCG@k
        
    Returns:
        NDCG@k score
    """
    def dcg(scores: List[float]) -> float:
        return sum(score / np.log2(i + 2) for i, score in enumerate(scores))
    
    # DCG for retrieved documents
    retrieved_scores = [relevance_scores.get(doc_id, 0.0) for doc_id in retrieved_docs[:k]]
    dcg_score = dcg(retrieved_scores)
    
    # IDCG (ideal DCG)
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg_score = dcg(ideal_scores)
    
    # NDCG
    if idcg_score == 0:
        return 0.0
    return dcg_score / idcg_score


# Analysis Utilities
def analyze_embedding_distribution(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Analyze the distribution of embeddings.
    
    Args:
        embeddings: Array of embeddings (n_samples, n_features)
        labels: Optional labels for embeddings
        
    Returns:
        Analysis results dictionary
    """
    n_samples, n_features = embeddings.shape
    
    # Basic statistics
    norms = np.linalg.norm(embeddings, axis=1)
    
    analysis = {
        "n_samples": n_samples,
        "n_features": n_features,
        "norm_stats": {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
            "min": float(np.min(norms)),
            "max": float(np.max(norms)),
            "median": float(np.median(norms))
        }
    }
    
    # Pairwise similarity analysis
    if n_samples <= 1000:  # Only for small datasets to avoid memory issues
        similarities = cosine_similarity(embeddings) if _SKLEARN_AVAILABLE else np.dot(embeddings, embeddings.T)
        
        # Remove diagonal (self-similarities)
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        similarity_values = similarities[mask]
        
        analysis["similarity_stats"] = {
            "mean": float(np.mean(similarity_values)),
            "std": float(np.std(similarity_values)),
            "min": float(np.min(similarity_values)),
            "max": float(np.max(similarity_values)),
            "median": float(np.median(similarity_values))
        }
    
    # Dimensionality analysis
    try:
        # Compute effective dimensionality
        cov_matrix = np.cov(embeddings.T)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero values
        
        # Participation ratio (effective dimensionality)
        participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        
        analysis["dimensionality"] = {
            "effective_dimension": float(participation_ratio),
            "explained_variance_ratio": float(participation_ratio / n_features),
            "n_significant_components": len(eigenvalues)
        }
    except Exception:
        analysis["dimensionality"] = {"error": "Could not compute dimensionality analysis"}
    
    return analysis

def compare_retrievers(
    retriever_results: Dict[str, List[Tuple[int, float, str]]],
    ground_truth: Optional[List[int]] = None,
    queries: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compare results from multiple retrievers.
    
    Args:
        retriever_results: Dictionary mapping retriever names to results
        ground_truth: List of relevant document IDs
        queries: List of query strings
        
    Returns:
        Comparison analysis
    """
    comparison = {
        "retrievers": list(retriever_results.keys()),
        "n_retrievers": len(retriever_results),
        "metrics": {}
    }
    
    # Basic statistics
    for name, results in retriever_results.items():
        scores = [score for _, score, _ in results]
        comparison["metrics"][name] = {
            "n_results": len(results),
            "score_stats": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores))
            }
        }
        
        # Compute retrieval metrics if ground truth available
        if ground_truth:
            doc_ids = [doc_id for doc_id, _, _ in results]
            metrics = compute_retrieval_metrics(doc_ids, ground_truth)
            comparison["metrics"][name].update(metrics)
    
    # Overlap analysis
    if len(retriever_results) >= 2:
        retriever_names = list(retriever_results.keys())
        overlap_matrix = {}
        
        for i, name1 in enumerate(retriever_names):
            for j, name2 in enumerate(retriever_names):
                if i <= j:
                    continue
                    
                docs1 = set(doc_id for doc_id, _, _ in retriever_results[name1])
                docs2 = set(doc_id for doc_id, _, _ in retriever_results[name2])
                
                overlap = len(docs1 & docs2)
                union = len(docs1 | docs2)
                jaccard = overlap / union if union > 0 else 0.0
                
                overlap_matrix[f"{name1}_vs_{name2}"] = {
                    "overlap": overlap,
                    "jaccard": jaccard
                }
        
        comparison["overlap_analysis"] = overlap_matrix
    
    return comparison


# Visualization Utilities
def plot_embedding_distribution(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    method: str = "tsne",
    output_path: Optional[str] = None
) -> None:
    """Plot 2D visualization of embeddings.
    
    Args:
        embeddings: Array of embeddings
        labels: Optional labels for coloring
        method: Dimensionality reduction method ('tsne', 'pca', 'umap')
        output_path: Path to save plot
    """
    if not _PLOTTING_AVAILABLE:
        warnings.warn("Matplotlib not available for plotting")
        return
    
    # Dimensionality reduction
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        except ImportError:
            warnings.warn("scikit-learn not available for t-SNE")
            return
    elif method == "pca":
        try:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        except ImportError:
            warnings.warn("scikit-learn not available for PCA")
            return
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        except ImportError:
            warnings.warn("UMAP not available")
            return
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    if labels:
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=[colors[i]], 
                label=label,
                alpha=0.6
            )
        plt.legend()
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
    
    plt.title(f"Embedding Visualization ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


# System Monitoring Utilities
class PerformanceMonitor:
    """Monitor system performance during operations."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.measurements = []
        
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        if _SYSTEM_MONITORING_AVAILABLE:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024**2  # MB
        
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return results."""
        if self.start_time is None:
            raise RuntimeError("Monitor not started")
            
        end_time = time.time()
        runtime = end_time - self.start_time
        
        results = {"runtime_seconds": runtime}
        
        if _SYSTEM_MONITORING_AVAILABLE and self.start_memory is not None:
            process = psutil.Process()
            end_memory = process.memory_info().rss / 1024**2  # MB
            results["memory_usage_mb"] = end_memory - self.start_memory
            results["peak_memory_mb"] = end_memory
        
        return results
    
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.stop()


# Configuration Utilities
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Validate configuration against schema."""
    errors = []
    
    # Simple validation logic
    for key, expected_type in schema.items():
        if key not in config:
            errors.append(f"Missing required key: {key}")
        elif not isinstance(config[key], expected_type):
            errors.append(f"Invalid type for {key}: expected {expected_type.__name__}, got {type(config[key]).__name__}")
    
    return errors


# Data Processing Utilities
def batch_process(
    items: List[Any],
    process_fn: Callable,
    batch_size: int = 32,
    show_progress: bool = True
) -> List[Any]:
    """Process items in batches with optional progress bar."""
    results = []
    
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(items), batch_size), desc="Processing batches")
        except ImportError:
            iterator = range(0, len(items), batch_size)
    else:
        iterator = range(0, len(items), batch_size)
    
    for i in iterator:
        batch = items[i:i + batch_size]
        batch_results = process_fn(batch)
        results.extend(batch_results)
    
    return results

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def deduplicate_documents(
    documents: List[str],
    method: str = "exact",
    threshold: float = 0.9
) -> List[str]:
    """Remove duplicate documents using various methods."""
    if method == "exact":
        seen = set()
        unique_docs = []
        for doc in documents:
            if doc not in seen:
                unique_docs.append(doc)
                seen.add(doc)
        return unique_docs
    
    elif method == "hash":
        seen_hashes = set()
        unique_docs = []
        for doc in documents:
            doc_hash = hashlib.md5(doc.encode()).hexdigest()
            if doc_hash not in seen_hashes:
                unique_docs.append(doc)
                seen_hashes.add(doc_hash)
        return unique_docs
    
    elif method == "similarity":
        if not _SKLEARN_AVAILABLE:
            warnings.warn("scikit-learn not available for similarity-based deduplication")
            return deduplicate_documents(documents, method="exact")
        
        # Simple similarity-based deduplication
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Vectorize documents
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        doc_vectors = vectorizer.fit_transform(documents)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(doc_vectors)
        
        # Mark documents to keep
        keep_mask = np.ones(len(documents), dtype=bool)
        
        for i in range(len(documents)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(documents)):
                if similarity_matrix[i, j] > threshold:
                    keep_mask[j] = False
        
        return [doc for i, doc in enumerate(documents) if keep_mask[i]]
    
    else:
        raise ValueError(f"Unknown deduplication method: {method}")


# Export utilities
__all__ = [
    # Text processing
    "clean_text",
    "preprocess_documents", 
    "extract_keywords",
    
    # Evaluation
    "compute_retrieval_metrics",
    "compute_ndcg",
    
    # Analysis
    "analyze_embedding_distribution",
    "compare_retrievers",
    
    # Visualization
    "plot_embedding_distribution",
    
    # System monitoring
    "PerformanceMonitor",
    
    # Configuration
    "load_config",
    "save_config",
    "validate_config",
    
    # Data processing
    "batch_process",
    "chunk_list",
    "deduplicate_documents",
]

def quantize_model(
    model: torch.nn.Module,
    quant_type: str = "8bit",
    dtype: torch.dtype = torch.float16
) -> torch.nn.Module:
    """Apply quantization to a model.
    
    Args:
        model: PyTorch model to quantize
        quant_type: Quantization type ('8bit' or '4bit')
        dtype: Data type for quantization
        
    Returns:
        Quantized model
    """
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear8bitLt, Linear4bit
        
        # Replace linear layers with quantized equivalents
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                if quant_type == "8bit":
                    quantized_layer = Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        has_fp16_weights=False,
                        threshold=6.0
                    )
                elif quant_type == "4bit":
                    quantized_layer = Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        quant_type="fp4",
                        compute_dtype=dtype
                    )
                else:
                    raise ValueError(f"Unsupported quant_type: {quant_type}")
                
                # Copy weights
                quantized_layer.weight = module.weight
                if module.bias is not None:
                    quantized_layer.bias = module.bias
                
                setattr(model, name, quantized_layer)
            else:
                # Recursively quantize child modules
                quantize_model(module, quant_type, dtype)
                
        return model
    except ImportError:
        raise RuntimeError("bitsandbytes required for quantization")

def convert_to_onnx(
    model: torch.nn.Module,
    tokenizer: Any,
    output_path: str,
    input_names: List[str] = ["input_ids", "attention_mask"],
    output_names: List[str] = ["last_hidden_state"],
    dynamic_axes: Dict[str, Dict[int, str]] = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"}
    },
    opset_version: int = 17
):
    """Convert PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer for the model
        output_path: Path to save ONNX model
        input_names: Input names for ONNX
        output_names: Output names for ONNX
        dynamic_axes: Dynamic axes configuration
        opset_version: ONNX opset version
    """
    import torch.onnx
    
    # Create dummy input
    dummy_input = tokenizer(
        ["dummy input"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Export model
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version
    ) 