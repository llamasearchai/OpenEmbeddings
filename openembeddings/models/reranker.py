"""Advanced cross-encoder re-ranking model with production features.

This module provides comprehensive re-ranking capabilities including:
- Multiple cross-encoder architectures
- Batch processing with memory optimization
- Intelligent caching and persistence
- Multi-stage re-ranking pipelines
- Quality assessment and analysis
- GPU acceleration and mixed precision

Author: Nik Jois <nikjois@llamasearch.ai>
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

try:
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    CrossEncoder = None
    CEBinaryClassificationEvaluator = None
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    _TRANSFORMERS_AVAILABLE = False

# Provide a safe alias for static type checking without requiring the library at runtime
if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import CrossEncoder as CrossEncoderType
else:
    CrossEncoderType = Any  # type: ignore

class ReRanker(nn.Module):
    """Advanced cross-encoder re-ranking model with production features.
    
    Args:
        model_name: Name or path of the cross-encoder model
        max_length: Maximum sequence length for input pairs
        device: Target device ('cpu', 'cuda', 'auto')
        batch_size: Default batch size for processing
        use_mixed_precision: Whether to use mixed precision training
        enable_caching: Whether to cache predictions
        cache_dir: Directory for caching
        trust_remote_code: Whether to trust remote code
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        device: Optional[str] = None,
        batch_size: int = 32,
        use_mixed_precision: bool = False,
        enable_caching: bool = True,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_mixed_precision = use_mixed_precision
        self.enable_caching = enable_caching
        self.trust_remote_code = trust_remote_code
        self.temperature = temperature
        
        # Device setup
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                self.device = f"cuda:{torch.cuda.current_device()}"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        # Cache setup
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".openembeddings_cache" / "reranker"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self._prediction_cache: Dict[str, float] = {}
        
        # Model initialization
        self.cross_encoder: Optional[CrossEncoderType] = None
        self.tokenizer = None
        self.model = None
        self.backend = "none"
        
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """Initialize the cross-encoder model."""
        if _SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device=self.device,
                    trust_remote_code=self.trust_remote_code
                )
                self.backend = "sentence_transformers"
                
                # Enable mixed precision
                if self.use_mixed_precision and torch.cuda.is_available():
                    self.cross_encoder.model = self.cross_encoder.model.half()
                    
            except Exception as e:
                warnings.warn(f"Failed to load with sentence-transformers: {e}")
                self._try_transformers_backend()
        else:
            self._try_transformers_backend()
            
    def _try_transformers_backend(self) -> None:
        """Try to initialize with transformers backend."""
        if _TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code
                ).to(self.device)
                
                if self.use_mixed_precision:
                    self.model = self.model.half()
                    
                self.backend = "transformers"
            except Exception as e:
                raise RuntimeError(f"Failed to load model with any backend: {e}")
        else:
            raise ImportError("Neither sentence-transformers nor transformers is available")

    def _get_cache_key(self, query: str, document: str) -> str:
        """Generate cache key for query-document pair."""
        import hashlib
        text = f"{self.model_name}:{query}:{document}:{self.temperature}"
        return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()

    def _predict_batch(
        self, 
        query_doc_pairs: List[Tuple[str, str]]
    ) -> np.ndarray:
        """Predict relevance scores for batch of query-document pairs."""
        if self.backend == "sentence_transformers":
            scores = self.cross_encoder.predict(
                query_doc_pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            return np.array(scores)
            
        elif self.backend == "transformers":
            # Process in batches
            all_scores = []
            
            for i in range(0, len(query_doc_pairs), self.batch_size):
                batch_pairs = query_doc_pairs[i:i + self.batch_size]
                
                # Tokenize batch
                texts = [f"{query} [SEP] {doc}" for query, doc in batch_pairs]
                
                inputs = self.tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Predict
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Apply temperature scaling
                    if self.temperature != 1.0:
                        logits = logits / self.temperature
                    
                    # Get probabilities or raw scores
                    if logits.shape[1] == 1:  # Regression
                        scores = logits.squeeze(-1)
                    else:  # Classification
                        scores = torch.softmax(logits, dim=1)[:, 1]  # Positive class
                    
                    all_scores.extend(scores.cpu().numpy())
            
            return np.array(all_scores)
        else:
            raise RuntimeError("No valid backend initialized")

    def rerank(
        self,
        query: str,
        initial_results: List[Tuple[int, float, str]],
        top_k: Optional[int] = None,
        use_cache: bool = True
    ) -> List[Tuple[int, float, str]]:
        """Re-rank initial retrieval results using cross-encoder.
        
        Args:
            query: Search query
            initial_results: List of (doc_id, score, document) tuples
            top_k: Number of results to return (default: same as input)
            use_cache: Whether to use prediction cache
            
        Returns:
            Re-ranked list of (doc_id, rerank_score, document) tuples
        """
        if not initial_results:
            return []
            
        top_k = top_k or len(initial_results)
        
        # Check cache first
        cached_scores = []
        uncached_pairs = []
        uncached_indices = []
        
        if use_cache and self.enable_caching:
            for i, (doc_id, score, document) in enumerate(initial_results):
                cache_key = self._get_cache_key(query, document)
                if cache_key in self._prediction_cache:
                    cached_scores.append((i, self._prediction_cache[cache_key]))
                else:
                    uncached_pairs.append((query, document))
                    uncached_indices.append(i)
        else:
            uncached_pairs = [(query, doc) for _, _, doc in initial_results]
            uncached_indices = list(range(len(initial_results)))
        
        # Predict uncached pairs
        new_scores = []
        if uncached_pairs:
            new_scores = self._predict_batch(uncached_pairs)
            
            # Save to cache
            if use_cache and self.enable_caching:
                for (q, doc), score in zip(uncached_pairs, new_scores):
                    cache_key = self._get_cache_key(q, doc)
                    self._prediction_cache[cache_key] = float(score)
        
        # Combine scores
        all_scores = {}
        
        # Add cached scores
        for idx, score in cached_scores:
            all_scores[idx] = score
            
        # Add new scores
        for idx, score in zip(uncached_indices, new_scores):
            all_scores[idx] = score
        
        # Create re-ranked results
        reranked_results = []
        for i, (doc_id, original_score, document) in enumerate(initial_results):
            rerank_score = all_scores[i]
            reranked_results.append((doc_id, float(rerank_score), document))
        
        # Sort by re-ranking score
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_results[:top_k]

    def score_pairs(
        self,
        query_doc_pairs: List[Tuple[str, str]],
        batch_size: Optional[int] = None,
        use_cache: bool = True
    ) -> List[float]:
        """Score query-document pairs directly.
        
        Args:
            query_doc_pairs: List of (query, document) pairs
            batch_size: Batch size for processing
            use_cache: Whether to use prediction cache
            
        Returns:
            List of relevance scores
        """
        batch_size = batch_size or self.batch_size
        
        if use_cache and self.enable_caching:
            # Check cache
            cached_scores = {}
            uncached_pairs = []
            
            for i, (query, doc) in enumerate(query_doc_pairs):
                cache_key = self._get_cache_key(query, doc)
                if cache_key in self._prediction_cache:
                    cached_scores[i] = self._prediction_cache[cache_key]
                else:
                    uncached_pairs.append((i, query, doc))
            
            # Predict uncached pairs
            if uncached_pairs:
                pairs_to_predict = [(query, doc) for _, query, doc in uncached_pairs]
                new_scores = self._predict_batch(pairs_to_predict)
                
                # Save to cache and results
                for (i, query, doc), score in zip(uncached_pairs, new_scores):
                    cache_key = self._get_cache_key(query, doc)
                    self._prediction_cache[cache_key] = float(score)
                    cached_scores[i] = float(score)
            
            # Return scores in original order
            return [cached_scores[i] for i in range(len(query_doc_pairs))]
        else:
            # Direct prediction
            scores = self._predict_batch(query_doc_pairs)
            return scores.tolist()

    def evaluate_quality(
        self,
        test_pairs: List[Tuple[str, str]],
        labels: List[int],
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """Evaluate re-ranker quality on test data.
        
        Args:
            test_pairs: List of (query, document) pairs
            labels: Binary relevance labels (0/1)
            metrics: Metrics to compute
            
        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1", "auc"]
            
        # Get predictions
        predictions = self.score_pairs(test_pairs, use_cache=False)
        binary_preds = [1 if score > 0.5 else 0 for score in predictions]
        
        results = {}
        
        # Compute metrics
        if "accuracy" in metrics:
            results["accuracy"] = sum(p == l for p, l in zip(binary_preds, labels)) / len(labels)
            
        if "precision" in metrics or "recall" in metrics or "f1" in metrics:
            tp = sum(p == 1 and l == 1 for p, l in zip(binary_preds, labels))
            fp = sum(p == 1 and l == 0 for p, l in zip(binary_preds, labels))
            fn = sum(p == 0 and l == 1 for p, l in zip(binary_preds, labels))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if "precision" in metrics:
                results["precision"] = precision
            if "recall" in metrics:
                results["recall"] = recall
            if "f1" in metrics:
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                results["f1"] = f1
                
        if "auc" in metrics:
            try:
                from sklearn.metrics import roc_auc_score
                results["auc"] = roc_auc_score(labels, predictions)
            except ImportError:
                warnings.warn("sklearn not available for AUC calculation")
                
        return results

    def analyze_predictions(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Analyze predictions for a query and documents.
        
        Args:
            query: Search query
            documents: List of candidate documents
            top_k: Number of top documents to analyze
            
        Returns:
            Analysis results including scores and statistics
        """
        # Score all pairs
        pairs = [(query, doc) for doc in documents]
        scores = self.score_pairs(pairs)
        
        # Create results with documents
        scored_docs = [(score, i, doc) for i, (score, doc) in enumerate(zip(scores, documents))]
        scored_docs.sort(reverse=True)
        
        analysis = {
            "query": query,
            "total_documents": len(documents),
            "score_statistics": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores))
            },
            "top_documents": [
                {
                    "rank": i + 1,
                    "score": float(score),
                    "document_id": doc_id,
                    "document": doc[:200] + "..." if len(doc) > 200 else doc
                }
                for i, (score, doc_id, doc) in enumerate(scored_docs[:top_k])
            ]
        }
        
        return analysis

    def save_pretrained(self, save_path: str) -> None:
        """Save model configuration and cache."""
        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "use_mixed_precision": self.use_mixed_precision,
            "trust_remote_code": self.trust_remote_code,
            "temperature": self.temperature,
            "backend": self.backend
        }
        
        with open(p / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            
        # Save prediction cache
        if self.enable_caching and self._prediction_cache:
            with open(p / "prediction_cache.json", "w", encoding="utf-8") as f:
                json.dump(self._prediction_cache, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_path: str, **kwargs) -> "ReRanker":
        """Load model from saved configuration."""
        p = Path(load_path)
        
        # Load configuration
        with open(p / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            
        # Override with kwargs
        config.update(kwargs)
        config.pop("backend", None)  # Remove as it's not a constructor param
        
        # Create instance
        instance = cls(**config)
        
        # Load prediction cache
        cache_file = p / "prediction_cache.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                instance._prediction_cache = json.load(f)
                
        return instance

    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self._prediction_cache.clear()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            return {
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,
                "cache_size": len(self._prediction_cache),
                "cache_memory_mb": sum(
                    len(str(v).encode()) for v in self._prediction_cache.values()
                ) / 1024**2
            }
        else:
            import psutil
            process = psutil.Process()
            return {
                "cpu_memory_usage": process.memory_info().rss / 1024**3,
                "cache_size": len(self._prediction_cache),
                "cache_memory_mb": sum(
                    len(str(v).encode()) for v in self._prediction_cache.values()
                ) / 1024**2
            }

    def __repr__(self) -> str:
        return (f"ReRanker(model_name='{self.model_name}', "
                f"backend='{self.backend}', device='{self.device}', "
                f"cached_predictions={len(self._prediction_cache)})") 