from __future__ import annotations

import json
from hashlib import blake2b
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import faiss
except ImportError:
    faiss = None

"""Advanced dense embedder with production-ready features.

This implementation supports multiple backends with enhanced capabilities:
1. **Sentence-Transformers**: (Default) High-quality semantic embeddings with GPU acceleration
2. **Hashing**: Deterministic backend for testing and CI environments
3. **Custom Models**: Support for custom transformer architectures

Features:
- Intelligent caching and persistence
- GPU/CPU optimization with automatic device selection
- Batch processing with memory management
- Progressive loading for large datasets
- Embedding quality assessment
- Multi-language support

Author: Nik Jois <nikjois@llamasearch.ai>
"""


class TextDataset(Dataset):
    """Custom dataset for batch processing of texts."""
    
    def __init__(self, texts: List[str]):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]


class DenseEmbedder(nn.Module):
    """Advanced multi-backend dense embedding model with production features.

    Args:
        model_name (str, optional):
            Model identifier. Can be Hugging Face model name, local path,
            or 'hashing-encoder' for deterministic hashing.
        pooling_strategy (str):
            Pooling strategy for embeddings ('mean', 'max', 'cls').
        normalize (bool):
            Whether to L2-normalize embeddings.
        max_length (int):
            Maximum sequence length for tokenization.
        device (str, optional):
            Target device ('cpu', 'cuda', 'auto').
        cache_dir (str, optional):
            Directory for caching embeddings.
        trust_remote_code (bool):
            Whether to trust remote code for custom models.
        precision (str):
            Precision for computations ('float32', 'float16', 'bfloat16').
        use_quantization (bool):
            Whether to use 8/4-bit quantization.
        quantization_config (dict, optional):
            Configuration for quantization.
        use_onnx (bool):
            Whether to use ONNX runtime.
        onnx_providers (list, optional):
            Execution providers for ONNX.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        pooling_strategy: str = "mean",
        normalize: bool = True,
        max_length: int = 512,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        precision: str = "float32",
        batch_size: int = 32,
        enable_caching: bool = True,
        use_quantization: bool = False,
        quantization_config: Optional[Dict] = None,
        use_onnx: bool = False,
        onnx_providers: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        
        # Configuration
        self.model_name = model_name or "all-MiniLM-L6-v2"
        self.pooling_strategy = pooling_strategy
        self.normalize = normalize
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code
        self.precision = precision
        self.default_batch_size = batch_size
        self.enable_caching = enable_caching
        self.use_quantization = use_quantization
        self.quantization_config = quantization_config or {}
        self.use_onnx = use_onnx
        self.onnx_providers = onnx_providers or ["CPUExecutionProvider"]
        
        # Device setup with intelligent selection
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
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".openembeddings_cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        # Model initialization
        self.backend: str
        self._sbert_model: Optional[SentenceTransformer] = None
        self._tokenizer = None
        self.ort_session = None  # ONNX runtime session
        
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """Initialize the embedding model based on configuration."""
        # ------------------------------------------------------------------
        # 1. Hashing encoder: fully offline, deterministic.
        # ------------------------------------------------------------------
        if self.model_name == "hashing-encoder":
            self.backend = "hash"
            self.dimension = 384  # Fixed dim for hashing backend
            self.projection = nn.Identity()
            # No other initialisation required
            return

        # ------------------------------------------------------------------
        # 2. Sentence-Transformers backend (default when available).
        # ------------------------------------------------------------------
        else:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is required for transformer models. "
                    "Install with: pip install sentence-transformers"
                )
            
            self.backend = "sbert"
            try:
                self._sbert_model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=self.trust_remote_code,
                    cache_folder=str(self.cache_dir / "models")
                )
                self.dimension = self._sbert_model.get_sentence_embedding_dimension()
                
                # Set precision
                if self.precision == "float16":
                    self._sbert_model = self._sbert_model.half()
                elif self.precision == "bfloat16" and torch.cuda.is_available():
                    self._sbert_model = self._sbert_model.bfloat16()
                    
            except Exception as e:
                # If we cannot download the model (e.g. offline), fall back to hashing backend.
                warnings.warn(
                    f"Falling back to hashing-encoder due to model load failure: {e}")
                self.backend = "hash"
                self.dimension = 384
                self.projection = nn.Identity()
                return

        # Additional initialisation
        if self.backend == "hash":
            # Nothing more to do
            return

        if self.use_onnx:
            self._initialize_onnx()
        # For SentenceTransformer backend, we can rely on the library's internal model
        # and do not need a separate AutoModel/Tokenizer. Only initialise raw transformers
        # stack when explicitly requested (future backend).
        elif self.backend not in {"sbert", "hash"}:
            self._initialize_pytorch()
        
    def _initialize_pytorch(self):
        from transformers import AutoModel, AutoTokenizer
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model with optional quantization
        if self.use_quantization:
            try:
                import bitsandbytes as bnb
                from bitsandbytes.nn import Linear8bitLt, Linear4bit
                
                # Set up quantization config
                quant_type = self.quantization_config.get("quant_type", "8bit")
                dtype = self.quantization_config.get("dtype", torch.float16)
                
                # Load model in quantized mode
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    load_in_8bit=quant_type == "8bit",
                    load_in_4bit=quant_type == "4bit",
                    torch_dtype=dtype
                )
            except ImportError:
                raise RuntimeError("bitsandbytes required for quantization")
        else:
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)

    def _initialize_onnx(self):
        """Initialize ONNX model session."""
        try:
            from onnxruntime import InferenceSession
        except ImportError:
            raise RuntimeError("ONNX runtime not installed")
            
        # Load tokenizer
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load ONNX model
        onnx_path = self.quantization_config.get("onnx_path")
        if not onnx_path:
            # TODO: Add logic to convert PyTorch model to ONNX if not provided
            raise ValueError("ONNX path required when using ONNX")
            
        self.ort_session = InferenceSession(
            onnx_path,
            providers=self.onnx_providers
        )
        
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for a text."""
        return blake2b(
            f"{self.model_name}:{self.normalize}:{text}".encode("utf-8"),
            digest_size=16
        ).hexdigest()

    def _load_from_cache(self, texts: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """Load embeddings from cache where available."""
        cached_embeddings = []
        uncached_texts = []
        
        for text in texts:
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                cached_embeddings.append(self._embedding_cache[cache_key])
            else:
                cached_embeddings.append(None)
                uncached_texts.append(text)
        
        return cached_embeddings, uncached_texts

    def _save_to_cache(self, texts: List[str], embeddings: np.ndarray) -> None:
        """Save embeddings to cache."""
        if not self.enable_caching:
            return
            
        for text, embedding in zip(texts, embeddings):
            cache_key = self._get_cache_key(text)
            self._embedding_cache[cache_key] = embedding

    @staticmethod
    def _text_to_vector(text: str, dim: int) -> np.ndarray:
        """Project text to deterministic vector on unit sphere."""
        h = blake2b(text.encode("utf-8"), digest_size=32).digest()
        seed = int.from_bytes(h[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return vec

    def forward(
        self, 
        texts: Union[str, List[str]], 
        return_attention: bool = False,
        batch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode texts into torch tensors with advanced features."""
        if isinstance(texts, str):
            texts = [texts]
            
        batch_size = batch_size or self.default_batch_size
        
        if self.backend == "sbert":
            assert self._sbert_model is not None
            
            # Process in batches for memory efficiency
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                with torch.no_grad():
                    embeddings = self._sbert_model.encode(
                        batch_texts,
                        convert_to_tensor=True,
                        normalize_embeddings=self.normalize,
                        device=self.device,
                        show_progress_bar=False
                    )
                all_embeddings.append(embeddings)
            
            embeddings = torch.cat(all_embeddings, dim=0)
            
        else:  # hash backend
            embeddings_np = np.stack([
                self._text_to_vector(t, self.dimension) for t in texts
            ])
            embeddings = torch.from_numpy(embeddings_np).to(self.device)
            
            if self.normalize:
                embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        # Attention weights placeholder
        attention_weights = None
        if return_attention:
            attention_weights = torch.zeros((len(texts), 1), device=self.device)

        return embeddings, attention_weights

    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        return_numpy: bool = True,
        use_cache: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """Enhanced encoding with caching and optimization."""
        if isinstance(texts, str):
            texts = [texts]
            
        batch_size = batch_size or self.default_batch_size
        
        # Try cache first
        if use_cache and self.enable_caching:
            cached_embeddings, uncached_texts = self._load_from_cache(texts)
            if not uncached_texts:  # All cached
                result = np.stack([emb for emb in cached_embeddings if emb is not None])
                return result if return_numpy else torch.from_numpy(result)
        else:
            uncached_texts = texts
            cached_embeddings = [None] * len(texts)

        # Encode uncached texts
        if uncached_texts:
            if self.backend == "sbert":
                assert self._sbert_model is not None
                
                uncached_embeddings = self._sbert_model.encode(
                    uncached_texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    normalize_embeddings=self.normalize,
                    convert_to_numpy=True
                )
                
                # Save to cache
                if use_cache:
                    self._save_to_cache(uncached_texts, uncached_embeddings)
                    
            else:  # hash backend
                uncached_embeddings = np.stack([
                    self._text_to_vector(t, self.dimension) for t in uncached_texts
                ])

        # Combine cached and newly computed embeddings
        if use_cache and any(emb is not None for emb in cached_embeddings):
            all_embeddings = []
            uncached_idx = 0
            
            for cached_emb in cached_embeddings:
                if cached_emb is not None:
                    all_embeddings.append(cached_emb)
                else:
                    all_embeddings.append(uncached_embeddings[uncached_idx])
                    uncached_idx += 1
                    
            result = np.stack(all_embeddings)
        else:
            result = uncached_embeddings

        return result if return_numpy else torch.from_numpy(result)

    def compute_similarity(
        self, 
        embeddings1: np.ndarray, 
        embeddings2: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """Compute similarity between embeddings."""
        if metric == "cosine":
            # Ensure normalized embeddings
            if not self.normalize:
                embeddings1 = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-9)
                embeddings2 = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-9)
            return np.dot(embeddings1, embeddings2.T)
        elif metric == "euclidean":
            return -np.linalg.norm(embeddings1[:, None] - embeddings2[None, :], axis=2)
        elif metric == "manhattan":
            return -np.sum(np.abs(embeddings1[:, None] - embeddings2[None, :]), axis=2)
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")

    def build_index(
        self, 
        embeddings: np.ndarray, 
        index_type: str = "flat",
        nlist: int = 100
    ) -> Optional[Any]:
        """Build FAISS index for fast similarity search."""
        if faiss is None:
            raise ImportError("FAISS is required for indexing. Install with: pip install faiss-cpu")
        
        embeddings = embeddings.astype(np.float32)
        dimension = embeddings.shape[1]
        
        if index_type == "flat":
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(embeddings)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 200
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        index.add(embeddings)
        return index

    def assess_embedding_quality(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Assess quality metrics for embeddings."""
        # Compute statistics
        mean_norm = np.mean(np.linalg.norm(embeddings, axis=1))
        std_norm = np.std(np.linalg.norm(embeddings, axis=1))
        
        # Compute average cosine similarity (should be low for good separation)
        similarities = np.dot(embeddings, embeddings.T)
        np.fill_diagonal(similarities, 0)  # Remove self-similarities
        avg_similarity = np.mean(similarities)
        
        # Compute effective dimension (participation ratio)
        S = np.cov(embeddings.T)
        eigenvals = np.linalg.eigvals(S)
        # Keep only the real component to avoid ComplexWarning due to numerical noise
        eigenvals = np.real(eigenvals)
        eigenvals = eigenvals[eigenvals > 1e-10]  # Remove near-zero eigenvalues
        participation_ratio = (np.sum(eigenvals) ** 2) / np.sum(eigenvals ** 2)
        
        return {
            "mean_norm": float(mean_norm),
            "std_norm": float(std_norm),
            "avg_cosine_similarity": float(avg_similarity),
            "effective_dimension": float(participation_ratio / len(eigenvals)),
            "intrinsic_dimension": float(participation_ratio)
        }

    def save_pretrained(self, save_path: str) -> None:
        """Save model configuration and state."""
        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            "model_name": self.model_name,
            "backend": self.backend,
            "normalize": self.normalize,
            "pooling_strategy": self.pooling_strategy,
            "max_length": self.max_length,
            "precision": self.precision,
            "dimension": self.dimension,
            "trust_remote_code": self.trust_remote_code,
            "use_quantization": self.use_quantization,
            "quantization_config": self.quantization_config,
            "use_onnx": self.use_onnx,
            "onnx_providers": self.onnx_providers,
        }
        
        with open(p / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Save sentence-transformer model
        if self.backend == "sbert" and self._sbert_model is not None:
            self._sbert_model.save(str(p / "sbert_model"))
            
        # Save embedding cache
        if self.enable_caching and self._embedding_cache:
            cache_file = p / "embedding_cache.json"
            cache_data = {k: v.tolist() for k, v in self._embedding_cache.items()}
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

    @classmethod
    def from_pretrained(cls, load_path: str, **kwargs) -> "DenseEmbedder":
        """Load model from saved configuration."""
        p = Path(load_path)
        
        # Load configuration
        with open(p / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        # Override with kwargs
        config.update(kwargs)

        # Handle model path for saved sentence-transformer
        if config["backend"] == "sbert":
            sbert_path = p / "sbert_model"
            if sbert_path.exists():
                config["model_name"] = str(sbert_path)

        # Remove fields that are not constructor parameters
        config.pop("backend", None)
        config.pop("dimension", None)  # This is computed during initialization
        
        # Create instance
        instance = cls(**config)
        
        # Load embedding cache
        cache_file = p / "embedding_cache.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
            instance._embedding_cache = {
                k: np.array(v) for k, v in cache_data.items()
            }
        
        return instance

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._embedding_cache.clear()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            return {
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,  # GB
                "cache_size": len(self._embedding_cache),
                "cache_memory_mb": sum(
                    emb.nbytes for emb in self._embedding_cache.values()
                ) / 1024**2
            }
        else:
            import psutil
            process = psutil.Process()
            return {
                "cpu_memory_usage": process.memory_info().rss / 1024**3,  # GB
                "cache_size": len(self._embedding_cache),
                "cache_memory_mb": sum(
                    emb.nbytes for emb in self._embedding_cache.values()
                ) / 1024**2
            }

    def __repr__(self) -> str:
        return (f"DenseEmbedder(model_name='{self.model_name}', "
                f"backend='{self.backend}', dimension={self.dimension}, "
                f"device='{self.device}', normalize={self.normalize})")

    def embed(
        self,
        texts: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """Generate embeddings for input texts."""
        if self.use_onnx:
            return self._embed_onnx(texts, batch_size)
        else:
            return self._embed_pytorch(texts, batch_size)
            
    def _embed_onnx(
        self,
        texts: List[str],
        batch_size: int
    ) -> np.ndarray:
        """Generate embeddings using ONNX runtime."""
        import onnxruntime
        
        # Tokenize in batches
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np"
            )
            
            # Run ONNX inference
            ort_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
            ort_outputs = self.ort_session.run(None, ort_inputs)
            
            # Extract embeddings (assume last hidden state)
            last_hidden_state = ort_outputs[0]
            embeddings = self._mean_pooling(last_hidden_state, inputs["attention_mask"])
            all_embeddings.append(embeddings)
            
        return np.vstack(all_embeddings)

    def _embed_pytorch(
        self,
        texts: List[str],
        batch_size: int
    ) -> np.ndarray:
        """Generate embeddings using the native PyTorch/transformers backend.

        This method supports both the `sentence-transformers` backend (self.backend == 'sbert')
        and a raw transformers model loaded via ``AutoModel``. The returned embeddings are
        always 2-D ``np.ndarray`` with shape ``(len(texts), hidden_size)``.
        """

        if self.backend == "sbert":
            # Fast path via SentenceTransformer which already handles batching internally.
            assert self._sbert_model is not None, "Sentence-Transformer model not initialised"

            embeddings = self._sbert_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
            )

            return embeddings.astype(np.float32)

        # Fallback: raw transformers model
        assert self._model is not None and self._tokenizer is not None, "Transformer model/tokenizer not initialised"

        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenise
            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self._model(**inputs)

                if hasattr(outputs, "last_hidden_state"):
                    last_hidden_state = outputs.last_hidden_state
                else:
                    # For models that return tuple
                    last_hidden_state = outputs[0]

                emb = self._mean_pooling(last_hidden_state, inputs["attention_mask"])

                # Convert to CPU/NumPy
                all_embeddings.append(emb.cpu().float().numpy())

        embeddings_np = np.vstack(all_embeddings)

        if self.normalize:
            embeddings_np = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-9)

        return embeddings_np

    def _mean_pooling(self, last_hidden_state, attention_mask):
        """Apply mean pooling over the token embeddings.

        The implementation supports both ``torch.Tensor`` and ``np.ndarray`` inputs so it can
        be shared by the PyTorch and ONNX inference paths.
        """

        # Torch pathway
        if isinstance(last_hidden_state, torch.Tensor):
            # Expand the attention mask so it can be broadcasted with last_hidden_state
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
            masked_embeddings = last_hidden_state * mask
            sum_embeddings = masked_embeddings.sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask

        # Numpy (ONNX) pathway
        elif isinstance(last_hidden_state, np.ndarray):
            mask = attention_mask[..., None].astype(last_hidden_state.dtype)
            masked_embeddings = last_hidden_state * mask
            sum_embeddings = masked_embeddings.sum(axis=1)
            sum_mask = mask.sum(axis=1)
            sum_mask[sum_mask == 0] = 1e-9
            return sum_embeddings / sum_mask

        else:
            raise TypeError("Unsupported tensor type for mean pooling")
