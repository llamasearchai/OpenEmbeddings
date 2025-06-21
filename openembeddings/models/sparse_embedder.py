from typing import List, Sequence, Optional, Dict, Any, Union
import json
import warnings
from pathlib import Path

from rank_bm25 import BM25Okapi, BM25L, BM25Plus
import re

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    from nltk.stem.snowball import SnowballStemmer
    _NLTK_AVAILABLE = True
except ImportError:
    nltk = None
    stopwords = None
    PorterStemmer = None
    word_tokenize = None
    SnowballStemmer = None
    _NLTK_AVAILABLE = False

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    _SPACY_AVAILABLE = False

"""Advanced sparse (lexical) embedding model with production features.

This implementation provides comprehensive BM25-based retrieval with:
- Multiple BM25 variants (Okapi, L, Plus)
- Advanced tokenization and preprocessing
- Multi-language support
- Custom stopword management
- Statistical analysis and optimization
- Caching and persistence

Author: Nik Jois <nikjois@llamasearch.ai>
"""

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class SparseEmbedder:
    """Advanced BM25-based sparse retriever with production features."""

    def __init__(
        self, 
        model_type: str = "bm25",
        bm25_variant: str = "okapi",
        language: str = "english",
        use_stemming: bool = True,
        use_stopwords: bool = True,
        custom_stopwords: Optional[List[str]] = None,
        tokenizer: str = "nltk",
        k1: float = 1.2,
        b: float = 0.75,
        epsilon: float = 0.25,
        delta: float = 1.0,
        enable_caching: bool = True,
        cache_dir: Optional[str] = None
    ):
        """Initialize enhanced sparse embedder.
        
        Args:
            model_type: Type of sparse model ('bm25')
            bm25_variant: BM25 variant ('okapi', 'l', 'plus')
            language: Language for stopwords and stemming
            use_stemming: Whether to apply stemming
            use_stopwords: Whether to remove stopwords
            custom_stopwords: Additional stopwords to use
            tokenizer: Tokenizer to use ('nltk', 'spacy', 'simple')
            k1: BM25 parameter k1 (term frequency saturation)
            b: BM25 parameter b (document length normalization)
            epsilon: BM25L parameter epsilon
            delta: BM25Plus parameter delta
            enable_caching: Whether to cache tokenized documents
            cache_dir: Directory for caching
        """
        if model_type.lower() != "bm25":
            raise ValueError("Currently only the 'bm25' model_type is supported")
        
        self.model_type = model_type.lower()
        self.bm25_variant = bm25_variant.lower()
        self.language = language
        self.use_stemming = use_stemming
        self.use_stopwords = use_stopwords
        self.tokenizer_type = tokenizer.lower()
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.delta = delta
        self.enable_caching = enable_caching
        
        # Cache setup
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".openembeddings_cache" / "sparse"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self._token_cache: Dict[str, List[str]] = {}
        
        # Model state
        self._bm25: Optional[Union[BM25Okapi, BM25L, BM25Plus]] = None
        self._docs: List[str] = []
        self._tokenized_docs: List[List[str]] = []
        self._vocab: Dict[str, int] = {}
        self._idf_values: Dict[str, float] = {}
        
        # Initialize components
        self._initialize_tokenizer()
        self._initialize_stopwords(custom_stopwords)
        self._initialize_stemmer()

    def _initialize_tokenizer(self) -> None:
        """Initialize tokenizer based on configuration."""
        if self.tokenizer_type == "nltk":
            if not _NLTK_AVAILABLE:
                warnings.warn("NLTK not available. Falling back to simple tokenizer.")
                self.tokenizer_type = "simple"
            else:
                try:
                    # Download required data if not present
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    warnings.warn("NLTK punkt tokenizer not found. Please run: python -m nltk.downloader punkt")
                    self.tokenizer_type = "simple"
        
        elif self.tokenizer_type == "spacy":
            if not _SPACY_AVAILABLE:
                warnings.warn("spaCy not available. Falling back to NLTK.")
                self.tokenizer_type = "nltk" if _NLTK_AVAILABLE else "simple"
            else:
                try:
                    # Try to load language model
                    self.nlp = spacy.load(f"{self.language[:2]}_core_web_sm")
                except OSError:
                    try:
                        self.nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        warnings.warn("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                        self.tokenizer_type = "nltk" if _NLTK_AVAILABLE else "simple"

    def _initialize_stopwords(self, custom_stopwords: Optional[List[str]]) -> None:
        """Initialize stopwords."""
        self._stopwords = set()
        
        if self.use_stopwords and _NLTK_AVAILABLE:
            try:
                self._stopwords = set(stopwords.words(self.language))
            except LookupError:
                warnings.warn(f"NLTK stopwords for '{self.language}' not found. Please run: python -m nltk.downloader stopwords")
            except OSError:
                # Language not supported, use English as fallback
                try:
                    self._stopwords = set(stopwords.words('english'))
                except LookupError:
                    warnings.warn("NLTK stopwords not available.")
        
        # Add custom stopwords
        if custom_stopwords:
            self._stopwords.update(custom_stopwords)

    def _initialize_stemmer(self) -> None:
        """Initialize stemmer based on language."""
        self._stemmer = None
        
        if self.use_stemming and _NLTK_AVAILABLE:
            try:
                # Use Snowball stemmer for better multi-language support
                if self.language in SnowballStemmer.languages:
                    self._stemmer = SnowballStemmer(self.language)
                else:
                    # Fallback to Porter stemmer
                    self._stemmer = PorterStemmer()
            except Exception:
                warnings.warn("Failed to initialize stemmer.")

    def _tokenize(self, text: str) -> List[str]:
        """Advanced tokenization with caching."""
        # Check cache first
        if self.enable_caching and text in self._token_cache:
            return self._token_cache[text]
        
        tokens = []
        
        if self.tokenizer_type == "spacy" and hasattr(self, 'nlp'):
            doc = self.nlp(text)
            tokens = [
                token.lemma_.lower() if self.use_stemming else token.text.lower()
                for token in doc
                if token.is_alpha and not token.is_space
            ]
            
        elif self.tokenizer_type == "nltk" and _NLTK_AVAILABLE:
            try:
                words = word_tokenize(text.lower())
                tokens = [
                    word for word in words
                    if word.isalnum() and len(word) > 1
                ]
            except LookupError:
                # Fallback to simple tokenization
                tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
                
        else:  # Simple tokenization
            tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
            
        # Apply stopword removal
        if self.use_stopwords and self._stopwords:
            tokens = [token for token in tokens if token not in self._stopwords]
        
        # Apply stemming
        if self.use_stemming and self._stemmer:
            tokens = [self._stemmer.stem(token) for token in tokens]
        elif self.use_stemming and not self._stemmer:
            # Simple stemming fallback
            processed = []
            for token in tokens:
                if token.endswith("es"):
                    token = token[:-2]
                elif token.endswith("s") and len(token) > 3:
                    token = token[:-1]
                processed.append(token)
            tokens = processed
        
        # Cache result
        if self.enable_caching:
            self._token_cache[text] = tokens
            
        return tokens

    def fit(self, documents: Sequence[str]) -> None:
        """Enhanced tokenization and indexing with vocabulary building."""
        self._docs = list(documents)
        if not self._docs:
            return
            
        # Tokenize all documents
        self._tokenized_docs = [self._tokenize(doc) for doc in self._docs]
        
        # Build vocabulary
        self._vocab = {}
        for tokens in self._tokenized_docs:
            for token in set(tokens):  # Use set to avoid counting duplicates per doc
                self._vocab[token] = self._vocab.get(token, 0) + 1
        
        # Initialize BM25 model based on variant
        if self.bm25_variant == "okapi":
            self._bm25 = BM25Okapi(self._tokenized_docs, k1=self.k1, b=self.b)
        elif self.bm25_variant == "l":
            self._bm25 = BM25L(self._tokenized_docs, k1=self.k1, b=self.b, delta=self.epsilon)
        elif self.bm25_variant == "plus":
            self._bm25 = BM25Plus(self._tokenized_docs, k1=self.k1, b=self.b, delta=self.delta)
        else:
            raise ValueError(f"Unsupported BM25 variant: {self.bm25_variant}")
            
        # Calculate IDF values for analysis
        self._calculate_idf_values()

    def _calculate_idf_values(self) -> None:
        """Calculate IDF values for vocabulary analysis."""
        if not self._bm25:
            return
            
        total_docs = len(self._docs)
        self._idf_values = {}
        
        for token, doc_freq in self._vocab.items():
            # Use same IDF calculation as BM25
            idf = self._bm25.idf.get(token, 0)
            self._idf_values[token] = idf

    def compute_scores(
        self, 
        queries: Sequence[str], 
        documents: Optional[Sequence[str]] = None
    ) -> List[List[float]]:
        """Enhanced BM25 scoring with document re-indexing support."""
        # Re-index if necessary
        if documents is not None:
            if list(documents) != self._docs:
                self.fit(documents)
        elif self._bm25 is None:
            raise ValueError(
                "BM25 model is not initialized. Call 'fit' first or provide 'documents'."
            )

        assert self._bm25 is not None

        scores: List[List[float]] = []
        for query in queries:
            tokenized_query = self._tokenize(query)
            query_scores = self._bm25.get_scores(tokenized_query).tolist()
            scores.append(query_scores)
            
        return scores

    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Get comprehensive vocabulary statistics."""
        if not self._vocab:
            return {}
            
        doc_lengths = [len(tokens) for tokens in self._tokenized_docs]
        vocab_size = len(self._vocab)
        total_tokens = sum(len(tokens) for tokens in self._tokenized_docs)
        unique_tokens = len(set(token for tokens in self._tokenized_docs for token in tokens))
        
        # Most/least frequent terms
        sorted_vocab = sorted(self._vocab.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "vocabulary_size": vocab_size,
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "avg_doc_length": sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0,
            "min_doc_length": min(doc_lengths) if doc_lengths else 0,
            "max_doc_length": max(doc_lengths) if doc_lengths else 0,
            "most_frequent_terms": sorted_vocab[:10],
            "least_frequent_terms": sorted_vocab[-10:] if len(sorted_vocab) > 10 else [],
            "type_token_ratio": unique_tokens / total_tokens if total_tokens > 0 else 0,
        }

    def get_query_analysis(self, query: str) -> Dict[str, Any]:
        """Analyze query terms and their importance."""
        tokenized_query = self._tokenize(query)
        
        analysis = {
            "original_query": query,
            "tokenized_query": tokenized_query,
            "query_length": len(tokenized_query),
            "unique_terms": len(set(tokenized_query)),
            "term_analysis": []
        }
        
        for term in set(tokenized_query):
            term_info = {
                "term": term,
                "frequency_in_query": tokenized_query.count(term),
                "document_frequency": self._vocab.get(term, 0),
                "idf_value": self._idf_values.get(term, 0),
                "in_vocabulary": term in self._vocab
            }
            analysis["term_analysis"].append(term_info)
            
        # Sort by IDF (importance)
        analysis["term_analysis"].sort(key=lambda x: x["idf_value"], reverse=True)
        
        return analysis

    def optimize_parameters(self, queries: List[str], relevance_judgments: Dict[str, List[int]]) -> Dict[str, float]:
        """Optimize BM25 parameters using grid search."""
        if not queries or not relevance_judgments:
            raise ValueError("Queries and relevance judgments are required for optimization")
            
        best_params = {"k1": self.k1, "b": self.b}
        best_score = 0.0
        
        # Parameter grid
        k1_values = [0.5, 1.0, 1.2, 1.5, 2.0]
        b_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for k1 in k1_values:
            for b in b_values:
                # Create temporary BM25 model
                if self.bm25_variant == "okapi":
                    temp_bm25 = BM25Okapi(self._tokenized_docs, k1=k1, b=b)
                elif self.bm25_variant == "l":
                    temp_bm25 = BM25L(self._tokenized_docs, k1=k1, b=b, delta=self.epsilon)
                else:  # plus
                    temp_bm25 = BM25Plus(self._tokenized_docs, k1=k1, b=b, delta=self.delta)
                
                # Evaluate
                total_score = 0.0
                for query in queries:
                    tokenized_query = self._tokenize(query)
                    scores = temp_bm25.get_scores(tokenized_query)
                    
                    if query in relevance_judgments:
                        relevant_docs = relevance_judgments[query]
                        # Simple evaluation: average score of relevant documents
                        if relevant_docs:
                            avg_relevant_score = sum(scores[i] for i in relevant_docs if i < len(scores)) / len(relevant_docs)
                            total_score += avg_relevant_score
                
                avg_score = total_score / len(queries)
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {"k1": k1, "b": b}
        
        return best_params

    def save_pretrained(self, save_path: str) -> None:
        """Save model configuration and state."""
        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            "model_type": self.model_type,
            "bm25_variant": self.bm25_variant,
            "language": self.language,
            "use_stemming": self.use_stemming,
            "use_stopwords": self.use_stopwords,
            "tokenizer_type": self.tokenizer_type,
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "delta": self.delta,
        }
        
        with open(p / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            
        # Save documents and tokenized documents
        if self._docs:
            with open(p / "documents.json", "w", encoding="utf-8") as f:
                json.dump(self._docs, f, indent=2)
                
        if self._tokenized_docs:
            with open(p / "tokenized_docs.json", "w", encoding="utf-8") as f:
                json.dump(self._tokenized_docs, f, indent=2)
                
        # Save vocabulary and IDF values
        if self._vocab:
            with open(p / "vocabulary.json", "w", encoding="utf-8") as f:
                json.dump(self._vocab, f, indent=2)
                
        if self._idf_values:
            with open(p / "idf_values.json", "w", encoding="utf-8") as f:
                json.dump(self._idf_values, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_path: str, **kwargs) -> "SparseEmbedder":
        """Load model from saved configuration."""
        p = Path(load_path)
        
        # Load configuration
        with open(p / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            
        # Override with kwargs
        config.update(kwargs)
        
        # Create instance
        instance = cls(**config)
        
        # Load documents if available
        docs_file = p / "documents.json"
        if docs_file.exists():
            with open(docs_file, "r", encoding="utf-8") as f:
                documents = json.load(f)
            instance.fit(documents)
        
        return instance

    def clear_cache(self) -> None:
        """Clear tokenization cache."""
        self._token_cache.clear()

    def __repr__(self) -> str:
        return (f"SparseEmbedder(model_type='{self.model_type}', "
                f"variant='{self.bm25_variant}', language='{self.language}', "
                f"tokenizer='{self.tokenizer_type}', fitted={self._bm25 is not None})")
