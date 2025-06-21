"""FastAPI Web Service for OpenEmbeddings

A production-ready REST API for serving embedding and retrieval models.
Includes authentication, rate limiting, monitoring, and comprehensive documentation.

Features:
- RESTful API endpoints for all core functionality
- Interactive API documentation with Swagger UI
- Authentication and API key management
- Rate limiting and request validation
- Performance monitoring and logging
- Async processing for high throughput
- Docker-ready deployment
- Health checks and metrics

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import os
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

from .models.hybrid_retriever import HybridRetriever
from .models.dense_embedder import DenseEmbedder
from .models.sparse_embedder import SparseEmbedder
from .models.reranker import ReRanker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model storage
models: Dict[str, Any] = {}
request_counts: Dict[str, int] = {}
rate_limits: Dict[str, List[float]] = {}

# Pydantic Models for API
class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed", max_items=100)
    model_name: Optional[str] = Field("default", description="Model name to use")
    normalize: bool = Field(True, description="Whether to normalize embeddings")
    batch_size: Optional[int] = Field(32, description="Batch size for processing")

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    processing_time: float = Field(..., description="Processing time in seconds")

class IndexRequest(BaseModel):
    documents: List[str] = Field(..., description="Documents to index", max_items=10000)
    index_name: str = Field(..., description="Name for the index")
    dense_model: Optional[str] = Field("all-MiniLM-L6-v2", description="Dense model name")
    fusion_strategy: Optional[str] = Field("rrf", description="Fusion strategy")
    use_ann: bool = Field(True, description="Use approximate nearest neighbor")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    index_name: str = Field(..., description="Index to search")
    top_k: int = Field(10, description="Number of results to return", ge=1, le=100)
    use_reranker: bool = Field(False, description="Use cross-encoder re-ranking")
    reranker_model: Optional[str] = Field("cross-encoder/ms-marco-MiniLM-L-6-v2", description="Re-ranker model")
    score_threshold: Optional[float] = Field(None, description="Minimum score threshold")

class SearchResult(BaseModel):
    document_id: int = Field(..., description="Document ID")
    score: float = Field(..., description="Relevance score")
    document: str = Field(..., description="Document text")

class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(..., description="Search results")
    query_info: Dict[str, Any] = Field(..., description="Query processing information")
    processing_time: float = Field(..., description="Processing time in seconds")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: List[str] = Field(..., description="Currently loaded models")
    uptime: float = Field(..., description="Service uptime in seconds")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage statistics")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Error details")
    timestamp: str = Field(..., description="Error timestamp")

# Authentication
security = HTTPBearer() if _FASTAPI_AVAILABLE else None

API_KEYS = {
    "demo_key": {"name": "Demo User", "rate_limit": 100},
    "admin_key": {"name": "Admin User", "rate_limit": 1000}
}

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication."""
    if not credentials or credentials.credentials not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Rate limiting
def check_rate_limit(api_key: str, limit: int = 100, window: int = 60):
    """Check if request exceeds rate limit."""
    current_time = time.time()
    
    if api_key not in rate_limits:
        rate_limits[api_key] = []
    
    # Remove old requests outside the window
    rate_limits[api_key] = [
        req_time for req_time in rate_limits[api_key] 
        if current_time - req_time < window
    ]
    
    # Check if limit exceeded
    if len(rate_limits[api_key]) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit} requests per {window} seconds"
        )
    
    # Add current request
    rate_limits[api_key].append(current_time)

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting OpenEmbeddings API...")
    
    # Load default models
    try:
        models["default_dense"] = DenseEmbedder()
        models["default_sparse"] = SparseEmbedder()
        models["default_reranker"] = ReRanker()
        logger.info("Default models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load default models: {e}")
    
    # Setup Redis if available
    if _REDIS_AVAILABLE:
        try:
            redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=0
            )
            models["redis"] = redis_client
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down OpenEmbeddings API...")

# Create FastAPI app
if _FASTAPI_AVAILABLE:
    app = FastAPI(
        title="OpenEmbeddings API",
        description="Production-ready REST API for embedding and retrieval services",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
    )
    
    # Store startup time
    startup_time = datetime.now()
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                details=str(exc),
                timestamp=datetime.now().isoformat()
            ).dict()
        )
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        uptime = (datetime.now() - startup_time).total_seconds()
        
        # Get memory usage
        memory_usage = {}
        if "default_dense" in models:
            try:
                memory_usage = models["default_dense"].get_memory_usage()
            except Exception:
                memory_usage = {"error": "Could not get memory usage"}
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            models_loaded=list(models.keys()),
            uptime=uptime,
            memory_usage=memory_usage
        )
    
    # Embedding endpoints
    @app.post("/embed", response_model=EmbeddingResponse)
    async def create_embeddings(
        request: EmbeddingRequest,
        api_key: str = Depends(verify_api_key)
    ):
        """Generate embeddings for input texts."""
        # Rate limiting
        user_info = API_KEYS[api_key]
        check_rate_limit(api_key, user_info["rate_limit"])
        
        start_time = time.time()
        
        try:
            # Get model
            model_name = request.model_name or "default_dense"
            if model_name not in models:
                if model_name == "default_dense":
                    models[model_name] = DenseEmbedder(normalize=request.normalize)
                else:
                    models[model_name] = DenseEmbedder(
                        model_name=model_name,
                        normalize=request.normalize
                    )
            
            embedder = models[model_name]
            
            # Generate embeddings
            embeddings = embedder.encode(
                request.texts,
                batch_size=request.batch_size,
                show_progress=False
            )
            
            processing_time = time.time() - start_time
            
            return EmbeddingResponse(
                embeddings=embeddings.tolist(),
                model_info={
                    "model_name": embedder.model_name,
                    "dimension": embedder.dimension,
                    "backend": embedder.backend
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embeddings: {str(e)}"
            )
    
    # Index management endpoints
    @app.post("/index")
    async def create_index(
        request: IndexRequest,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(verify_api_key)
    ):
        """Create a new search index."""
        # Rate limiting
        user_info = API_KEYS[api_key]
        check_rate_limit(api_key, user_info["rate_limit"])
        
        def build_index():
            try:
                retriever = HybridRetriever(
                    dense_model=request.dense_model,
                    fusion_strategy=request.fusion_strategy,
                    use_ann=request.use_ann
                )
                
                retriever.index(request.documents, show_progress=False)
                models[f"index_{request.index_name}"] = retriever
                
                logger.info(f"Index '{request.index_name}' created successfully")
                
            except Exception as e:
                logger.error(f"Index creation failed: {e}")
        
        # Build index in background
        background_tasks.add_task(build_index)
        
        return {
            "message": f"Index '{request.index_name}' creation started",
            "status": "processing",
            "documents_count": len(request.documents)
        }
    
    @app.get("/index/{index_name}")
    async def get_index_info(
        index_name: str,
        api_key: str = Depends(verify_api_key)
    ):
        """Get information about an index."""
        index_key = f"index_{index_name}"
        
        if index_key not in models:
            raise HTTPException(
                status_code=404,
                detail=f"Index '{index_name}' not found"
            )
        
        retriever = models[index_key]
        
        return {
            "index_name": index_name,
            "document_count": len(retriever.corpus),
            "fusion_strategy": retriever.fusion_strategy,
            "dimension": retriever.dimension,
            "is_indexed": retriever.is_indexed
        }
    
    @app.delete("/index/{index_name}")
    async def delete_index(
        index_name: str,
        api_key: str = Depends(verify_api_key)
    ):
        """Delete an index."""
        index_key = f"index_{index_name}"
        
        if index_key not in models:
            raise HTTPException(
                status_code=404,
                detail=f"Index '{index_name}' not found"
            )
        
        del models[index_key]
        
        return {
            "message": f"Index '{index_name}' deleted successfully"
        }
    
    # Search endpoints
    @app.post("/search", response_model=SearchResponse)
    async def search_documents(
        request: SearchRequest,
        api_key: str = Depends(verify_api_key)
    ):
        """Search documents in an index."""
        # Rate limiting
        user_info = API_KEYS[api_key]
        check_rate_limit(api_key, user_info["rate_limit"])
        
        start_time = time.time()
        
        # Get index
        index_key = f"index_{request.index_name}"
        if index_key not in models:
            raise HTTPException(
                status_code=404,
                detail=f"Index '{request.index_name}' not found"
            )
        
        try:
            retriever = models[index_key]
            
            # Search
            results = retriever.retrieve(
                request.query,
                top_k=request.top_k,
                score_threshold=request.score_threshold
            )
            
            # Re-rank if requested
            if request.use_reranker:
                reranker_key = f"reranker_{request.reranker_model}"
                if reranker_key not in models:
                    models[reranker_key] = ReRanker(model_name=request.reranker_model)
                
                reranker = models[reranker_key]
                results = reranker.rerank(request.query, results, top_k=request.top_k)
            
            processing_time = time.time() - start_time
            
            # Format results
            search_results = [
                SearchResult(
                    document_id=doc_id,
                    score=score,
                    document=document
                )
                for doc_id, score, document in results
            ]
            
            return SearchResponse(
                results=search_results,
                query_info={
                    "query": request.query,
                    "index_name": request.index_name,
                    "total_documents": len(retriever.corpus),
                    "fusion_strategy": retriever.fusion_strategy,
                    "reranked": request.use_reranker
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Search failed: {str(e)}"
            )
    
    # Analytics endpoints
    @app.get("/analytics/usage")
    async def get_usage_analytics(api_key: str = Depends(verify_api_key)):
        """Get usage analytics."""
        return {
            "total_requests": sum(request_counts.values()),
            "requests_by_endpoint": request_counts,
            "active_models": len(models),
            "rate_limit_status": {
                key: len(requests) 
                for key, requests in rate_limits.items()
            }
        }
    
    @app.get("/models")
    async def list_models(api_key: str = Depends(verify_api_key)):
        """List all loaded models."""
        model_info = {}
        
        for key, model in models.items():
            if hasattr(model, '__class__'):
                model_info[key] = {
                    "type": model.__class__.__name__,
                    "loaded": True
                }
                
                # Add specific info based on model type
                if hasattr(model, 'model_name'):
                    model_info[key]["model_name"] = model.model_name
                if hasattr(model, 'dimension'):
                    model_info[key]["dimension"] = model.dimension
                if hasattr(model, 'backend'):
                    model_info[key]["backend"] = model.backend
        
        return model_info

else:
    # Create dummy app if FastAPI not available
    app = None
    logger.warning("FastAPI not available. Web API disabled.")

# Utility functions for running the server
def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
    log_level: str = "info"
):
    """Run the FastAPI server."""
    if not _FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required to run the web server. Install with: pip install openembeddings[web]")
    
    uvicorn.run(
        "openembeddings.web_api:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level=log_level
    )

def create_client(base_url: str = "http://localhost:8000", api_key: str = "demo_key"):
    """Create an API client for the OpenEmbeddings service."""
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required for the API client. Install with: pip install httpx")
    
    class OpenEmbeddingsClient:
        def __init__(self, base_url: str, api_key: str):
            self.base_url = base_url.rstrip("/")
            self.headers = {"Authorization": f"Bearer {api_key}"}
            self.client = httpx.Client(base_url=self.base_url, headers=self.headers)
        
        def embed(self, texts: List[str], **kwargs) -> Dict[str, Any]:
            """Generate embeddings."""
            response = self.client.post("/embed", json={"texts": texts, **kwargs})
            response.raise_for_status()
            return response.json()
        
        def create_index(self, index_name: str, documents: List[str], **kwargs) -> Dict[str, Any]:
            """Create search index."""
            response = self.client.post("/index", json={
                "index_name": index_name,
                "documents": documents,
                **kwargs
            })
            response.raise_for_status()
            return response.json()
        
        def search(self, query: str, index_name: str, **kwargs) -> Dict[str, Any]:
            """Search documents."""
            response = self.client.post("/search", json={
                "query": query,
                "index_name": index_name,
                **kwargs
            })
            response.raise_for_status()
            return response.json()
        
        def health(self) -> Dict[str, Any]:
            """Check service health."""
            response = self.client.get("/health")
            response.raise_for_status()
            return response.json()
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.client.close()
    
    return OpenEmbeddingsClient(base_url, api_key)

if __name__ == "__main__":
    run_server() 