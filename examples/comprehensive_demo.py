#!/usr/bin/env python3
"""Comprehensive demonstration of OpenEmbeddings capabilities.

This script demonstrates:
1. Dense and sparse embeddings
2. Hybrid retrieval with different fusion strategies
3. Re-ranking capabilities
4. Performance profiling
5. Benchmarking on research datasets
6. Visualization and analysis

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

# Add the parent directory to the path to import openembeddings
sys.path.insert(0, str(Path(__file__).parent.parent))

from openembeddings.models.dense_embedder import DenseEmbedder
from openembeddings.models.sparse_embedder import SparseEmbedder
from openembeddings.models.hybrid_retriever import HybridRetriever
from openembeddings.models.reranker import ReRanker
from openembeddings.benchmarks import BenchmarkSuite, DatasetManager

console = Console()

def demo_embedding_models():
    """Demonstrate dense and sparse embedding capabilities."""
    console.print(Panel("Dense and Sparse Embedding Demo", style="bold blue"))
    
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret visual information.",
        "Reinforcement learning trains agents through trial and error.",
        "Supervised learning uses labeled data for training models.",
        "Unsupervised learning finds patterns in unlabeled data.",
        "Transfer learning leverages pre-trained models for new tasks."
    ]
    
    # Dense embeddings
    console.print("\n[bold green]Dense Embeddings:[/]")
    dense_embedder = DenseEmbedder(model_name="all-MiniLM-L6-v2")
    dense_embeddings = dense_embedder.encode(documents)
    console.print(f"Dense embedding shape: {dense_embeddings.shape}")
    console.print(f"Dense embedding dimension: {dense_embeddings.shape[1]}")
    
    # Sparse embeddings
    console.print("\n[bold green]Sparse Embeddings:[/]")
    sparse_embedder = SparseEmbedder()
    sparse_embedder.fit(documents)
    # Compute scores using a sample query to demonstrate functionality
    sample_query = "machine learning artificial intelligence"
    sparse_scores = sparse_embedder.compute_scores([sample_query], documents)
    console.print(f"Sparse scores shape: {len(sparse_scores)} queries x {len(sparse_scores[0])} documents")
    console.print(f"Sample query: '{sample_query}'")
    console.print(f"Top 3 scores: {sorted(sparse_scores[0], reverse=True)[:3]}")
    
    return documents, dense_embeddings, sparse_scores

def demo_hybrid_retrieval(documents):
    """Demonstrate hybrid retrieval with different fusion strategies."""
    console.print(Panel("Hybrid Retrieval Demo", style="bold magenta"))
    
    # Test queries
    queries = [
        "artificial intelligence and machine learning",
        "neural networks for deep learning",
        "computer vision and image processing"
    ]
    
    fusion_strategies = ["rrf", "linear"]
    
    for strategy in fusion_strategies:
        console.print(f"\n[bold cyan]Fusion Strategy: {strategy.upper()}[/]")
        
        # Create hybrid retriever
        retriever = HybridRetriever(
            dense_model="all-MiniLM-L6-v2",
            fusion_strategy=strategy,
            use_ann=True
        )
        
        # Index documents
        retriever.index(documents, show_progress=True)
        
        # Test retrieval
        for query in queries:
            results = retriever.retrieve(query, top_k=3)
            
            table = Table(title=f"Results for: '{query}' ({strategy})")
            table.add_column("Rank", style="magenta")
            table.add_column("Score", style="green")
            table.add_column("Document")
            
            for i, (idx, score, doc) in enumerate(results, 1):
                table.add_row(str(i), f"{score:.4f}", doc[:60] + "...")
            
            console.print(table)

def demo_reranking(documents):
    """Demonstrate cross-encoder re-ranking."""
    console.print(Panel("Re-ranking Demo", style="bold yellow"))
    
    # Create retriever and reranker
    retriever = HybridRetriever(dense_model="all-MiniLM-L6-v2")
    retriever.index(documents, show_progress=True)
    
    reranker = ReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    query = "machine learning and artificial intelligence techniques"
    
    # Get initial results
    initial_results = retriever.retrieve(query, top_k=5)
    console.print("\n[bold green]Before Re-ranking:[/]")
    
    table = Table()
    table.add_column("Rank", style="magenta")
    table.add_column("Score", style="green")
    table.add_column("Document")
    
    for i, (idx, score, doc) in enumerate(initial_results, 1):
        table.add_row(str(i), f"{score:.4f}", doc[:60] + "...")
    
    console.print(table)
    
    # Re-rank results
    reranked_results = reranker.rerank(query, initial_results)
    console.print("\n[bold green]After Re-ranking:[/]")
    
    table = Table()
    table.add_column("Rank", style="magenta")
    table.add_column("Score", style="green")
    table.add_column("Document")
    
    for i, (idx, score, doc) in enumerate(reranked_results, 1):
        table.add_row(str(i), f"{score:.4f}", doc[:60] + "...")
    
    console.print(table)

def demo_performance_profiling():
    """Demonstrate performance profiling capabilities."""
    console.print(Panel("Performance Profiling Demo", style="bold red"))
    
    # Create sample datasets of different sizes
    base_docs = [
        "Machine learning algorithms learn patterns from data.",
        "Deep neural networks have multiple hidden layers.",
        "Natural language processing analyzes human language.",
        "Computer vision processes visual information.",
        "Reinforcement learning learns through rewards."
    ]
    
    corpus_sizes = [10, 50, 100]
    performance_data = []
    
    for size in corpus_sizes:
        console.print(f"\n[bold cyan]Testing with {size} documents...[/]")
        
        # Generate corpus
        documents = (base_docs * ((size // len(base_docs)) + 1))[:size]
        
        # Create retriever
        retriever = HybridRetriever(dense_model="all-MiniLM-L6-v2")
        
        # Time indexing
        import time
        start_time = time.time()
        retriever.index(documents, show_progress=False)
        index_time = time.time() - start_time
        
        # Time retrieval
        query = "machine learning and neural networks"
        start_time = time.time()
        results = retriever.retrieve(query, top_k=5)
        retrieval_time = time.time() - start_time
        
        performance_data.append({
            'corpus_size': size,
            'index_time': index_time,
            'retrieval_time': retrieval_time
        })
        
        console.print(f"  Indexing time: {index_time:.3f}s")
        console.print(f"  Retrieval time: {retrieval_time:.3f}s")
    
    # Create performance visualization
    if performance_data:
        create_performance_plot(performance_data)

def create_performance_plot(performance_data):
    """Create performance visualization."""
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sizes = [d['corpus_size'] for d in performance_data]
    index_times = [d['index_time'] for d in performance_data]
    retrieval_times = [d['retrieval_time'] for d in performance_data]
    
    # Indexing time plot
    ax1.plot(sizes, index_times, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Corpus Size')
    ax1.set_ylabel('Indexing Time (seconds)')
    ax1.set_title('Indexing Performance')
    ax1.grid(True, alpha=0.3)
    
    # Retrieval time plot
    ax2.plot(sizes, retrieval_times, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Corpus Size')
    ax2.set_ylabel('Retrieval Time (seconds)')
    ax2.set_title('Retrieval Performance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/performance_analysis.png', dpi=300, bbox_inches='tight')
    console.print(f"\n[bold green]Performance plot saved to: examples/performance_analysis.png[/]")

def demo_dataset_management():
    """Demonstrate dataset management capabilities."""
    console.print(Panel("Dataset Management Demo", style="bold green"))
    
    dataset_manager = DatasetManager(cache_dir="examples/dataset_cache")
    
    # Create synthetic dataset
    console.print("\n[bold cyan]Creating synthetic dataset...[/]")
    synthetic_data = dataset_manager.create_synthetic_dataset(
        size=20,
        domains=["technology", "science", "health"]
    )
    
    console.print(f"Generated {len(synthetic_data['documents'])} documents")
    console.print(f"Generated {len(synthetic_data['queries'])} queries")
    
    # Show sample documents
    table = Table(title="Sample Synthetic Documents")
    table.add_column("ID", style="magenta")
    table.add_column("Document")
    
    for i, doc in enumerate(synthetic_data['documents'][:3]):
        table.add_row(str(i), doc[:80] + "...")
    
    console.print(table)
    
    # Show sample queries
    table = Table(title="Sample Synthetic Queries")
    table.add_column("ID", style="magenta")
    table.add_column("Query")
    
    for i, query in enumerate(synthetic_data['queries'][:3]):
        table.add_row(str(i), query)
    
    console.print(table)

def demo_comprehensive_benchmark():
    """Demonstrate comprehensive benchmarking."""
    console.print(Panel("Comprehensive Benchmarking Demo", style="bold purple"))
    
    # Create a simple benchmark suite
    benchmark_suite = BenchmarkSuite(output_dir="examples/benchmark_results")
    
    # Test documents and queries
    documents = [
        "Artificial intelligence enables machines to simulate human intelligence.",
        "Machine learning algorithms improve automatically through experience.",
        "Deep learning is a subset of machine learning using neural networks.",
        "Natural language processing helps computers understand human language.",
        "Computer vision allows machines to interpret and analyze visual content.",
        "Robotics combines AI with mechanical engineering for automation.",
        "Data science extracts insights and knowledge from structured data.",
        "Big data refers to extremely large datasets that require special tools."
    ]
    
    queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are neural networks used for?"
    ]
    
    # Create retriever
    retriever = HybridRetriever(dense_model="all-MiniLM-L6-v2")
    retriever.index(documents, show_progress=True)
    
    # Run performance profiling
    console.print("\n[bold cyan]Running performance profiling...[/]")
    profile_results = benchmark_suite.profile_performance(
        retriever=retriever,
        queries=queries,
        corpus_sizes=[len(documents)],
        fusion_strategies=["rrf", "linear"]
    )
    
    # Display results
    if profile_results:
        table = Table(title="Performance Profile Results")
        table.add_column("Metric", style="magenta")
        table.add_column("Value", style="green")
        
        for metric, value in profile_results.items():
            if isinstance(value, (int, float)):
                table.add_row(str(metric), f"{value:.4f}")
            else:
                table.add_row(str(metric), str(value))
        
        console.print(table)

def main():
    """Run comprehensive demonstration."""
    console.print(Panel("OpenEmbeddings Comprehensive Demo", style="bold white on blue"))
    console.print("This demo showcases the key capabilities of OpenEmbeddings.")
    
    # Create output directory
    os.makedirs("examples", exist_ok=True)
    
    try:
        # Run demonstrations
        documents, dense_emb, sparse_scores = demo_embedding_models()
        demo_hybrid_retrieval(documents)
        demo_reranking(documents)
        demo_performance_profiling()
        demo_dataset_management()
        demo_comprehensive_benchmark()
        
        console.print(Panel("Demo completed successfully!", style="bold green"))
        console.print("\nCheck the 'examples/' directory for generated outputs:")
        console.print("  • performance_analysis.png - Performance visualization")
        console.print("  • dataset_cache/ - Cached research datasets")
        console.print("  • benchmark_results/ - Benchmark outputs")
        
    except Exception as e:
        console.print(f"[bold red]Error during demo: {e}[/]")
        raise

if __name__ == "__main__":
    main() 