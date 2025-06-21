from __future__ import annotations

"""Light-weight command-line interface for the OpenEmbeddings demo package.

Invoked via `python -m openembeddings` **or** after installation via the
console-script `openembeddings`.

This CLI provides a production-oriented workflow:
1. `build-index`: Create and save a reusable search index from a corpus.
2. `retrieve`: Load a pre-built index and query it for documents.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import sys
import enum
from pathlib import Path
from typing import List, Optional

import typer
import click  # Used for low-level echo utilities
import torch  # Needed for dtype resolution in quantization CLI

from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

from .models.dense_embedder import DenseEmbedder
from .models.hybrid_retriever import HybridRetriever
from .models.reranker import ReRanker

app = typer.Typer(
    add_completion=False,
    help="OpenEmbeddings: Build a search index and retrieve documents.",
)
console = Console()


class FusionStrategy(str, enum.Enum):
    """Enum for fusion strategies."""

    RRF = "rrf"
    LINEAR = "linear"


@app.command()
def encode(
    texts: List[str],
    model_name: str = typer.Option(
        "all-MiniLM-L6-v2", help="Name of the sentence-transformer model to use."
    ),
):
    """Return dense vector embeddings for one or more texts."""
    console.print(Panel(f"Using model: [bold cyan]{model_name}[/]", expand=False))
    embedder = DenseEmbedder(model_name=model_name)
    embs = embedder.encode(list(texts))
    for t, v in zip(texts, embs):
        console.print(f"• [dim]'{t}'[/] -> {v[:6]}…")


@app.command()
def build_index(
    output_path: Path = typer.Option(
        ...,
        "--output-path",
        "-o",
        help="Directory to save the generated index.",
        writable=True,
    ),
    input_file: Optional[Path] = typer.Option(
        None,
        "--input-file",
        "-f",
        help="Path to a text file with one document per line.",
        exists=True,
        readable=True,
    ),
    docs: Optional[List[str]] = typer.Argument(
        None, help="Raw document strings to index."
    ),
    dense_model: str = typer.Option(
        "all-MiniLM-L6-v2", help="Sentence-transformer model for dense embeddings."
    ),
    fusion_strategy: FusionStrategy = typer.Option(
        FusionStrategy.RRF,
        "--fusion",
        help="Fusion strategy to combine dense and sparse scores.",
        case_sensitive=False,
    ),
    disable_ann: bool = typer.Option(
        False,
        "--disable-ann",
        help="Disable Approximate Nearest Neighbor (ANN) for dense retrieval.",
    ),
):
    """Build and save a hybrid retrieval index from a document corpus."""
    if not input_file and not docs:
        console.print("[bold red]Error:[/] Must provide documents via --input-file or arguments.")
        raise typer.Exit(1)

    documents = []
    if input_file:
        documents.extend(l.strip() for l in input_file.read_text().splitlines() if l.strip())
    if docs:
        documents.extend(docs)

    console.print(
        f"Building index from [bold green]{len(documents)}[/] documents "
        f"using dense model [cyan]{dense_model}[/]."
    )
    console.print(
        f"Fusion strategy: [bold yellow]{fusion_strategy.value}[/], "
        f"ANN enabled: [bold yellow]{not disable_ann}[/]"
    )
    hr = HybridRetriever(
        dense_model=dense_model,
        fusion_strategy=fusion_strategy.value,
        use_ann=not disable_ann,
    )
    hr.index(documents, show_progress=True)
    hr.save_pretrained(str(output_path))
    console.print(
        f"\n[bold green]Index saved successfully to:[/] [dim]{output_path}[/]"
    )


@app.command()
def retrieve(
    query: str,
    index_path: Path = typer.Option(
        ...,
        "--index-path",
        "-i",
        help="Path to a pre-built index directory.",
        exists=True,
        readable=True,
    ),
    top_k: int = typer.Option(3, help="Number of top documents to retrieve."),
    rerank: bool = typer.Option(
        False, "--rerank", help="Enable cross-encoder re-ranking for higher accuracy."
    ),
    rerank_model: str = typer.Option(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model for re-ranking.",
    ),
):
    """Load an index and retrieve top-k documents for a given query."""
    console.print(f"Loading index from: [dim]{index_path}[/]")
    hr = HybridRetriever.from_pretrained(str(index_path))

    results = hr.retrieve(query, top_k=top_k)

    if rerank:
        console.print(
            f"Re-ranking top-{top_k} results with [cyan]{rerank_model}[/]..."
        )
        reranker = ReRanker(model_name=rerank_model)
        results = reranker.rerank(query, results)

    table = Table(
        title=f"Top-{top_k} results for: [bold cyan]'{query}'[/]"
        f"{' (Re-ranked)' if rerank else ''}"
    )
    table.add_column("Rank", style="magenta")
    table.add_column("Score", style="green")
    table.add_column("Document")
    for i, (idx, score, doc) in enumerate(results, 1):
        table.add_row(str(i), f"{score:.4f}", doc)
    console.print(table)


@app.command()
def search(
    query: str,
    index_path: Path = typer.Option(
        ...,
        "--index-path",
        "-i",
        help="Path to a search index. If it doesn't exist, it will be created.",
    ),
    top_k: int = typer.Option(3, help="Number of top documents to retrieve."),
    # Indexing options (only used if index doesn't exist)
    input_file: Optional[Path] = typer.Option(
        None,
        "--input-file",
        "-f",
        help="Path to a text file with one document per line (for index creation).",
        readable=True,
    ),
    docs: Optional[List[str]] = typer.Argument(
        None, help="Raw document strings to index (for index creation)."
    ),
    dense_model: str = typer.Option(
        "all-MiniLM-L6-v2", help="Sentence-transformer model for dense embeddings."
    ),
    fusion_strategy: FusionStrategy = typer.Option(
        FusionStrategy.RRF,
        "--fusion",
        help="Fusion strategy to combine dense and sparse scores.",
        case_sensitive=False,
    ),
    disable_ann: bool = typer.Option(
        False,
        "--disable-ann",
        help="Disable Approximate Nearest Neighbor (ANN) for dense retrieval.",
    ),
    # Re-ranking options
    rerank: bool = typer.Option(
        False, "--rerank", help="Enable cross-encoder re-ranking for higher accuracy."
    ),
    rerank_model: str = typer.Option(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model for re-ranking.",
    ),
):
    """Search an index, building it first if it doesn't exist."""
    if not index_path.exists():
        console.print(f"Index not found at [dim]{index_path}[/]. Building it now...")
        # Call build_index logic
        if not input_file and not docs:
            console.print(
                "[bold red]Error:[/] To build an index, must provide documents "
                "via --input-file or arguments."
            )
            raise typer.Exit(1)

        documents = []
        if input_file:
            documents.extend(
                l.strip() for l in input_file.read_text().splitlines() if l.strip()
            )
        if docs:
            documents.extend(docs)

        console.print(
            f"Building index from [bold green]{len(documents)}[/] documents "
            f"using dense model [cyan]{dense_model}[/]."
        )
        console.print(
            f"Fusion strategy: [bold yellow]{fusion_strategy.value}[/], "
            f"ANN enabled: [bold yellow]{not disable_ann}[/]"
        )
        hr = HybridRetriever(
            dense_model=dense_model,
            fusion_strategy=fusion_strategy.value,
            use_ann=not disable_ann,
        )
        hr.index(documents, show_progress=True)
        hr.save_pretrained(str(index_path))
        console.print(
            f"\n[bold green]Index saved successfully to:[/] [dim]{index_path}[/]\n"
        )

    # Call retrieve logic
    console.print(f"Retrieving from index: [dim]{index_path}[/]")
    hr = HybridRetriever.from_pretrained(str(index_path))

    results = hr.retrieve(query, top_k=top_k)

    if rerank:
        console.print(
            f"Re-ranking top-{top_k} results with [cyan]{rerank_model}[/]..."
        )
        reranker = ReRanker(model_name=rerank_model)
        results = reranker.rerank(query, results)

    table = Table(
        title=f"Top-{top_k} results for: [bold cyan]'{query}'[/]"
        f"{' (Re-ranked)' if rerank else ''}"
    )
    table.add_column("Rank", style="magenta")
    table.add_column("Score", style="green")
    table.add_column("Document")
    for i, (idx, score, doc) in enumerate(results, 1):
        table.add_row(str(i), f"{score:.4f}", doc)
    console.print(table)


@app.command()
def research():
    """Launch advanced research CLI with comprehensive tools."""
    console.print(Panel("Launching OpenEmbeddings Research CLI", style="bold blue"))
    console.print("For advanced research capabilities, use:")
    console.print("  [bold cyan]python -m openembeddings.advanced_cli --help[/]")
    console.print("\nAvailable research commands:")
    console.print("  • [green]benchmark[/] - Run comprehensive BEIR evaluations")
    console.print("  • [green]experiment[/] - Conduct multi-model experiments")
    console.print("  • [green]optimize[/] - Hyperparameter optimization")
    console.print("  • [green]profile[/] - Performance profiling")
    console.print("  • [green]datasets[/] - Dataset management")
    console.print("  • [green]compare[/] - Compare experiment results")


@app.command()
def version():
    """Print package version."""
    from importlib.metadata import version as _v

    console.print(f"OpenEmbeddings version: [bold green]{_v('openembeddings')}[/]")


@app.command()
@typer.argument("query")
@typer.option("--dense-model", default="all-MiniLM-L6-v2", help="Dense embedding model")
@typer.option("--sparse-model", default="bm25", help="Sparse embedding model")
@typer.option("--top-k", default=10, help="Number of results to return")
@typer.option("--rrf-k", default=60, help="RRF constant for hybrid fusion")
@typer.option("--dense-weight", default=0.5, help="Weight for dense scores")
@typer.option("--sparse-weight", default=0.5, help="Weight for sparse scores")
def hybrid_retrieve(query, dense_model, sparse_model, top_k, rrf_k, dense_weight, sparse_weight):
    """Perform hybrid search combining dense and sparse retrieval."""
    from openembeddings.models.dense_embedder import DenseEmbedder
    from openembeddings.models.sparse_embedder import SparseEmbedder
    from openembeddings.models.hybrid_retriever import HybridRetriever
    
    console.print(f"Performing hybrid search with '{query}'...")
    
    # Initialize models
    dense_embedder = DenseEmbedder(model_name=dense_model)
    sparse_embedder = SparseEmbedder(model_name=sparse_model)
    
    # Create hybrid retriever
    hybrid = HybridRetriever(
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        k=top_k,
        rrf_k=rrf_k
    )
    
    # Retrieve results
    results = hybrid.retrieve(
        query,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight
    )
    
    # Display results
    console.print("\nTop results:")
    for rank, (doc_id, score) in enumerate(results, 1):
        console.print(f"{rank}. {doc_id} (score: {score:.4f})")
    
    console.print(f"\nHybrid search completed with {len(results)} results")


@app.command()
@typer.argument("model_name")
@typer.option("--output-path", required=True, help="Output path for quantized model")
@typer.option("--quant-type", default="8bit", help="Quantization type (8bit or 4bit)")
@typer.option("--dtype", default="float16", help="Data type for quantization")
def quantize_model(model_name, output_path, quant_type, dtype):
    """Quantize a model for efficient inference."""
    from openembeddings.utils import quantize_model
    from transformers import AutoModel
    
    console.print(f"Quantizing {model_name} to {quant_type} precision...")
    
    # Load model
    model = AutoModel.from_pretrained(model_name)
    
    # Apply quantization
    quantized_model = quantize_model(
        model,
        quant_type=quant_type,
        dtype=getattr(torch, dtype)
    )
    
    # Save quantized model
    quantized_model.save_pretrained(output_path)
    console.print(f"Quantized model saved to {output_path}")


@app.command()
@typer.argument("model_name")
@typer.option("--output-path", required=True, help="Output path for ONNX model")
def export_onnx(model_name, output_path):
    """Export model to ONNX format."""
    from openembeddings.utils import convert_to_onnx
    from transformers import AutoModel, AutoTokenizer
    
    console.print(f"Exporting {model_name} to ONNX format...")
    
    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Convert to ONNX
    convert_to_onnx(model, tokenizer, output_path)
    console.print(f"ONNX model saved to {output_path}")


@app.command()
@typer.option("--model-name", default="all-MiniLM-L6-v2", help="Model to evaluate")
@typer.option("--task-types", multiple=True, help="Task types to evaluate (e.g. Retrieval)")
@typer.option("--task-categories", multiple=True, help="Task categories (e.g. s2s)")
@typer.option("--task-langs", multiple=True, default=["en"], help="Languages to evaluate")
@typer.option("--tasks", multiple=True, help="Specific tasks to evaluate")
@typer.option("--output-dir", default="mteb_results", help="Output directory for results")
@typer.option("--batch-size", default=32, help="Batch size for encoding")
@typer.option("--visualize", is_flag=True, help="Generate visualization of results")
def evaluate_mteb(model_name, task_types, task_categories, task_langs, tasks, output_dir, batch_size, visualize):
    """Evaluate model using Massive Text Embedding Benchmark (MTEB)."""
    from openembeddings.evaluation_harness import evaluate_with_mteb, visualize_results
    from openembeddings.models.dense_embedder import DenseEmbedder
    
    click.echo("Running MTEB evaluation...")
    
    # Initialize model
    model = DenseEmbedder(model_name=model_name)
    
    # Convert multiple options to lists
    task_types = list(task_types) if task_types else None
    task_categories = list(task_categories) if task_categories else None
    task_langs = list(task_langs)
    tasks = list(tasks) if tasks else None
    
    # Run evaluation
    results = evaluate_with_mteb(
        model,
        task_types=task_types,
        task_categories=task_categories,
        task_langs=task_langs,
        tasks=tasks,
        output_folder=output_dir,
        batch_size=batch_size,
        verbosity=1
    )
    
    # Print summary
    click.echo("\nMTEB Evaluation Results:")
    for task_name, metrics in results.items():
        click.echo(f"\n{task_name}:")
        for metric, score in metrics.items():
            click.echo(f"  {metric}: {score:.4f}")
    
    # Generate visualization
    if visualize:
        visualize_path = f"{output_dir}/results_radar.png"
        visualize_results(results, visualize_path)
        click.echo(f"\nVisualization saved to {visualize_path}")
    
    click.echo(f"\nEvaluation complete. Full results in {output_dir}")


def _main():  # pragma: no cover
    try:
        app()
    except Exception as exc:
        console.print(f"[bold red]FATAL ERROR:[/] {exc}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    _main()
