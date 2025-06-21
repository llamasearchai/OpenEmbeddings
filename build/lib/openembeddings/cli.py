from __future__ import annotations

"""Light-weight command-line interface for the OpenEmbeddings demo package.

Invoked via `python -m openembeddings` **or** after installation via the
console-script `openembeddings`.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import sys
from pathlib import Path
from typing import List

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from .models.dense_embedder import DenseEmbedder
from .models.hybrid_retriever import HybridRetriever
from .models.sparse_embedder import SparseEmbedder

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def encode(texts: List[str]):
    """Return deterministic dense embeddings for *texts*."""
    embedder = DenseEmbedder()
    embs = embedder.encode(list(texts))
    for t, v in zip(texts, embs):
        console.print(f"[bold cyan]{t}[/] -> {v[:6]} …")


@app.command()
def bm25(query: str, *docs: str):
    """Quick BM25 scoring of QUERY against DOCS."""
    se = SparseEmbedder()
    scores = se.compute_scores([query], list(docs))[0]
    for d, s in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True):
        console.print(f"[magenta]{s:.3f}[/] {d}")


@app.command()
def retrieve(query: str, *docs: str, top_k: int = typer.Option(3)):
    """Hybrid retrieval demo combining dense + sparse signals."""
    hr = HybridRetriever()
    results = hr.retrieve(query, list(docs), top_k=top_k)
    table = Table(title=f"Top-{top_k} for '{query}'")
    table.add_column("Rank")
    table.add_column("Score")
    table.add_column("Document")
    for i, (idx, score, doc) in enumerate(results, 1):
        table.add_row(str(i), f"{score:.3f}", doc)
    console.print(table)


@app.command()
def version():
    """Print package version."""
    from importlib.metadata import version as _v

    console.print(_v("openembeddings"))


def _main():  # pragma: no cover
    try:
        app()
    except Exception as exc:  # pylint: disable=broad-except
        console.print(f"[red]ERROR:[/] {exc}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    _main() 