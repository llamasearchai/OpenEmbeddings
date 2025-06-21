"""Advanced CLI for OpenEmbeddings research and experimentation.

This module provides comprehensive command-line tools for:
- Running benchmarks and evaluations
- Managing research datasets
- Conducting experiments
- Hyperparameter optimization
- Performance profiling
- Visualization and reporting

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich.text import Text

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    warnings.warn("Pandas not available. Some features may be limited.")

from .models.hybrid_retriever import HybridRetriever
from .models.dense_embedder import DenseEmbedder
from .models.sparse_embedder import SparseEmbedder
from .models.reranker import ReRanker
from .benchmarks import BenchmarkSuite, DatasetManager
from .experiments import ExperimentRunner, ExperimentConfig, AutoMLExperiment

app = typer.Typer(
    add_completion=False,
    help="OpenEmbeddings Advanced Research CLI - Comprehensive tools for embedding research and experimentation.",
)
console = Console()


@app.command()
def benchmark(
    model_path: Path = typer.Option(
        ...,
        "--model-path",
        "-m",
        help="Path to trained model or model configuration",
        exists=True,
    ),
    datasets: List[str] = typer.Option(
        ["scifact", "nfcorpus"],
        "--datasets",
        "-d",
        help="BEIR datasets to evaluate on",
    ),
    output_dir: Path = typer.Option(
        "benchmark_results",
        "--output-dir",
        "-o",
        help="Directory to save benchmark results",
    ),
    use_reranker: bool = typer.Option(
        False,
        "--rerank",
        help="Enable cross-encoder re-ranking",
    ),
    reranker_model: str = typer.Option(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "--reranker-model",
        help="Cross-encoder model for re-ranking",
    ),
    top_k: int = typer.Option(
        100,
        "--top-k",
        help="Number of documents to retrieve for evaluation",
    ),
):
    """Run comprehensive benchmarks on BEIR datasets."""
    
    console.print(Panel("Starting OpenEmbeddings Benchmark Suite", style="bold blue"))
    
    # Load model
    try:
        retriever = HybridRetriever.from_pretrained(str(model_path))
        console.print(f"Loaded model from: {model_path}")
    except Exception as e:
        console.print(f"Error loading model: {e}", style="bold red")
        raise typer.Exit(1)
    
    # Initialize benchmark suite
    benchmark_suite = BenchmarkSuite(str(output_dir))
    
    # Run benchmarks
    results = {}
    
    for dataset_name in track(datasets, description="Running benchmarks..."):
        console.print(f"\nEvaluating on {dataset_name.upper()}")
        
        try:
            dataset_results = benchmark_suite.evaluate_retrieval_system(
                retriever=retriever,
                dataset_name=dataset_name,
                top_k=top_k,
                use_reranker=use_reranker,
                reranker_model=reranker_model
            )
            results[dataset_name] = dataset_results
            
            # Display results
            table = Table(title=f"Results for {dataset_name.upper()}")
            table.add_column("Metric", style="cyan")
            table.add_column("Score", style="green")
            
            for metric, score in dataset_results.items():
                table.add_row(metric, f"{score:.4f}")
            
            console.print(table)
            
        except Exception as e:
            console.print(f"Error evaluating {dataset_name}: {e}", style="bold red")
            continue
    
    # Create comprehensive report
    console.print("\nCreating visualization dashboard...")
    benchmark_suite.create_visualization_dashboard({"beir": results})
    
    # Save results
    results_file = output_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\nBenchmark complete! Results saved to: {output_dir}")
    console.print(f"View dashboard at: {output_dir}/performance_dashboard.html")


@app.command()
def experiment(
    config_file: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to experiment configuration file",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        "experiments",
        "--output-dir",
        "-o",
        help="Directory to save experiment results",
    ),
    use_wandb: bool = typer.Option(
        False,
        "--wandb",
        help="Enable Weights & Biases logging",
    ),
    wandb_project: str = typer.Option(
        "openembeddings",
        "--wandb-project",
        help="Weights & Biases project name",
    ),
):
    """Run comprehensive experiments with multiple models and datasets."""
    
    console.print(Panel("Starting OpenEmbeddings Experiment Suite", style="bold green"))
    
    # Load experiment configuration
    try:
        with open(config_file, "r") as f:
            config_data = json.load(f)
        
        config = ExperimentConfig(
            **config_data,
            output_dir=str(output_dir),
            use_wandb=use_wandb,
            wandb_project=wandb_project
        )
        console.print(f"Loaded experiment config: {config.name}")
        
    except Exception as e:
        console.print(f"Error loading config: {e}", style="bold red")
        raise typer.Exit(1)
    
    # Initialize experiment runner
    runner = ExperimentRunner(config)
    
    # Load datasets
    console.print("Loading datasets...")
    dataset_manager = DatasetManager()
    
    datasets = []
    for dataset_name in config.datasets:
        if dataset_name == "synthetic":
            dataset = dataset_manager.create_synthetic_dataset(size=1000)
            dataset["name"] = "synthetic"
            datasets.append(dataset)
        else:
            # Try to load from BEIR or other sources
            try:
                beir_datasets = dataset_manager.load_beir_datasets([dataset_name])
                if dataset_name in beir_datasets:
                    beir_data = beir_datasets[dataset_name]
                    dataset = {
                        "name": dataset_name,
                        "documents": [doc["text"] for doc in beir_data["corpus"].values()],
                        "queries": list(beir_data["queries"].values())
                    }
                    datasets.append(dataset)
            except Exception as e:
                console.print(f"Could not load {dataset_name}: {e}", style="yellow")
                continue
    
    if not datasets:
        console.print("No datasets loaded successfully", style="bold red")
        raise typer.Exit(1)
    
    console.print(f"Loaded {len(datasets)} datasets")
    
    # Run comparison study
    console.print("\nRunning comparison study...")
    results_df = runner.run_comparison_study(
        model_configs=config.model_configs,
        datasets=datasets,
        metrics=config.metrics
    )
    
    # Display results summary
    if _PANDAS_AVAILABLE and not results_df.empty:
        console.print("\nResults Summary:")
        
        summary_table = Table(title="Experiment Results")
        summary_table.add_column("Model", style="cyan")
        summary_table.add_column("Dataset", style="magenta")
        summary_table.add_column("Accuracy", style="green")
        summary_table.add_column("Runtime (s)", style="yellow")
        
        for _, row in results_df.iterrows():
            summary_table.add_row(
                row["model"],
                row["dataset"],
                f"{row.get('accuracy', 0):.4f}",
                f"{row.get('runtime', 0):.2f}"
            )
        
        console.print(summary_table)
    
    # Generate report
    console.print("\nGenerating experiment report...")
    report_path = runner.generate_experiment_report()
    results_path = runner.save_results()
    
    console.print(f"\nExperiment complete!")
    console.print(f"Report: {report_path}")
    console.print(f"Results: {results_path}")


@app.command()
def optimize(
    dataset_path: Path = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to dataset file or directory",
    ),
    output_dir: Path = typer.Option(
        "optimization_results",
        "--output-dir",
        "-o",
        help="Directory to save optimization results",
    ),
    n_trials: int = typer.Option(
        50,
        "--trials",
        "-n",
        help="Number of optimization trials",
    ),
    time_budget: int = typer.Option(
        60,
        "--time-budget",
        "-t",
        help="Time budget in minutes",
    ),
    metric: str = typer.Option(
        "accuracy",
        "--metric",
        "-m",
        help="Optimization metric",
    ),
):
    """Run hyperparameter optimization for retrieval system."""
    
    console.print(Panel("Starting Hyperparameter Optimization", style="bold yellow"))
    
    # Load dataset
    console.print("Loading dataset...")
    try:
        if dataset_path.is_file():
            with open(dataset_path, "r") as f:
                dataset = json.load(f)
        else:
            # Try to load as BEIR dataset
            dataset_manager = DatasetManager()
            datasets = dataset_manager.load_beir_datasets([dataset_path.name])
            if dataset_path.name in datasets:
                beir_data = datasets[dataset_path.name]
                dataset = {
                    "documents": [doc["text"] for doc in beir_data["corpus"].values()],
                    "queries": list(beir_data["queries"].values())
                }
            else:
                raise ValueError(f"Could not load dataset from {dataset_path}")
        
        console.print(f"Loaded dataset with {len(dataset.get('documents', []))} documents")
        
    except Exception as e:
        console.print(f"Error loading dataset: {e}", style="bold red")
        raise typer.Exit(1)
    
    # Run AutoML optimization
    automl = AutoMLExperiment(str(output_dir))
    
    console.print(f"Running optimization with {time_budget} minute budget...")
    
    results = automl.auto_optimize_retrieval_system(
        datasets=[dataset],
        time_budget_minutes=time_budget,
        metric=metric
    )
    
    # Display results
    console.print("\nOptimization Results:")
    
    if results["best_config"]:
        config_table = Table(title="Best Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")
        
        for param, value in results["best_config"].items():
            config_table.add_row(param, str(value))
        
        console.print(config_table)
        console.print(f"\nBest Score: {results['best_score']:.4f}")
        console.print(f"Configurations Evaluated: {results['n_evaluated']}")
    
    # Save results
    results_file = output_dir / "optimization_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    console.print(f"\nOptimization complete! Results saved to: {results_file}")


@app.command()
def profile(
    model_path: Path = typer.Option(
        ...,
        "--model-path",
        "-m",
        help="Path to trained model",
        exists=True,
    ),
    queries: List[str] = typer.Option(
        ["What is machine learning?", "How do neural networks work?"],
        "--queries",
        "-q",
        help="Test queries for profiling",
    ),
    corpus_sizes: List[int] = typer.Option(
        [100, 1000, 10000],
        "--corpus-sizes",
        help="Corpus sizes to test",
    ),
    output_dir: Path = typer.Option(
        "profiling_results",
        "--output-dir",
        "-o",
        help="Directory to save profiling results",
    ),
):
    """Profile performance across different configurations and corpus sizes."""
    
    console.print(Panel("Starting Performance Profiling", style="bold cyan"))
    
    # Load model
    try:
        retriever = HybridRetriever.from_pretrained(str(model_path))
        console.print(f"Loaded model from: {model_path}")
    except Exception as e:
        console.print(f"Error loading model: {e}", style="bold red")
        raise typer.Exit(1)
    
    # Initialize benchmark suite
    benchmark_suite = BenchmarkSuite(str(output_dir))
    
    # Run performance profiling
    console.print("Running performance profiling...")
    
    results = benchmark_suite.profile_performance(
        retriever=retriever,
        queries=queries,
        corpus_sizes=corpus_sizes,
        fusion_strategies=["linear", "rrf"]
    )
    
    # Display results
    console.print("\nProfiling Results:")
    
    # Indexing time
    indexing_table = Table(title="Indexing Performance")
    indexing_table.add_column("Configuration", style="cyan")
    indexing_table.add_column("Time (s)", style="green")
    
    for config, time_taken in results["indexing_time"].items():
        indexing_table.add_row(config, f"{time_taken:.2f}")
    
    console.print(indexing_table)
    
    # Retrieval performance
    retrieval_table = Table(title="Retrieval Performance")
    retrieval_table.add_column("Configuration", style="cyan")
    retrieval_table.add_column("Avg Time (s)", style="green")
    retrieval_table.add_column("Throughput (q/s)", style="yellow")
    
    for config in results["retrieval_time"]:
        retrieval_table.add_row(
            config,
            f"{results['retrieval_time'][config]:.4f}",
            f"{results['throughput'][config]:.2f}"
        )
    
    console.print(retrieval_table)
    
    # Create visualization
    console.print("\nCreating performance dashboard...")
    benchmark_suite.create_visualization_dashboard(results)
    
    # Save results
    results_file = output_dir / "profiling_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\nProfiling complete! Results saved to: {output_dir}")


@app.command()
def datasets(
    action: str = typer.Argument(
        ...,
        help="Action to perform: list, download, create, info"
    ),
    dataset_name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Dataset name for specific actions"
    ),
    output_dir: Path = typer.Option(
        "datasets",
        "--output-dir",
        "-o",
        help="Directory to save datasets"
    ),
    size: int = typer.Option(
        1000,
        "--size",
        "-s",
        help="Size for synthetic dataset creation"
    ),
):
    """Manage research datasets."""
    
    console.print(Panel("Dataset Management", style="bold magenta"))
    
    dataset_manager = DatasetManager(str(output_dir))
    
    if action == "list":
        console.print("Available BEIR Datasets:")
        
        beir_datasets = [
            "scifact", "nfcorpus", "arguana", "quora", "scidocs",
            "fever", "climate-fever", "dbpedia-entity", "hotpotqa",
            "fiqa", "trec-covid", "nq", "msmarco"
        ]
        
        table = Table()
        table.add_column("Dataset", style="cyan")
        table.add_column("Domain", style="green")
        table.add_column("Description", style="white")
        
        dataset_info = {
            "scifact": ("Science", "Scientific fact verification"),
            "nfcorpus": ("Medical", "Medical information retrieval"),
            "arguana": ("Debate", "Argument retrieval"),
            "quora": ("QA", "Question answering"),
            "scidocs": ("Science", "Scientific document retrieval"),
            "fever": ("Fact", "Fact verification"),
            "climate-fever": ("Climate", "Climate change fact verification"),
            "dbpedia-entity": ("Knowledge", "Entity retrieval"),
            "hotpotqa": ("QA", "Multi-hop question answering"),
            "fiqa": ("Finance", "Financial question answering"),
            "trec-covid": ("Medical", "COVID-19 research"),
            "nq": ("QA", "Natural questions"),
            "msmarco": ("Web", "Web search")
        }
        
        for dataset in beir_datasets:
            domain, desc = dataset_info.get(dataset, ("Unknown", "No description"))
            table.add_row(dataset, domain, desc)
        
        console.print(table)
    
    elif action == "download":
        if not dataset_name:
            console.print("Dataset name required for download", style="bold red")
            raise typer.Exit(1)
        
        console.print(f"Downloading {dataset_name}...")
        
        try:
            datasets = dataset_manager.load_beir_datasets([dataset_name])
            if dataset_name in datasets:
                console.print(f"Downloaded {dataset_name} successfully")
                
                # Show dataset info
                dataset = datasets[dataset_name]
                info_table = Table(title=f"{dataset_name.upper()} Info")
                info_table.add_column("Metric", style="cyan")
                info_table.add_column("Count", style="green")
                
                info_table.add_row("Documents", str(len(dataset["corpus"])))
                info_table.add_row("Queries", str(len(dataset["queries"])))
                info_table.add_row("Relevance Judgments", str(len(dataset["qrels"])))
                
                console.print(info_table)
            else:
                console.print(f"Failed to download {dataset_name}", style="bold red")
                
        except Exception as e:
            console.print(f"Error downloading {dataset_name}: {e}", style="bold red")
    
    elif action == "create":
        console.print(f"Creating synthetic dataset with {size} documents...")
        
        dataset = dataset_manager.create_synthetic_dataset(size=size)
        
        # Save dataset
        output_file = output_dir / "synthetic_dataset.json"
        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=2)
        
        console.print(f"Created synthetic dataset: {output_file}")
        console.print(f"Documents: {len(dataset['documents'])}")
        console.print(f"Queries: {len(dataset['queries'])}")
    
    elif action == "info":
        if not dataset_name:
            console.print("Dataset name required for info", style="bold red")
            raise typer.Exit(1)
        
        # Try to load dataset info
        try:
            datasets = dataset_manager.load_beir_datasets([dataset_name])
            if dataset_name in datasets:
                dataset = datasets[dataset_name]
                
                console.print(f"Dataset Information: {dataset_name.upper()}")
                
                info_table = Table()
                info_table.add_column("Metric", style="cyan")
                info_table.add_column("Value", style="green")
                
                info_table.add_row("Documents", str(len(dataset["corpus"])))
                info_table.add_row("Queries", str(len(dataset["queries"])))
                info_table.add_row("Relevance Judgments", str(len(dataset["qrels"])))
                
                # Sample documents
                sample_docs = list(dataset["corpus"].values())[:3]
                for i, doc in enumerate(sample_docs):
                    info_table.add_row(f"Sample Doc {i+1}", doc["text"][:100] + "...")
                
                console.print(info_table)
            else:
                console.print(f"Dataset {dataset_name} not found", style="bold red")
                
        except Exception as e:
            console.print(f"Error loading dataset info: {e}", style="bold red")
    
    else:
        console.print(f"Unknown action: {action}", style="bold red")
        console.print("Available actions: list, download, create, info")


@app.command()
def compare(
    results_dir: Path = typer.Option(
        "experiments",
        "--results-dir",
        "-r",
        help="Directory containing experiment results",
    ),
    metric: str = typer.Option(
        "accuracy",
        "--metric",
        "-m",
        help="Metric to compare",
    ),
    output_file: Path = typer.Option(
        "comparison_results.html",
        "--output",
        "-o",
        help="Output file for comparison dashboard",
    ),
):
    """Compare results from multiple experiments."""
    
    console.print(Panel("Comparing Experiment Results", style="bold blue"))
    
    # Find all result files
    result_files = list(results_dir.glob("*_results.json"))
    
    if not result_files:
        console.print(f"No result files found in {results_dir}", style="bold red")
        raise typer.Exit(1)
    
    console.print(f"Found {len(result_files)} result files")
    
    # Load and compare results
    if _PANDAS_AVAILABLE:
        from .experiments import compare_experiment_results, create_experiment_dashboard
        
        try:
            results_df = compare_experiment_results(
                [str(f) for f in result_files],
                metric=metric
            )
            
            # Display summary
            console.print(f"\nComparison Summary ({metric}):")
            
            summary_table = Table()
            summary_table.add_column("Experiment", style="cyan")
            summary_table.add_column("Best Model", style="green")
            summary_table.add_column(f"Best {metric}", style="yellow")
            summary_table.add_column("Avg Runtime", style="magenta")
            
            for experiment in results_df["experiment"].unique():
                exp_data = results_df[results_df["experiment"] == experiment]
                best_idx = exp_data[metric].idxmax()
                best_row = exp_data.loc[best_idx]
                avg_runtime = exp_data["runtime"].mean()
                
                summary_table.add_row(
                    experiment,
                    best_row["model"],
                    f"{best_row[metric]:.4f}",
                    f"{avg_runtime:.2f}s"
                )
            
            console.print(summary_table)
            
            # Create dashboard
            console.print(f"\nCreating comparison dashboard...")
            create_experiment_dashboard(results_df, str(output_file))
            
            console.print(f"Comparison complete! Dashboard: {output_file}")
            
        except Exception as e:
            console.print(f"Error comparing results: {e}", style="bold red")
    
    else:
        console.print("Pandas not available for comparison", style="bold red")


@app.command()
def version():
    """Show version information."""
    console.print("OpenEmbeddings Advanced CLI v1.0.0")
    console.print("Research-grade embedding and retrieval framework")
    console.print("Author: Nik Jois <nikjois@llamasearch.ai>")


def _main():  # pragma: no cover
    try:
        app()
    except Exception as exc:
        console.print(f"[bold red]FATAL ERROR:[/] {exc}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    _main() 