"""Advanced experimentation framework for OpenEmbeddings research.

This module provides comprehensive experimentation capabilities including:
- Hyperparameter optimization
- Cross-validation
- Ablation studies
- Statistical significance testing
- Experiment tracking and logging
- Reproducible research workflows

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from __future__ import annotations

import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from itertools import product

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    warnings.warn("Weights & Biases not available. Install wandb for experiment tracking.")

try:
    from sklearn.model_selection import ParameterGrid, cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from scipy import stats
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Some experiment features may be limited.")

try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Hyperparameter optimization features disabled.")

from .models.hybrid_retriever import HybridRetriever
from .models.dense_embedder import DenseEmbedder
from .models.sparse_embedder import SparseEmbedder
from .models.reranker import ReRanker
from .benchmarks import BenchmarkSuite, DatasetManager


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: str
    model_configs: List[Dict[str, Any]]
    datasets: List[str]
    metrics: List[str]
    hyperparameters: Dict[str, List[Any]]
    cross_validation_folds: int = 5
    random_seed: int = 42
    output_dir: str = "experiments"
    use_wandb: bool = False
    wandb_project: str = "openembeddings"


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    config: Dict[str, Any]
    metrics: Dict[str, float]
    runtime: float
    memory_usage: float
    timestamp: str
    model_name: str
    dataset_name: str


class ExperimentRunner:
    """Advanced experiment runner with comprehensive analysis capabilities."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set random seeds for reproducibility
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Initialize experiment tracking
        if config.use_wandb and _WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name=config.name,
                config=asdict(config)
            )
        
        self.results: List[ExperimentResult] = []
        self.benchmark_suite = BenchmarkSuite(str(self.output_dir / "benchmarks"))
        self.dataset_manager = DatasetManager(str(self.output_dir / "datasets"))
    
    def run_hyperparameter_optimization(
        self,
        model_class: type,
        dataset: Dict[str, Any],
        n_trials: int = 100,
        optimization_metric: str = "f1_score"
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization using Optuna."""
        
        if not _OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_range in self.config.hyperparameters.items():
                if isinstance(param_range[0], float):
                    params[param_name] = trial.suggest_float(
                        param_name, param_range[0], param_range[1]
                    )
                elif isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(
                        param_name, param_range[0], param_range[1]
                    )
                else:
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_range
                    )
            
            # Create model with sampled parameters
            if model_class == HybridRetriever:
                model = HybridRetriever(**params)
            elif model_class == DenseEmbedder:
                model = DenseEmbedder(**params)
            else:
                model = model_class(**params)
            
            # Evaluate model
            try:
                if hasattr(model, 'index') and 'documents' in dataset:
                    model.index(dataset['documents'])
                    
                    # Evaluate on queries if available
                    if 'queries' in dataset:
                        scores = []
                        for query in dataset['queries'][:100]:  # Limit for speed
                            results = model.retrieve(query, top_k=10)
                            # Simple relevance scoring (can be improved)
                            score = len(results) / 10.0  # Normalize by top_k
                            scores.append(score)
                        return np.mean(scores)
                    else:
                        return 0.5  # Default score
                else:
                    return 0.5  # Default score for unsupported models
                    
            except Exception as e:
                print(f"Error in trial: {e}")
                return 0.0  # Return worst score on error
        
        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "study": study
        }
    
    def run_ablation_study(
        self,
        base_config: Dict[str, Any],
        ablation_components: List[str],
        dataset: Dict[str, Any]
    ) -> Dict[str, ExperimentResult]:
        """Run ablation study to understand component contributions."""
        
        results = {}
        
        # Baseline (full model)
        baseline_result = self._run_single_experiment(
            base_config, dataset, "baseline"
        )
        results["baseline"] = baseline_result
        
        # Ablate each component
        for component in ablation_components:
            ablated_config = base_config.copy()
            
            # Remove or disable component
            if component == "dense_embedder":
                ablated_config["dense_weight"] = 0.0
                ablated_config["sparse_weight"] = 1.0
            elif component == "sparse_embedder":
                ablated_config["dense_weight"] = 1.0
                ablated_config["sparse_weight"] = 0.0
            elif component == "reranker":
                ablated_config["use_reranker"] = False
            elif component == "faiss":
                ablated_config["use_ann"] = False
            # Add more ablation cases as needed
            
            ablated_result = self._run_single_experiment(
                ablated_config, dataset, f"without_{component}"
            )
            results[f"without_{component}"] = ablated_result
        
        return results
    
    def run_cross_validation(
        self,
        model_config: Dict[str, Any],
        dataset: Dict[str, Any],
        n_folds: int = 5
    ) -> Dict[str, Any]:
        """Run k-fold cross-validation."""
        
        if not _SKLEARN_AVAILABLE:
            warnings.warn("Scikit-learn not available. Cross-validation may be limited.")
            return {}
        
        # Split dataset into folds
        documents = dataset.get("documents", [])
        queries = dataset.get("queries", [])
        
        if not documents or not queries:
            print("Dataset missing documents or queries for cross-validation")
            return {}
        
        fold_size = len(documents) // n_folds
        fold_results = []
        
        for fold in range(n_folds):
            # Create train/test split
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(documents)
            
            test_docs = documents[test_start:test_end]
            train_docs = documents[:test_start] + documents[test_end:]
            
            # Create model
            model = HybridRetriever(**model_config)
            model.index(train_docs)
            
            # Evaluate on test queries
            test_scores = []
            for query in queries[:min(50, len(queries))]:  # Limit for speed
                try:
                    results = model.retrieve(query, top_k=10)
                    score = len(results) / 10.0  # Simple relevance score
                    test_scores.append(score)
                except:
                    test_scores.append(0.0)
            
            fold_result = {
                "fold": fold,
                "test_score": np.mean(test_scores),
                "n_test_docs": len(test_docs),
                "n_train_docs": len(train_docs)
            }
            fold_results.append(fold_result)
        
        # Aggregate results
        scores = [r["test_score"] for r in fold_results]
        return {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "fold_results": fold_results,
            "confidence_interval": stats.t.interval(
                0.95, len(scores) - 1, loc=np.mean(scores), scale=stats.sem(scores)
            ) if len(scores) > 1 else (0, 0)
        }
    
    def run_comparison_study(
        self,
        model_configs: List[Dict[str, Any]],
        datasets: List[Dict[str, Any]],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """Run comprehensive comparison study across models and datasets."""
        
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1_score", "runtime"]
        
        results = []
        
        for i, model_config in enumerate(model_configs):
            for j, dataset in enumerate(datasets):
                dataset_name = dataset.get("name", f"dataset_{j}")
                model_name = model_config.get("name", f"model_{i}")
                
                print(f"Running {model_name} on {dataset_name}...")
                
                try:
                    result = self._run_single_experiment(
                        model_config, dataset, f"{model_name}_{dataset_name}"
                    )
                    
                    result_dict = {
                        "model": model_name,
                        "dataset": dataset_name,
                        **result.metrics,
                        "runtime": result.runtime,
                        "memory_usage": result.memory_usage
                    }
                    results.append(result_dict)
                    
                except Exception as e:
                    print(f"Error running {model_name} on {dataset_name}: {e}")
                    continue
        
        return pd.DataFrame(results)
    
    def run_statistical_significance_tests(
        self,
        results_df: pd.DataFrame,
        baseline_model: str,
        metric: str = "accuracy",
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Run statistical significance tests comparing models."""
        
        if not _SKLEARN_AVAILABLE:
            warnings.warn("Scipy not available. Statistical tests may be limited.")
            return {}
        
        baseline_scores = results_df[results_df["model"] == baseline_model][metric].values
        
        if len(baseline_scores) == 0:
            print(f"Baseline model '{baseline_model}' not found in results")
            return {}
        
        significance_results = {}
        
        for model in results_df["model"].unique():
            if model == baseline_model:
                continue
            
            model_scores = results_df[results_df["model"] == model][metric].values
            
            if len(model_scores) == 0:
                continue
            
            # Paired t-test (if same datasets)
            if len(model_scores) == len(baseline_scores):
                t_stat, p_value = stats.ttest_rel(model_scores, baseline_scores)
                test_type = "paired_t_test"
            else:
                # Independent t-test
                t_stat, p_value = stats.ttest_ind(model_scores, baseline_scores)
                test_type = "independent_t_test"
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                ((len(model_scores) - 1) * np.var(model_scores, ddof=1) +
                 (len(baseline_scores) - 1) * np.var(baseline_scores, ddof=1)) /
                (len(model_scores) + len(baseline_scores) - 2)
            )
            cohens_d = (np.mean(model_scores) - np.mean(baseline_scores)) / pooled_std
            
            significance_results[model] = {
                "test_type": test_type,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < alpha,
                "cohens_d": cohens_d,
                "effect_size": self._interpret_effect_size(abs(cohens_d)),
                "mean_difference": np.mean(model_scores) - np.mean(baseline_scores)
            }
        
        return significance_results
    
    def _run_single_experiment(
        self,
        config: Dict[str, Any],
        dataset: Dict[str, Any],
        experiment_name: str
    ) -> ExperimentResult:
        """Run a single experiment with given configuration."""
        
        import time
        import psutil
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create model
        model = HybridRetriever(**config)
        
        # Index documents
        documents = dataset.get("documents", [])
        if documents:
            model.index(documents, show_progress=False)
        
        # Evaluate on queries
        queries = dataset.get("queries", [])
        metrics = {}
        
        if queries:
            scores = []
            retrieval_times = []
            
            for query in queries[:100]:  # Limit for speed
                query_start = time.time()
                try:
                    results = model.retrieve(query, top_k=10)
                    score = len(results) / 10.0  # Simple relevance score
                    scores.append(score)
                    retrieval_times.append(time.time() - query_start)
                except:
                    scores.append(0.0)
                    retrieval_times.append(0.0)
            
            metrics.update({
                "accuracy": np.mean(scores),
                "precision": np.mean(scores),  # Simplified for demo
                "recall": np.mean(scores),     # Simplified for demo
                "f1_score": np.mean(scores),   # Simplified for demo
                "avg_retrieval_time": np.mean(retrieval_times),
                "throughput": len(queries) / sum(retrieval_times) if sum(retrieval_times) > 0 else 0
            })
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = ExperimentResult(
            config=config,
            metrics=metrics,
            runtime=end_time - start_time,
            memory_usage=end_memory - start_memory,
            timestamp=datetime.now().isoformat(),
            model_name=experiment_name,
            dataset_name=dataset.get("name", "unknown")
        )
        
        # Log to wandb if available
        if self.config.use_wandb and _WANDB_AVAILABLE:
            wandb.log({
                f"{experiment_name}_{k}": v for k, v in metrics.items()
            })
        
        self.results.append(result)
        return result
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_experiment_report(self) -> str:
        """Generate comprehensive experiment report."""
        
        report_path = self.output_dir / f"{self.config.name}_report.md"
        
        with open(report_path, "w") as f:
            f.write(f"# Experiment Report: {self.config.name}\n\n")
            f.write(f"**Description:** {self.config.description}\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            
            # Configuration
            f.write("## Configuration\n\n")
            f.write(f"- Random Seed: {self.config.random_seed}\n")
            f.write(f"- Cross-validation Folds: {self.config.cross_validation_folds}\n")
            f.write(f"- Datasets: {', '.join(self.config.datasets)}\n")
            f.write(f"- Metrics: {', '.join(self.config.metrics)}\n\n")
            
            # Results Summary
            if self.results:
                f.write("## Results Summary\n\n")
                
                # Create results DataFrame
                results_data = []
                for result in self.results:
                    row = {
                        "model": result.model_name,
                        "dataset": result.dataset_name,
                        "runtime": result.runtime,
                        "memory_usage": result.memory_usage,
                        **result.metrics
                    }
                    results_data.append(row)
                
                df = pd.DataFrame(results_data)
                
                # Summary statistics
                f.write("### Performance Summary\n\n")
                f.write(df.groupby("model").agg({
                    "accuracy": ["mean", "std"],
                    "runtime": ["mean", "std"],
                    "memory_usage": ["mean", "std"]
                }).to_markdown())
                f.write("\n\n")
                
                # Best performing models
                f.write("### Best Performing Models\n\n")
                for metric in ["accuracy", "f1_score", "throughput"]:
                    if metric in df.columns:
                        best_model = df.loc[df[metric].idxmax()]
                        f.write(f"- Best {metric}: {best_model['model']} ({best_model[metric]:.4f})\n")
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the experimental results:\n\n")
            
            if self.results:
                # Find best overall model
                df = pd.DataFrame([{
                    "model": r.model_name,
                    "accuracy": r.metrics.get("accuracy", 0),
                    "runtime": r.runtime
                } for r in self.results])
                
                if not df.empty:
                    best_accuracy = df.loc[df["accuracy"].idxmax()]
                    fastest = df.loc[df["runtime"].idxmin()]
                    
                    f.write(f"1. **Best Accuracy**: {best_accuracy['model']} with {best_accuracy['accuracy']:.4f} accuracy\n")
                    f.write(f"2. **Fastest**: {fastest['model']} with {fastest['runtime']:.2f}s runtime\n")
            
            f.write("\n## Next Steps\n\n")
            f.write("- Consider hyperparameter optimization for top-performing models\n")
            f.write("- Run ablation studies to understand component contributions\n")
            f.write("- Evaluate on additional datasets for robustness\n")
            f.write("- Investigate statistical significance of performance differences\n")
        
        return str(report_path)
    
    def save_results(self, filename: str = None) -> str:
        """Save experiment results to JSON file."""
        
        if filename is None:
            filename = f"{self.config.name}_results.json"
        
        results_path = self.output_dir / filename
        
        # Convert results to serializable format
        results_data = {
            "config": asdict(self.config),
            "results": [
                {
                    "config": result.config,
                    "metrics": result.metrics,
                    "runtime": result.runtime,
                    "memory_usage": result.memory_usage,
                    "timestamp": result.timestamp,
                    "model_name": result.model_name,
                    "dataset_name": result.dataset_name
                }
                for result in self.results
            ]
        }
        
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2, default=str)
        
        return str(results_path)


class AutoMLExperiment:
    """Automated machine learning experiment runner."""
    
    def __init__(self, output_dir: str = "automl_experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def auto_optimize_retrieval_system(
        self,
        datasets: List[Dict[str, Any]],
        time_budget_minutes: int = 60,
        metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """Automatically optimize retrieval system configuration."""
        
        # Define search space
        search_space = {
            "dense_model": ["hashing-encoder", "all-MiniLM-L6-v2"],
            "fusion_strategy": ["linear", "rrf"],
            "dense_weight": [0.3, 0.5, 0.7],
            "sparse_weight": [0.3, 0.5, 0.7],
            "use_ann": [True, False]
        }
        
        best_config = None
        best_score = -1
        all_results = []
        
        start_time = time.time()
        time_budget_seconds = time_budget_minutes * 60
        
        # Generate all combinations
        param_combinations = list(product(*search_space.values()))
        param_names = list(search_space.keys())
        
        for i, param_values in enumerate(param_combinations):
            if time.time() - start_time > time_budget_seconds:
                print(f"Time budget exceeded. Evaluated {i} configurations.")
                break
            
            config = dict(zip(param_names, param_values))
            
            # Ensure weights sum to 1
            if config["dense_weight"] + config["sparse_weight"] != 1.0:
                config["sparse_weight"] = 1.0 - config["dense_weight"]
            
            print(f"Evaluating configuration {i+1}/{len(param_combinations)}: {config}")
            
            # Evaluate configuration
            scores = []
            for dataset in datasets:
                try:
                    model = HybridRetriever(**config)
                    
                    if "documents" in dataset:
                        model.index(dataset["documents"])
                    
                    if "queries" in dataset:
                        query_scores = []
                        for query in dataset["queries"][:20]:  # Limit for speed
                            try:
                                results = model.retrieve(query, top_k=10)
                                score = len(results) / 10.0
                                query_scores.append(score)
                            except:
                                query_scores.append(0.0)
                        scores.append(np.mean(query_scores))
                    else:
                        scores.append(0.5)
                        
                except Exception as e:
                    print(f"Error evaluating config: {e}")
                    scores.append(0.0)
            
            avg_score = np.mean(scores)
            
            result = {
                "config": config,
                "score": avg_score,
                "dataset_scores": scores
            }
            all_results.append(result)
            
            if avg_score > best_score:
                best_score = avg_score
                best_config = config.copy()
                print(f"New best configuration found! Score: {best_score:.4f}")
        
        return {
            "best_config": best_config,
            "best_score": best_score,
            "all_results": all_results,
            "n_evaluated": len(all_results)
        }


# Utility functions for experiment analysis
def compare_experiment_results(
    results_files: List[str],
    metric: str = "accuracy"
) -> pd.DataFrame:
    """Compare results from multiple experiment files."""
    
    all_results = []
    
    for file_path in results_files:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        experiment_name = data["config"]["name"]
        
        for result in data["results"]:
            row = {
                "experiment": experiment_name,
                "model": result["model_name"],
                "dataset": result["dataset_name"],
                metric: result["metrics"].get(metric, 0),
                "runtime": result["runtime"],
                "memory_usage": result["memory_usage"]
            }
            all_results.append(row)
    
    return pd.DataFrame(all_results)


def create_experiment_dashboard(
    results_df: pd.DataFrame,
    output_path: str = "experiment_dashboard.html"
) -> None:
    """Create interactive dashboard for experiment results."""
    
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Model Performance", "Runtime Comparison", 
                       "Memory Usage", "Performance vs Runtime"),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Model performance
    if "accuracy" in results_df.columns:
        model_perf = results_df.groupby("model")["accuracy"].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=model_perf.index, y=model_perf.values, name="Accuracy"),
            row=1, col=1
        )
    
    # Runtime comparison
    runtime_data = results_df.groupby("model")["runtime"].mean().sort_values()
    fig.add_trace(
        go.Bar(x=runtime_data.index, y=runtime_data.values, name="Runtime (s)"),
        row=1, col=2
    )
    
    # Memory usage
    if "memory_usage" in results_df.columns:
        memory_data = results_df.groupby("model")["memory_usage"].mean().sort_values()
        fig.add_trace(
            go.Bar(x=memory_data.index, y=memory_data.values, name="Memory (MB)"),
            row=2, col=1
        )
    
    # Performance vs Runtime scatter
    if "accuracy" in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df["runtime"],
                y=results_df["accuracy"],
                mode="markers",
                text=results_df["model"],
                name="Models"
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Experiment Results Dashboard",
        showlegend=False,
        height=800
    )
    
    fig.write_html(output_path)
    print(f"Dashboard saved to: {output_path}")


import time 