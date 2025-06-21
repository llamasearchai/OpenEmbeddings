"""Comprehensive evaluation harness for embedding models.

This module integrates with the Massive Text Embedding Benchmark (MTEB)
for standardized evaluation of embedding models.

Author: Nik Jois <nikjois@llamasearch.ai>
"""
import logging
from typing import Dict, List, Optional, Union
import numpy as np
from tqdm import tqdm

try:
    from mteb import MTEB
    from mteb.evaluation.evaluators import DRESEvaluator
    _MTEB_AVAILABLE = True
except ImportError:
    _MTEB_AVAILABLE = False

from openembeddings.models.dense_embedder import DenseEmbedder

logger = logging.getLogger(__name__)

class EmbeddingModelWrapper:
    """Wrapper to make our embedder compatible with MTEB."""
    def __init__(self, embedder: DenseEmbedder, batch_size: int = 32):
        self.embedder = embedder
        self.batch_size = batch_size
        
    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """Encode sentences into embeddings."""
        return self.embedder.embed(sentences, batch_size=batch_size)
        
    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        """Encode corpus for retrieval tasks."""
        texts = [doc["text"] for doc in corpus]
        return self.embedder.embed(texts, batch_size=self.batch_size)

def evaluate_with_mteb(
    model: DenseEmbedder,
    task_types: Optional[List[str]] = None,
    task_categories: Optional[List[str]] = None,
    task_langs: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    output_folder: str = "results",
    batch_size: int = 32,
    verbosity: int = 1
) -> Dict[str, Dict[str, float]]:
    """Evaluate embedding model using MTEB.
    
    Args:
        model: DenseEmbedder instance
        task_types: List of task types to evaluate (e.g. ['Retrieval', 'Clustering'])
        task_categories: List of task categories (e.g. ['s2s', 'p2p'])
        task_langs: List of languages to evaluate on (e.g. ['en'])
        tasks: Specific tasks to evaluate (overrides types/categories)
        output_folder: Path to store results
        batch_size: Batch size for encoding
        verbosity: Verbosity level (0: silent, 1: progress, 2: debug)
        
    Returns:
        Dictionary of task_name -> metric_name -> score
    """
    if not _MTEB_AVAILABLE:
        raise ImportError("MTEB not installed. Run `pip install mteb`")
        
    # Create model wrapper
    model_wrapper = EmbeddingModelWrapper(model, batch_size=batch_size)
    
    # Configure evaluation
    evaluation = MTEB(
        task_types=task_types,
        task_categories=task_categories,
        task_langs=task_langs,
        tasks=tasks
    )
    
    # Run evaluation
    results = evaluation.run(
        model_wrapper,
        output_folder=output_folder,
        batch_size=batch_size,
        verbosity=verbosity
    )
    
    # Format results
    formatted_results = {}
    for task in evaluation.tasks:
        task_name = task.description["name"]
        formatted_results[task_name] = {
            metric: score for metric, score in task._evaluation_results.items()
        }
        
    return formatted_results

def visualize_results(results: Dict[str, Dict[str, float]], output_path: str = "results.png"):
    """Visualize evaluation results as a radar chart."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from math import pi
    
    # Prepare data
    metrics = set()
    for task_results in results.values():
        metrics.update(task_results.keys())
        
    # Create dataframe
    df = pd.DataFrame(index=list(metrics))
    for task_name, task_results in results.items():
        for metric, score in task_results.items():
            df.loc[metric, task_name] = score
            
    # Fill missing values
    df = df.fillna(0)
    
    # Normalize scores (0-1)
    df = (df - df.min()) / (df.max() - df.min())
    
    # Plot radar chart
    categories = list(df.index)
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'polar': True})
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot each task
    for task in df.columns:
        values = df[task].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=task)
        ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("MTEB Evaluation Results", size=20)
    plt.savefig(output_path)
    plt.close() 