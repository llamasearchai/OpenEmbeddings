from setuptools import setup, find_packages

setup(
    name="openembeddings",
    version="0.1.0",
    description="Lightweight embedding & retrieval utilities for unit-testing.",
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "numpy>=1.21.0,<2.0",
        "rank_bm25>=0.2.2",
        "torch>=2.0.0",
        "typer>=0.12.3",
        "rich>=13.7.0",
        "sentence-transformers>=2.2.0",
        "nltk>=3.8.0",
        "faiss-cpu>=1.7.4",
        # Advanced ML and Research Libraries
        "datasets>=2.14.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "evaluate>=0.4.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "black>=24.0.0",
            "flake8>=7.0.0",
            "pytest>=8.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "openembeddings=openembeddings.cli:_main",
        ],
    },
)
