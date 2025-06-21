"""Entry point for the OpenEmbeddings package when run as a module.

This allows the package to be executed as:
    python -m openembeddings

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from .cli import _main

if __name__ == "__main__":
    _main() 