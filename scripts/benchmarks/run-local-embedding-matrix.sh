#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

uv run python hindsight-dev/benchmarks/embeddings/local_embedding_matrix.py "$@"
