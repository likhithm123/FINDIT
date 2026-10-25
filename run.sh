#!/usr/bin/env bash
set -e

# Multimodal Vector Search Engine (ChromaDB)
cd "$(dirname "$0")"

echo "=========================================================="
echo " Starting ChromaDB Multimodal Vector Search Engine..."
echo "=========================================================="

export PYTHONPATH=.
python3 -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
