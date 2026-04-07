#!/usr/bin/env python3
"""Helper: Download AI model via model_manager."""
import os, sys

# Set HF cache to /tmp on CI runners
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_cache/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache/transformers")

sys.path.insert(0, os.getcwd())
from app.model_manager import model_manager

model_url = os.environ["MODEL_ID"]
model_source = os.environ.get("MODEL_SOURCE", "huggingface")
hf_token = os.environ.get("HF_TOKEN") or None

try:
    info = model_manager.download_model(
        model_id=model_url,
        source=model_source,
        hf_token=hf_token,
    )
    print(f"OK:{info.name}:{info.size_mb}:{info.type}")
except Exception as e:
    print(f"FAIL:{e}", file=sys.stderr)
    sys.exit(1)
