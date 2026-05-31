"""LongFact 本地配置与默认值。

这里集中管理数据集、模型名和 HuggingFace 缓存相关设置，方便在本地机器上一次配置后复用。
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_local_env_file() -> None:
    """Load simple KEY=VALUE pairs from .env.local if it exists.

    Existing environment variables are preserved so shell-level overrides still win.
    """
    env_path = PROJECT_ROOT / ".env.local"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env_file()

DEFAULT_DATA_DIR = Path(os.getenv("LONGFACT_DATA_DIR", PROJECT_ROOT / "data" / "cache"))
DEFAULT_OUTPUT_DIR = Path(os.getenv("LONGFACT_OUTPUT_DIR", PROJECT_ROOT / "results"))
DEFAULT_GOVREPORT_DATASET = os.getenv("LONGFACT_GOVREPORT_DATASET", "ccdv/govreport-summarization")
DEFAULT_GOVREPORT_SPLIT = os.getenv("LONGFACT_GOVREPORT_SPLIT", "validation")
DEFAULT_SUMMARIZER_MODEL = os.getenv("LONGFACT_SUMMARIZER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEFAULT_NLI_MODEL = os.getenv("LONGFACT_NLI_MODEL", "facebook/bart-large-mnli")
DEFAULT_CORRECTOR_MODEL = os.getenv("LONGFACT_CORRECTOR_MODEL", DEFAULT_SUMMARIZER_MODEL)
DEFAULT_RETRIEVER_MODEL = os.getenv("LONGFACT_RETRIEVER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_USE_BM25 = os.getenv("LONGFACT_USE_BM25", "0").strip().lower() in {"1", "true", "yes", "y"}
# When debugging, it's sometimes useful to fall back to the reference `summary`
# if `document` is missing. This should be False for formal experiments.
FALLBACK_TO_SUMMARY = os.getenv("LONGFACT_FALLBACK_TO_SUMMARY", "0").strip().lower() in {"1", "true", "yes", "y"}


def ensure_local_dirs() -> None:
    """Create the common local cache/output directories if they do not exist."""
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Precision / performance toggles (project-local only)
# Options: 'auto', 'fp32', 'fp16', '8bit'
PREFERRED_PRECISION = os.getenv("LONGFACT_PREFERRED_PRECISION", os.getenv("PREFERRED_PRECISION", "auto"))
# When True, attempt to call `torch.compile` on model objects where supported
DEFAULT_TORCH_COMPILE = os.getenv("LONGFACT_TORCH_COMPILE", "0").strip().lower() in {"1", "true", "yes", "y"}

# Embedding cache directory (for Retriever). Relative to project by default.
EMBEDDING_CACHE_DIR = Path(os.getenv("LONGFACT_EMB_CACHE_DIR", PROJECT_ROOT / "data" / "emb_cache"))
EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Retriever tuning
DEFAULT_RETRIEVER_ENCODE_BATCH_SIZE = int(os.getenv("LONGFACT_RETRIEVER_ENCODE_BATCH_SIZE", "64"))
# Retriever index method: 'auto'|'flat'|'hnsw'|'ivf'
DEFAULT_RETRIEVER_INDEX_METHOD = os.getenv("LONGFACT_RETRIEVER_INDEX_METHOD", "auto")
