from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DocumentProcessingConfig:
    split_length: int = 150
    split_overlap: int = 50
    embedding_model: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    ranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cache_dir: Path = Path(".document_cache")
    tracking_dir: Path = Path(".tracking")
    top_k: int = 10


@dataclass
class PathConfig:
    data_dir: Path = Path("data")
    temp_dir: Path = Path("temp")
    tracking_dir: Path = Path(".tracking")
    resources_dir: Path = Path("resources")
