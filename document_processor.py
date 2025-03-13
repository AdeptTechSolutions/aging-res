import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set

import torch
from dotenv import load_dotenv
from haystack import Document, Pipeline, component
from haystack.components.converters import (
    MarkdownToDocument,
    PyPDFToDocument,
    TextFileToDocument,
)
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.utils import ComponentDevice, Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from qdrant_client import QdrantClient

from config import DocumentProcessingConfig
from pymupdf_component import PyMuPDFToDocument


@component
class MetadataEnricher:
    def __init__(self, metadata_dict: Dict[str, Dict[str, str]]):
        self.metadata_dict = metadata_dict

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        for doc in documents:
            file_path = doc.meta.get("file_path")
            if file_path:
                filename = Path(file_path).name
                metadata = self.metadata_dict.get(filename, {})

                if "author" in metadata:
                    doc.meta["author"] = metadata["author"]
                if "title" in metadata:
                    doc.meta["title"] = metadata["title"]
                if "link" in metadata:
                    doc.meta["link"] = metadata["link"]

        return {"documents": documents}


load_dotenv()


def get_device():
    """Helper function to determine the device to use."""
    if torch.cuda.is_available():
        return ComponentDevice.from_str("cuda:0")
    return None


class DocumentProcessor:
    def __init__(self, config: DocumentProcessingConfig):
        self.config = config
        self.document_store = self._initialize_document_store()
        self.metadata_dict = self._load_metadata()
        self._setup_pipeline()
        self.file_tracking_path = Path("./.tracking") / "file_tracking.json"
        self.file_tracking_path.parent.mkdir(parents=True, exist_ok=True)

    def _initialize_document_store(self):
        """Initialize QdrantDocumentStore with proper collection handling."""
        qdrant_url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        collection_name = "aging_res"

        client = QdrantClient(url=qdrant_url, api_key=api_key)
        collections = client.get_collections().collections
        collection_exists = any(col.name == collection_name for col in collections)

        store = QdrantDocumentStore(
            url=qdrant_url,
            api_key=Secret.from_env_var("QDRANT_API_KEY"),
            index=collection_name,
            embedding_dim=768,
        )

        if collection_exists:
            print(f"Using existing collection: {collection_name}")
        else:
            print(f"Creating new collection: {collection_name}")

        return store

    def _setup_pipeline(self):
        components = [
            (
                "router",
                FileTypeRouter(
                    mime_types=["text/plain", "application/pdf", "text/markdown"]
                ),
            ),
            ("text_converter", TextFileToDocument()),
            ("markdown_converter", MarkdownToDocument()),
            ("pdf_converter", PyPDFToDocument()),
            ("joiner", DocumentJoiner()),
            ("enricher", MetadataEnricher(metadata_dict=self.metadata_dict)),
            ("cleaner", DocumentCleaner()),
            (
                "splitter",
                DocumentSplitter(
                    split_by="word",
                    split_length=self.config.split_length,
                    split_overlap=self.config.split_overlap,
                ),
            ),
            (
                "embedder",
                SentenceTransformersDocumentEmbedder(
                    model=self.config.embedding_model,
                    device=get_device(),
                    meta_fields_to_embed=["title", "author", "link"],
                ),
            ),
            ("writer", DocumentWriter(self.document_store)),
        ]

        self.pipeline = Pipeline()
        for name, component in components:
            self.pipeline.add_component(instance=component, name=name)

        self.pipeline.connect("router.text/plain", "text_converter.sources")
        self.pipeline.connect("router.application/pdf", "pdf_converter.sources")
        self.pipeline.connect("router.text/markdown", "markdown_converter.sources")
        self.pipeline.connect("text_converter", "joiner")
        self.pipeline.connect("pdf_converter", "joiner")
        self.pipeline.connect("markdown_converter", "joiner")
        self.pipeline.connect("joiner", "enricher")
        self.pipeline.connect("enricher", "cleaner")
        self.pipeline.connect("cleaner", "splitter")
        self.pipeline.connect("splitter", "embedder")
        self.pipeline.connect("embedder", "writer")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _load_file_tracking(self) -> Dict[str, str]:
        """Load the file tracking information from JSON."""
        if self.file_tracking_path.exists():
            with open(self.file_tracking_path, "r") as f:
                return json.load(f)
        return {}

    def _save_file_tracking(self, tracking_info: Dict[str, str]) -> None:
        """Save the file tracking information to JSON."""
        with open(self.file_tracking_path, "w") as f:
            json.dump(tracking_info, f, indent=2)

    def _get_changed_files(self, directory: Path) -> Set[Path]:
        """Compare current files with tracked files and return changed or new files."""
        current_files = {
            file_path for file_path in directory.glob("**/*") if file_path.is_file()
        }

        tracked_files = self._load_file_tracking()
        changed_files = set()

        for file_path in current_files:
            relative_path = str(file_path.relative_to(directory))
            current_hash = self._calculate_file_hash(file_path)

            if (
                relative_path not in tracked_files
                or tracked_files[relative_path] != current_hash
            ):
                changed_files.add(file_path)

        return changed_files

    def _load_metadata(self) -> Dict[str, Dict[str, str]]:
        """Load the metadata information from meta.json"""
        metadata_path = self.config.tracking_dir / "meta.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def process_json_files(self, directory: Path) -> Set[Path]:
        """Process JSON files in the directory, extract articles to text files.

        Returns a set of newly created text file paths.
        """
        json_files = list(directory.glob("*.json"))
        if not json_files:
            return set()

        tracking_info = self._load_file_tracking()
        metadata_dict = self._load_metadata()
        new_text_files = set()

        for json_file in json_files:
            try:
                print(f"Processing JSON file: {json_file}")
                with open(json_file, "r", encoding="utf-8") as f:
                    articles = json.load(f)

                base_name = json_file.stem

                for i, article in enumerate(articles):
                    title = article.get("title", f"Article_{i}")
                    link = article.get("link", "")
                    full_text = article.get("full_text", "")

                    if not full_text:
                        print(
                            f"Skipping article {i} in {json_file} - no full_text found"
                        )
                        continue

                    safe_title = re.sub(r"[^\w\s]", "", title.lower())
                    file_name = re.sub(r"\s+", "_", safe_title)

                    file_name = f"{base_name}_{i}_{file_name[:50]}.txt"

                    text_file_path = directory / file_name
                    relative_path = str(text_file_path.relative_to(directory))

                    if not text_file_path.exists():
                        print(f"Creating new text file: {text_file_path}")
                        with open(text_file_path, "w", encoding="utf-8") as f:
                            f.write(full_text)
                        new_text_files.add(text_file_path)
                    else:
                        print(f"Text file already exists: {text_file_path}")

                        if relative_path not in tracking_info:
                            new_text_files.add(text_file_path)

                    if file_name not in metadata_dict:
                        metadata_dict[file_name] = {
                            "title": title,
                            "link": link,
                        }

            except Exception as e:
                import traceback

                print(f"Error processing JSON file {json_file}: {e}")
                print(traceback.format_exc())

        if new_text_files:
            self._save_metadata(metadata_dict)

        return new_text_files

    def _save_metadata(self, metadata_dict: dict) -> None:
        """Save the metadata information to meta.json"""
        metadata_path = self.config.tracking_dir / "meta.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=4)

    def process_documents(self, directory: Path) -> bool:
        """Process documents in the directory.

        Returns True if any documents were processed."""
        tracking_info = self._load_file_tracking()
        files_to_process = set()
        processed = False

        new_text_files = self.process_json_files(directory)

        all_files = {
            f
            for f in directory.glob("**/*")
            if f.is_file() and not f.name.endswith(".json")
        }

        for file_path in all_files:
            relative_path = str(file_path.relative_to(directory))

            if relative_path not in tracking_info or tracking_info[
                relative_path
            ] != self._calculate_file_hash(file_path):
                files_to_process.add(file_path)

        files_to_process.update(new_text_files)

        if files_to_process:
            print(
                f"Processing {len(files_to_process)} files: {[f.name for f in files_to_process]}"
            )
            try:
                self.pipeline.run({"router": {"sources": list(files_to_process)}})

                for file_path in files_to_process:
                    relative_path = str(file_path.relative_to(directory))
                    tracking_info[relative_path] = self._calculate_file_hash(file_path)

                self._save_file_tracking(tracking_info)
                processed = True
            except Exception as e:
                import traceback

                print(f"Error processing files: {e}")
                print(traceback.format_exc())
        else:
            print("No files need processing")

        return processed

    @property
    def store(self):
        return self.document_store
