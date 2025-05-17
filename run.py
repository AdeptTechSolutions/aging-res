# run.py

import os
import sys
import time
import json
from pathlib import Path
from typing import List
import traceback

from dotenv import load_dotenv
from document_processor import DocumentProcessor
from config import DocumentProcessingConfig, PathConfig
from qdrant_client import QdrantClient
from haystack import Document, Pipeline
from haystack.components.converters import TextFileToDocument, PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.utils import Secret, ComponentDevice
import torch

def get_device():
    """Helper function to determine the device to use."""
    if torch.cuda.is_available():
        return ComponentDevice.from_str("cuda:0")
    return None

def process_in_batches(files: List[Path], batch_size: int, path_config: PathConfig, doc_config: DocumentProcessingConfig):
    """Process files in batches to avoid memory issues and provide better progress updates."""

    qdrant_url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    collection_name = "aging_res"

    document_store = QdrantDocumentStore(
        url=qdrant_url,
        api_key=Secret.from_token(api_key),
        index=collection_name,
        embedding_dim=768,
    )

    text_converter = TextFileToDocument()
    pdf_converter = PyPDFToDocument()
    cleaner = DocumentCleaner()
    splitter = DocumentSplitter(
        split_by="word",
        split_length=doc_config.split_length,
        split_overlap=doc_config.split_overlap,
    )
    embedder = SentenceTransformersDocumentEmbedder(
        model=doc_config.embedding_model,
        device=get_device(),
    )

    print("Warming up the embedding model...")
    embedder.warm_up()
    print("Embedding model loaded successfully")

    total_files = len(files)
    file_tracking = {}
    success_count = 0

    print(f"Processing {total_files} files in batches of {batch_size}")

    for i in range(0, total_files, batch_size):
        batch_files = files[i:i+batch_size]
        print(f"\nBatch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}: Processing {len(batch_files)} files")

        try:

            print(f"  Converting files to documents...")
            documents = []
            for file_path in batch_files:
                try:
                    if file_path.suffix.lower() == '.pdf':

                        converted_docs = pdf_converter.run(sources=[file_path])["documents"]
                    else:

                        converted_docs = text_converter.run(sources=[file_path])["documents"]

                    documents.extend(converted_docs)
                    print(f"  ✓ Converted {file_path.name}")

                    file_tracking[str(file_path.relative_to(path_config.data_dir))] = calculate_file_hash(file_path)

                except Exception as e:
                    print(f"  ✗ Failed to convert {file_path.name}: {str(e)}")

            if not documents:
                print("  No documents were converted in this batch. Skipping.")
                continue

            print(f"  Cleaning {len(documents)} documents...")
            documents = cleaner.run(documents=documents)["documents"]

            print(f"  Splitting documents...")
            documents = splitter.run(documents=documents)["documents"]
            print(f"  Split into {len(documents)} chunks")

            print(f"  Embedding {len(documents)} document chunks (this may take some time)...")
            start_time = time.time()
            documents = embedder.run(documents=documents)["documents"]
            embed_time = time.time() - start_time
            print(f"  ✓ Embedding completed in {embed_time:.2f} seconds")

            print(f"  Storing documents in Qdrant...")
            start_time = time.time()
            document_store.write_documents(documents)
            store_time = time.time() - start_time
            print(f"  ✓ Storage completed in {store_time:.2f} seconds")

            tracking_path = doc_config.tracking_dir / "file_tracking.json"
            save_file_tracking(tracking_path, file_tracking)

            print(f"  ✓ Batch {i//batch_size + 1} completed successfully!")
            success_count += 1

        except Exception as e:
            print(f"  ✗ Error processing batch {i//batch_size + 1}: {str(e)}")
            print(traceback.format_exc())

            if file_tracking:
                tracking_path = doc_config.tracking_dir / "file_tracking.json"
                save_file_tracking(tracking_path, file_tracking)

    tracking_path = doc_config.tracking_dir / "file_tracking.json"
    if file_tracking:
        save_file_tracking(tracking_path, file_tracking)
        print(f"\nTracking information saved to {tracking_path}")
    else:
        print("\nNo files were successfully processed, no tracking information saved.")

    return success_count > 0

def calculate_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    import hashlib
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def save_file_tracking(file_path: Path, tracking_info: dict) -> None:
    """Save the file tracking information to JSON."""
    try:
        with open(file_path, "w") as f:
            json.dump(tracking_info, f, indent=2)
        print(f"  Saved tracking information for {len(tracking_info)} files")
    except Exception as e:
        print(f"  ⚠️ Error saving tracking information: {str(e)}")

def main():

    load_dotenv()

    print("Starting vector database regeneration...")

    path_config = PathConfig()
    doc_config = DocumentProcessingConfig()

    for dir_path in [
        path_config.data_dir,
        path_config.temp_dir,
        doc_config.cache_dir,
        doc_config.tracking_dir,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    qdrant_url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    collection_name = "aging_res"  

    if qdrant_url and api_key:
        print("Connecting to Qdrant...")
        client = QdrantClient(url=qdrant_url, api_key=api_key)
        collections = client.get_collections().collections
        collection_exists = any(col.name == collection_name for col in collections)

        if collection_exists:
            print(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
            print(f"Collection {collection_name} deleted successfully")

            print(f"Creating collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config={"size": 768, "distance": "Cosine"}
            )
            print(f"Collection {collection_name} created successfully")
    else:
        print("Warning: QDRANT_URL or QDRANT_API_KEY environment variables not set.")
        sys.exit(1)

    file_tracking_path = doc_config.tracking_dir / "file_tracking.json"
    if file_tracking_path.exists():
        print(f"Removing existing file tracking: {file_tracking_path}")
        file_tracking_path.unlink()

    print("Collecting files to process...")
    files_to_process = []
    for file_path in path_config.data_dir.glob("**/*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.pdf']:
            files_to_process.append(file_path)

    print(f"Found {len(files_to_process)} files to process")

    success = process_in_batches(files_to_process, batch_size=10, path_config=path_config, doc_config=doc_config)

    if success:
        print("\n✅ Vector database regenerated successfully!")
    else:
        print("\n❌ Vector database regeneration failed.")

if __name__ == "__main__":
    main()