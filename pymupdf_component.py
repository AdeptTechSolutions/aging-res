import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

from haystack import Document, component, default_to_dict
from haystack.components.converters.utils import (
    get_bytestream_from_source,
    normalize_metadata,
)
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install PyMuPDF'") as pymupdf_import:
    import fitz


logger = logging.getLogger(__name__)


class PyMuPDFConverter(Protocol):
    """
    A protocol that defines a converter which takes a fitz.Document object and converts it into a Document object.
    """

    def convert(self, document: "fitz.Document") -> Document: ...


class DefaultConverter:
    """
    The default converter class that extracts text from a fitz.Document object's pages and returns a Document.
    """

    def convert(self, document: "fitz.Document") -> Document:
        """Extract text from the PDF and return a Document object with the text content."""
        text = []
        for page in document:
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (b[1], b[0]))
            for b in blocks:
                text.append(b[4])
        return Document(content="\n".join(text))


CONVERTERS_REGISTRY: Dict[str, PyMuPDFConverter] = {"default": DefaultConverter()}


@component
class PyMuPDFToDocument:
    """
    Converts PDF files to Document objects using PyMuPDF.
    """

    def __init__(self, converter_name: str = "default"):
        """
        Initializes the PyMuPDFToDocument component with an optional custom converter.
        """
        pymupdf_import.check()

        try:
            converter = CONVERTERS_REGISTRY[converter_name]
        except KeyError:
            msg = f"Invalid converter_name: {converter_name}.\n Available converters: {list(CONVERTERS_REGISTRY.keys())}"
            raise ValueError(msg) from KeyError
        self.converter_name = converter_name
        self._converter: PyMuPDFConverter = converter

    def to_dict(self):
        return default_to_dict(self, converter_name=self.converter_name)

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts a list of PDF sources into Document objects using the configured converter.
        """
        documents = []
        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read %s. Skipping it. Error: %s", source, e)
                continue
            try:
                # Use PyMuPDF to read the PDF
                pdf_document = fitz.open("pdf", bytestream.data)
                document = self._converter.convert(pdf_document)
                pdf_document.close()
            except Exception as e:
                logger.warning(
                    "Could not read %s and convert it to Document, skipping. %s",
                    source,
                    e,
                )
                continue

            merged_metadata = {**bytestream.meta, **metadata}
            document.meta = merged_metadata
            documents.append(document)

        return {"documents": documents}
