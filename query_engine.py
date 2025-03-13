from pathlib import Path
from typing import Any, Dict

import torch
from haystack import Pipeline
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.utils import ComponentDevice, Secret
from haystack_integrations.components.generators.google_ai import (
    GoogleAIGeminiGenerator,
)
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

from config import DocumentProcessingConfig


def get_device():
    """Helper function to determine the device to use."""
    if torch.cuda.is_available():
        return ComponentDevice.from_str("cuda:0")
    return None


class QueryEngine:
    def __init__(self, document_store, config: DocumentProcessingConfig):
        self.config = config
        self.document_store = document_store
        self._setup_pipeline()

    def _setup_pipeline(self):
        prompt_template = """
        You are an expert in Islamic texts and literature.
        You provide comprehensive and detailed answers based on the following context. Your response must be at least one full page in length (at least 500 words or more).


        Instructions:
        - Answer the question thoroughly and in detail using the information available in the provided documents.
        - Provide comprehensive explanations that include relevant historical context, interpretations, and significance.
        - Explore different perspectives or interpretations when present in the source materials.
        - Structure your answer with clear sections when addressing complex topics.
        - Include specific quotations from the texts when directly relevant to the question.
        - Explain specialized terminology or concepts that may be unfamiliar to general readers.
        - Do not use information outside of the provided documents.
        - If no relevant information is found, state that directly.
        - When referencing specific texts, include both the title and author.
        - Always cite your sources clearly, specifying which document each piece of information comes from.
        - Ensure that your response is formatted and detailed enough to completely fill one full page.

        Documents:
        {% for doc in documents %}
            Title: {{ doc.meta.title }}
            Author: {{ doc.meta.author }}
            Content: {{ doc.content }}
            ---
        {% endfor %}

        Question: {{query}}

        Detailed Answer:
        """

        components = [
            (
                "embedder",
                SentenceTransformersTextEmbedder(
                    model=self.config.embedding_model,
                    device=get_device(),
                ),
            ),
            (
                "retriever",
                QdrantEmbeddingRetriever(
                    document_store=self.document_store, top_k=self.config.top_k
                ),
            ),
            (
                "ranker",
                TransformersSimilarityRanker(
                    model=self.config.ranker_model, top_k=self.config.top_k
                ),
            ),
            ("prompt_builder", PromptBuilder(template=prompt_template)),
            (
                "llm",
                GoogleAIGeminiGenerator(
                    model="gemini-2.0-flash",
                    api_key=Secret.from_env_var("GEMINI_API_KEY"),
                ),
            ),
            ("answer_builder", AnswerBuilder()),
        ]

        self.pipeline = Pipeline()
        for name, component in components:
            self.pipeline.add_component(instance=component, name=name)

        self.pipeline.connect("embedder.embedding", "retriever.query_embedding")
        self.pipeline.connect("retriever.documents", "ranker.documents")
        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder", "llm")
        self.pipeline.connect("llm", "answer_builder")
        self.pipeline.connect("retriever", "answer_builder.documents")

    def _process_context(self, context) -> Dict[str, Any]:
        """Process a single context document into a source information dictionary."""
        if not context:
            return None

        return {
            "source": Path(context.meta["file_path"]).name,
            "page": context.meta.get("page_number", 1),
            "score": context.score if hasattr(context, "score") else 0.0,
            "content": context.content,
            "title": context.meta.get("title", ""),
            "author": context.meta.get("author", ""),
            "language": context.meta.get("language", ""),
        }

    def query(self, query: str, filters: dict = None) -> Dict[str, Any]:
        result = self.pipeline.run(
            {
                "embedder": {"text": query},
                "retriever": {"filters": filters},
                "ranker": {"query": query, "top_k": self.config.top_k},
                "prompt_builder": {"query": query},
                "answer_builder": {"query": query},
            }
        )

        answer = result["answer_builder"]["answers"][0].data
        contexts = result["ranker"]["documents"]

        if not contexts:
            return {"answer": "No relevant information found.", "sources": []}
        else:
            source_infos = [
                self._process_context(context)
                for context in contexts[:3]
                if context is not None
            ]

            return {"answer": answer, "sources": source_infos}
