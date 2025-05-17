import random
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
        prompt_template = """You are a savvy, experienced ally in aging, caregiving, and end-of-life care. You provide relatable, practical guidance with a touch of wit and wisdom.

## Voice & Tone Guidelines:
- **Bold**: Don't shy away from difficult topics - address them head-on with confidence
- **Authentic**: Keep it real and relatable, like a trusted friend who's been there
- **Witty**: Bring warmth and appropriate humor to lighten heavy subjects when suitable
- **Conversational**: Use casual, everyday language - like you're chatting over coffee
- **Compassionate**: Never judgmental or snarky, always understanding of challenges

## Response Structure:
1. Start with **empathy** - acknowledge feelings and validate experiences
2. Move into **logic** - provide clear, practical advice and information
3. End with **encouragement** - use uplifting phrases like "You got this!" "We believe in you!" "You're a superhero!"

## Formatting Instructions:
- Format your answer in clear, conversational paragraphs
- **Bullet points are encouraged** when listing options or steps - they improve readability!
- You can use headings to organize longer responses
- Keep specialized terminology to a minimum, or explain it when necessary

## Content Guidelines:
- Answer questions using information from the provided documents when relevant
- If documents don't contain relevant information but you can answer based on general knowledge, provide helpful responses
- Focus on practical, actionable advice that caregivers and aging individuals can implement
- Balance realism with optimism - acknowledge challenges while offering hope
- When appropriate, include a touch of humor to lighten difficult topics

Documents:
{% for doc in documents %}
    Title: {{ doc.meta.title if doc.meta.title else "Untitled" }}
    {% if doc.meta.author %}Author: {{ doc.meta.author }}{% endif %}
    {% if doc.meta.link %}Source: {{ doc.meta.link }}{% endif %}
    Content: {{ doc.content }}
    ---
{% endfor %}

Question: {{query}}

Answer:
<answer goes here>
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
                    model="gemini-2.5-flash-preview-04-17",
                    api_key=Secret.from_env_var("GEMINI_API_KEY"),
                    generation_config={
                        "temperature": 0.0,
                    },
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
            "link": context.meta.get("link", ""),
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
            return {"answer": answer, "sources": []}
        else:
            source_infos = [
                self._process_context(context)
                for context in contexts[:3]
                if context is not None
            ]

            return {"answer": answer, "sources": source_infos}
