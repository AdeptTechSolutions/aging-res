import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback, Select, TruSession
from trulens.providers.openai import OpenAI

from config import DocumentProcessingConfig, PathConfig
from document_processor import DocumentProcessor
from query_engine import QueryEngine

load_dotenv()

session = TruSession()
session.reset_database()

provider = OpenAI(model_engine="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))


class InstrumentedQueryEngine(QueryEngine):
    @instrument
    def query(self, query: str):
        return super().query(query)


f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on_output()
    .on(Select.RecordCalls.query.args.query)
)

f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on(Select.RecordCalls.query.args.query)
    .on_output()
)

f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
    .on(Select.RecordCalls.query.args.query)
    .on_output()
    .aggregate(np.mean)
)


def evaluate_system(questions: list[str]):
    path_config = PathConfig()
    doc_config = DocumentProcessingConfig()

    processor = DocumentProcessor(doc_config)
    processor.process_documents(path_config.data_dir)

    instrumented_engine = InstrumentedQueryEngine(processor.store, doc_config)

    tru_app = TruCustomApp(
        instrumented_engine,
        app_name="Islamic_Texts_QA",
        app_version="haystack",
        feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
    )

    with tru_app as recording:
        for question in questions:
            instrumented_engine.query(question)

    leaderboard = session.get_leaderboard()
    print("\nEvaluation Results:")
    print(leaderboard)

    leaderboard.to_csv("qa_evaluation_results.csv")
    print("\nResults saved to qa_evaluation_results.csv")

    return leaderboard


if __name__ == "__main__":
    evaluation_questions = [
        "What is the Sunni path?",
        "Why are pride and arrogance considered dangerous in Islam?",
        "Is tobacco-smoking permissible in Islam?",
        "What are the main pillars of Islam?",
        "How should Muslims treat their neighbors?",
    ]

    results = evaluate_system(evaluation_questions)
