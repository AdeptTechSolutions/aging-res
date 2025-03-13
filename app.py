import json
import os
from pathlib import Path

import fitz
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from config import DocumentProcessingConfig, PathConfig
from document_processor import DocumentProcessor
from query_engine import QueryEngine


def initialize_session():
    """Initialize session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.unique_authors = []
        st.session_state.unique_languages = []
        st.session_state.all_metadata = {}

        load_dotenv()

        path_config = PathConfig()
        doc_config = DocumentProcessingConfig()

        for dir_path in [
            path_config.data_dir,
            path_config.temp_dir,
            doc_config.cache_dir,
            doc_config.tracking_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        processor = DocumentProcessor(doc_config)

        st.session_state.path_config = path_config
        st.session_state.doc_config = doc_config
        st.session_state.processor = processor

        documents_processed = processor.process_documents(path_config.data_dir)

        metadata_path = doc_config.tracking_dir / "meta.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                st.session_state.all_metadata = metadata
                st.session_state.unique_authors = sorted(
                    set(item["author"] for item in metadata.values())
                )
                st.session_state.unique_languages = sorted(
                    set(item["language"] for item in metadata.values())
                )

        st.session_state.initialized = True
        return documents_processed


def display_pdf_page(pdf_path: Path, page_num: int, temp_dir: Path):
    try:
        pdf_path_str = str(pdf_path).replace("\\", "/")

        if not Path(pdf_path_str).exists():
            st.error(f"PDF file not found: {pdf_path_str}")
            return

        doc = fitz.open(pdf_path_str)
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

        img_path = temp_dir / f"temp_page_{page_num}.png"
        pix.save(str(img_path))

        st.image(str(img_path), width=600)

        img_path.unlink()
        doc.close()
    except Exception as e:
        st.error(f"Error displaying PDF page: {str(e)}")
        import traceback

        st.sidebar.code(traceback.format_exc())


def get_pdf_download_link(filename: str) -> str:
    """Generate the download link for the full PDF.

    Args:
        filename: Just the filename without path prefixes
    """
    base_url = "https://raw.githubusercontent.com/AdeptTechSolutions/haystack-rag/refs/heads/main/data/"
    clean_filename = filename.split("/")[-1] if "/" in filename else filename
    return f"{base_url}{clean_filename}"


def display_source_information(source, path_config):
    """Display source information in an organized manner"""
    if source:
        st.markdown("### 📄 Source Details")

        source_path = source["source"].replace("\\", "/")
        filename = source_path.split("/")[-1] if "/" in source_path else source_path

        metadata = st.session_state.all_metadata.get(filename, {})
        title = metadata.get("title", filename)
        author = metadata.get("author", "Unknown")
        language = metadata.get("language", "Unknown")

        st.info(
            f"""
            **Title:** {title}  
            **Author:** {author}  
            **Language:** {language}  
            **Page:** {source['page']}  
            **Relevance Score:** {source['score']:.4f}
            """
        )

        with st.expander("🔍 View Context", expanded=False):
            st.markdown(source["content"])

            st.markdown("#### 📑 Page Preview")
            pdf_path = path_config.data_dir / filename

            col1, col2, col3 = st.columns([1, 1.5, 1])
            with col2:
                display_pdf_page(
                    pdf_path,
                    source["page"],
                    path_config.temp_dir,
                )

                download_link = get_pdf_download_link(filename)
                st.markdown(f"📥  [Download Complete PDF]({download_link})")
    else:
        st.warning("⚠️ No source information available for this result.")


def main():
    st.set_page_config(
        page_title="Aging RAG",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    docs_processed = initialize_session()

    if not st.session_state.initialized:
        st.warning("Initializing application...")
        st.stop()

    col1, col2, col3 = st.columns([1, 0.4, 1])
    with col2:
        logo = Image.open("resources/logo.png")
        st.image(logo, use_container_width=True)

    with st.sidebar:
        st.markdown("## 📚 Filters")

        selected_authors = st.multiselect(
            "Select authors:",
            options=st.session_state.unique_authors,
            key="author_filter",
            help="Filter results by specific authors.",
        )

        if not selected_authors:
            st.warning("Note: All sources would be used if no author is selected")

        selected_languages = st.multiselect(
            "Select languages:",
            options=st.session_state.unique_languages,
            key="language_filter",
            help="Filter results by languages.",
            default=["English"],
        )

        # if docs_processed:
        #     st.success("New or modified documents have been processed!")
        # else:
        #     st.info("No document changes detected.")

        st.markdown("### ℹ️ About")
        st.markdown(
            """
            This application allows you to search through Islamic texts 
            and receive relevant information from verified sources.
            
            Simply enter your question and use the `🔍 Search` button
            to get answers. You can also filter the results by authors and languages.
            """
        )

    st.markdown("")

    with st.form("query_form"):
        query = st.text_input(
            "Enter your question:",
            placeholder="Ask a question about the Islamic texts...",
        )
        submit_button = st.form_submit_button("🔍 Search")

    if submit_button and query:
        with st.spinner("Searching through texts..."):
            filters = []
            if selected_authors:
                filters.append(
                    {
                        "field": "meta.author",
                        "operator": "in",
                        "value": selected_authors,
                    }
                )
            if selected_languages:
                filters.append(
                    {
                        "field": "meta.language",
                        "operator": "in",
                        "value": selected_languages,
                    }
                )

            combined_filters = None
            if filters:
                if len(filters) > 1:
                    combined_filters = {
                        "operator": "AND",
                        "conditions": filters,
                    }
                else:
                    combined_filters = filters[0]

            query_engine = QueryEngine(
                st.session_state.processor.store, st.session_state.doc_config
            )
            results = query_engine.query(query, combined_filters)

            st.markdown("#### 📝 Answer")
            st.write(results["answer"])

            st.markdown("#### 📚 Sources")

            if not results["sources"]:
                st.warning("⚠️ No sources found for this query.")
            else:
                tabs = st.tabs(["Source 1", "Source 2", "Source 3"])

                st.warning(
                    "⚠️ Please refer to the original sources for verification. AI-generated responses may be incomplete or incorrect."
                )

                for idx, (tab, source) in enumerate(zip(tabs, results["sources"][:3])):
                    with tab:
                        display_source_information(source, st.session_state.path_config)


if __name__ == "__main__":
    main()
