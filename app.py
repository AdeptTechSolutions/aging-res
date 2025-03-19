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
        is_pdf = filename.lower().endswith(".pdf")

        metadata = st.session_state.all_metadata.get(filename, {})
        title = metadata.get("title", filename)

        source_info = f"**Title:** {title}  \n"

        if "author" in metadata:
            source_info += f"**Author:** {metadata['author']}  \n"

        if "link" in metadata:
            source_info += f"**Source Link:** [View Original]({metadata['link']})  \n"

        if is_pdf and source.get("page", 0) > 0:
            source_info += f"**Page:** {source['page']}  \n"

        source_info += f"**Relevance Score:** {source['score']:.4f}"

        st.info(source_info)

        with st.expander("🔍 View Context", expanded=False):
            st.markdown(source["content"])

            if is_pdf and source.get("page", 0) > 0:
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


def render_sidebar():
    with st.sidebar:
        col1, col2, col3 = st.columns([0.3, 2, 0.3])
        with col2:
            st.image("resources/logo.png", use_container_width=True)

        st.markdown(
            "<h3 style='text-align: center;'>Eldercare is chaos.<br>We make it easier.</h3>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            """
            Welcome to Caregiver Basecamp—your guide for surviving the wilds of eldercare. 
            
            Caring for aging parents feels like trekking through unknown terrain with a faulty map. 
            We've got your back. Just type your question, hit search, and get straight-up answers 
            from trusted sources—no fluff, no detours, just the essentials to keep you on course.
            """
        )

        st.markdown("#### Resources include:")
        st.markdown(
            """
        - Articles from the most trusted sources
        - Insights from authority websites
        - Best selling books on aging and caregiving
        - Practical guides and advice from experts
        """
        )


def main():
    st.set_page_config(
        page_title="Eldercare Hero Basecamp",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    docs_processed = initialize_session()

    if not st.session_state.initialized:
        st.warning("Initializing application...")
        st.stop()

    col1, col2, col3 = st.columns([0.825, 0.825, 0.825])
    with col2:
        right_image = Image.open("resources/header_image_updated.png")
        st.image(right_image, use_container_width=True)

    render_sidebar()

    st.markdown(
        """
        <div style="text-align: center;">
        Simplifying caregiving with real-time guidance & support.<br>
        What's weighing on you? Need to vent or find guidance?<br>
        Ask your question, and we'll deliver reliable answers from trusted sources. Caregiving is tough—we've got your back.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    with st.form("query_form"):
        query = st.text_input(
            "What's on your mind?",
            placeholder="Need to vent? Wondering how to navigate Medicare? Need meal planning tips?",
        )
        submit_button = st.form_submit_button("🔍 Search")

    if submit_button and query:
        with st.spinner("Searching through resources..."):
            query_engine = QueryEngine(
                st.session_state.processor.store, st.session_state.doc_config
            )
            results = query_engine.query(query, None)

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
