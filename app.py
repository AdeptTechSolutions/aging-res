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
            "<h3 style='text-align: center; font-family: Open Sans, sans-serif; font-weight: 700;'>Eldercare is chaos.<br>We make it easier.</h3>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            "<div class='sidebar-heading'>ℹ️ About</div>", unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class='sidebar-text'>
            Welcome to Caregiver Basecamp—your guide for surviving the wilds of eldercare.<br><br>
            Caring for aging parents feels like trekking through unknown terrain with a faulty map. 
            We've got your back. Just type your question, hit search, and get straight-up answers 
            from trusted sources—no fluff, no detours, just the essentials to keep you on course.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='sidebar-subheading'>Resources include:</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <ul class='sidebar-list'>
                <li>Articles from the most trusted sources</li>
                <li>Insights from authority websites</li>
                <li>Best selling books on aging and caregiving</li>
                <li>Practical guides and advice from experts</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(
        page_title="Eldercare Hero Basecamp",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
        
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 0rem;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
            overflow-x: hidden;
        }
        
        /* Prevent horizontal scrolling */
        .main .block-container {
            overflow-x: hidden;
        }
        
        div.stImage {
            margin-top: 10px;
        }
        
        /* Main content styling */
        .main-heading {
            font-family: 'Open Sans', sans-serif;
            font-weight: 700;
            font-size: 2rem;
            line-height: 1.3;
            text-align: center;
            margin-bottom: 25px;
            color: #494a4c;
        }
        
        .sub-heading-bold {
            font-family: 'Open Sans', sans-serif;
            font-weight: 700;
            font-size: 1.3rem;
            text-align: center;
            margin-bottom: 15px;
            color: #494a4c;
        }
        
        .normal-text {
            font-family: 'Open Sans', sans-serif;
            font-weight: 400;
            font-size: 1rem;
            text-align: center;
            margin-bottom: 25px;
            color: #494a4c;
        }
        
        .closing-bold {
            font-family: 'Open Sans', sans-serif;
            font-weight: 700;
            font-size: 1.2rem;
            text-align: center;
            margin-bottom: 20px;
            color: #494a4c;
        }
        
        /* Sidebar styling */
        .sidebar-heading {
            font-family: 'Open Sans', sans-serif;
            font-weight: 700;
            font-size: 1.2rem;
            margin-bottom: 15px;
            color: #494a4c;
        }
        
        .sidebar-subheading {
            font-family: 'Open Sans', sans-serif;
            font-weight: 700;
            font-size: 1.1rem;
            margin-top: 20px;
            margin-bottom: 10px;
            color: #494a4c;
        }
        
        .sidebar-text {
            font-family: 'Open Sans', sans-serif;
            font-weight: 520;
            line-height: 1.5;
            font-size: 0.95rem;
            color: #494a4c;
        }
        
        .sidebar-list {
            font-family: 'Open Sans', sans-serif;
            font-weight: 520;
            margin-left: 20px;
            padding-left: 0;
            font-size: 0.95rem;
            color: #494a4c;
        }
        
        .sidebar .stMarkdown h3 {
            font-family: 'Open Sans', sans-serif;
            font-weight: 600;
            text-align: center;
            color: #494a4c;
        }
        
        .sidebar .stMarkdown p, .sidebar .stMarkdown li {
            font-family: 'Open Sans', sans-serif;
            font-weight: 400;
            color: #494a4c;
        }
        
        .sidebar .stMarkdown h4 {
            font-family: 'Open Sans', sans-serif;
            font-weight: 600;
            margin-top: 20px;
            color: #494a4c;
        }
        
        /* Apply Open Sans to all text in the app */
        html, body, [class*="css"] {
            font-family: 'Open Sans', sans-serif;
            color: #494a4c;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    docs_processed = initialize_session()

    if not st.session_state.initialized:
        st.warning("Initializing application...")
        st.stop()

    col1, col2, col3 = st.columns([1, 0.5, 1])
    with col2:
        right_image = Image.open("resources/header_image.png")
        st.image(right_image, use_container_width=True)

    render_sidebar()

    st.markdown(
        """
        <div>
            <div class="main-heading">
                Simplifying caregiving with<br>real-time guidance & support
            </div>
            <div class="sub-heading-bold">
                Need to find guidance, a resource or just vent?
            </div>
            <div class="normal-text">
                Ask your question, and we'll deliver reliable answers from<br>trusted sources in eldercare.
            </div>
            <div class="closing-bold">
                Caregiving is tough—we've got your back.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    with st.form("query_form"):
        query = st.text_area(
            "What's on your mind?",
            placeholder="Need to vent? Wondering how to navigate Medicare? Need meal planning tips?",
            height=100  # Adjust height as needed
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            submit_button = st.form_submit_button("🔍 Search")
        with col2:
            feedback_url = "https://pamela988959.typeform.com/to/mqVIFBr6"
            st.markdown(f'<a href="{feedback_url}" target="_blank"><button style="width:100%;padding:10px;background-color:#FF4B4B;color:white;border:none;border-radius:4px;cursor:pointer;">💭 Give Feedback</button></a>', unsafe_allow_html=True)

    if submit_button and query:
        with st.spinner("Searching through resources..."):
            query_engine = QueryEngine(
                st.session_state.processor.store, st.session_state.doc_config
            )
            results = query_engine.query(query, None)

            st.markdown("#### 📝 Answer")
            st.markdown(results["answer"])

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
