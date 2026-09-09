"""
app.py — Main Streamlit application for the AI College Admission Agent.

Run with:  streamlit run app.py
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import streamlit as st

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).parent))

import config
from components.ui import apply_custom_css, render_header, render_info_box
from components.chat import (
    render_chat_message,
    render_sources,
    render_agent_status,
    render_suggested_questions,
    render_error_message,
)
from components.cards import render_feature_cards, render_knowledge_base_stats
from components.sources import render_source_card
from utils.document_loader import ingest_documents, get_index_status, save_uploaded_file
from utils.validators import sanitize_query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI College Admission Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_session_state():
    """Initialise all required session state keys."""
    defaults = {
        "page": "home",
        "messages": [],           # [{role, content, agent_name, sources, timestamp}]
        "student_profile": {},
        "coordinator": None,
        "retriever": None,
        "llm": None,
        "system_status": None,    # cached on first load
        "ingestion_running": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# System initialisation (cached in session state)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_system():
    """
    Load LLM, retriever, and coordinator once per session.
    Returns (coordinator, retriever, llm, status_dict).
    """
    from llm.factory import get_llm
    from rag.retriever import RAGRetriever
    from agents.coordinator import AgentCoordinator

    status = {
        "llm_ok": False,
        "llm_msg": "",
        "index_ok": False,
        "index_msg": "",
        "provider": config.get_active_provider(),
    }

    # 1. LLM
    try:
        llm = get_llm()
        ok, msg = llm.health_check()
        status["llm_ok"] = ok
        status["llm_msg"] = msg
        if ok:
            # Warm up the model so it's loaded into Ollama's memory before the
            # first user query arrives — prevents 120 s timeout on first use.
            from llm.local import LocalLLM
            if isinstance(llm, LocalLLM):
                llm.warm_up()
        else:
            logger.warning(f"LLM health check failed: {msg}")
    except Exception as e:
        llm = None
        status["llm_msg"] = f"Failed to initialise LLM: {e}"
        logger.error(status["llm_msg"])

    # 2. Retriever
    try:
        retriever = RAGRetriever(
            top_k=config.TOP_K,
            similarity_threshold=config.SIMILARITY_THRESHOLD,
        )
        if retriever.is_ready:
            status["index_ok"] = True
            info = retriever.vector_store.info
            status["index_msg"] = (
                f"Index loaded: {info.get('num_chunks', '?')} chunks, "
                f"{info.get('num_documents', '?')} documents"
            )
        else:
            status["index_msg"] = (
                "Knowledge base not found. "
                "Please run 'python ingest.py' or use the Knowledge Base page."
            )
    except Exception as e:
        retriever = None
        status["index_msg"] = f"Retriever error: {e}"
        logger.error(status["index_msg"])

    # 3. Coordinator
    coordinator = None
    if llm and retriever:
        try:
            coordinator = AgentCoordinator(llm=llm, retriever=retriever)
        except Exception as e:
            logger.error(f"Coordinator init failed: {e}")

    return coordinator, retriever, llm, status


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    """Render the navigation sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:1.2rem 0 0.5rem;">
                <div style="font-size:2.2rem;">🎓</div>
                <div style="color:white;font-size:1.1rem;font-weight:700;margin-top:0.3rem;">
                    College Admission
                </div>
                <div style="color:#94a3b8;font-size:0.8rem;">AI Admission Agent</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        pages = {
            "🏠 Home": "home",
            "🤖 Admission Assistant": "chat",
            "📚 Knowledge Base": "knowledge",
            "ℹ️ About": "about",
        }
        for label, page_id in pages.items():
            is_active = st.session_state.page == page_id
            if st.button(
                label,
                key=f"nav_{page_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.page = page_id
                st.rerun()

        st.markdown("---")
        render_sidebar_status()
        render_student_profile_sidebar()


def render_sidebar_status():
    """Show LLM and index status in the sidebar."""
    _, _, _, status = load_system()

    st.markdown(
        "<p style='color:#94a3b8;font-size:0.75rem;font-weight:600;text-transform:uppercase;"
        "letter-spacing:0.8px;margin-bottom:0.5rem;'>System Status</p>",
        unsafe_allow_html=True,
    )

    # LLM status
    llm_icon = "🟢" if status.get("llm_ok") else "🔴"
    provider = status.get("provider", "local").upper()
    st.markdown(
        f"<div style='color:#e2e8f0;font-size:0.8rem;margin-bottom:0.3rem;'>"
        f"{llm_icon} LLM ({provider})</div>",
        unsafe_allow_html=True,
    )

    # Index status
    idx_icon = "🟢" if status.get("index_ok") else "🟡"
    st.markdown(
        f"<div style='color:#e2e8f0;font-size:0.8rem;margin-bottom:0.3rem;'>"
        f"{idx_icon} Knowledge Base</div>",
        unsafe_allow_html=True,
    )

    if not status.get("llm_ok"):
        with st.expander("⚠️ LLM Issue", expanded=False):
            st.markdown(
                f"<div style='font-size:0.78rem;color:#fca5a5;'>"
                f"{status.get('llm_msg', '')}</div>",
                unsafe_allow_html=True,
            )

    if not status.get("index_ok"):
        with st.expander("⚠️ Index Issue", expanded=False):
            st.markdown(
                f"<div style='font-size:0.78rem;color:#fde68a;'>"
                f"{status.get('index_msg', '')}</div>",
                unsafe_allow_html=True,
            )


def render_student_profile_sidebar():
    """Render the collapsible student profile form in the sidebar."""
    st.markdown("---")
    with st.expander("👤 Your Profile (Optional)", expanded=False):
        st.markdown(
            "<p style='color:#94a3b8;font-size:0.78rem;margin-bottom:0.6rem;'>"
            "Share your background for personalised recommendations.</p>",
            unsafe_allow_html=True,
        )
        profile = st.session_state.student_profile

        background = st.text_input(
            "Academic Background",
            value=profile.get("background", ""),
            placeholder="e.g., Class 12, Science",
            key="profile_background",
        )
        marks = st.text_input(
            "Marks / Percentage",
            value=profile.get("marks", ""),
            placeholder="e.g., 85%",
            key="profile_marks",
        )
        subjects = st.text_input(
            "Subjects Studied",
            value=profile.get("subjects", ""),
            placeholder="e.g., Maths, Physics, CS",
            key="profile_subjects",
        )
        interests = st.text_input(
            "Interests",
            value=profile.get("interests", ""),
            placeholder="e.g., Artificial Intelligence, Data Science",
            key="profile_interests",
        )
        budget = st.selectbox(
            "Budget Range",
            ["", "Under ₹1L/yr", "₹1L–3L/yr", "₹3L–5L/yr", "Above ₹5L/yr"],
            index=0,
            key="profile_budget",
        )
        scholarship = st.checkbox(
            "Looking for scholarship",
            value=profile.get("scholarship_needed", False),
            key="profile_scholarship",
        )

        if st.button("💾 Save Profile", key="save_profile", use_container_width=True):
            st.session_state.student_profile = {
                "background": background,
                "marks": marks,
                "subjects": subjects,
                "interests": interests,
                "budget": budget,
                "scholarship_needed": scholarship,
            }
            st.success("Profile saved!")


# ---------------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------------
def render_home_page():
    """Render the landing/home page."""
    render_header(
        title="AI College Admission Agent",
        subtitle="Personalized, reliable and intelligent admission guidance powered by RAG and Agentic AI",
        badge="🤖 Powered by IBM Granite & Agentic AI",
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "🚀 Start Admission Assistant",
            key="cta_start",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.page = "chat"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    render_feature_cards()
    st.markdown("---")

    # Quick stats
    idx_status = get_index_status()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-value">5</div>'
            '<div class="stat-label">Specialized AI Agents</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col2:
        doc_count = idx_status.get("num_documents", 0)
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value">{doc_count}</div>'
            f'<div class="stat-label">Knowledge Base Documents</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col3:
        chunk_count = idx_status.get("num_chunks", 0)
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value">{chunk_count}</div>'
            f'<div class="stat-label">Indexed Text Chunks</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    st.markdown("### How It Works")
    flow_cols = st.columns(5)
    steps = [
        ("1️⃣", "Ask a Question", "Type your admission query"),
        ("2️⃣", "Route to Agent", "Intelligent query classification"),
        ("3️⃣", "Search Knowledge Base", "RAG retrieves relevant chunks"),
        ("4️⃣", "Generate Answer", "LLM crafts a grounded response"),
        ("5️⃣", "Source Citations", "Every answer cites its source"),
    ]
    for col, (icon, title, desc) in zip(flow_cols, steps):
        with col:
            st.markdown(
                f'<div style="text-align:center;padding:0.8rem;">'
                f'<div style="font-size:1.6rem">{icon}</div>'
                f'<strong style="font-size:0.85rem">{title}</strong>'
                f'<p style="font-size:0.78rem;color:#64748b;margin-top:0.2rem;">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Page: Chat (Admission Assistant)
# ---------------------------------------------------------------------------

SUGGESTED_QUESTIONS = [
    "Which courses are suitable for me?",
    "Am I eligible for B.Tech Computer Science?",
    "What documents are required?",
    "What are the admission fees?",
    "What scholarships are available?",
    "When is the application deadline?",
]


def render_chat_page():
    """Render the main conversational interface."""
    st.markdown("## 🤖 AI Admission Assistant")
    st.markdown(
        "<p style='color:#64748b;font-size:0.9rem;margin-top:-0.5rem;'>"
        "Ask me anything about courses, eligibility, fees, scholarships, documents, or deadlines.</p>",
        unsafe_allow_html=True,
    )

    coordinator, retriever, llm, status = load_system()

    # System warnings
    if not status.get("llm_ok"):
        st.warning(
            f"⚠️ **LLM Unavailable**: {status.get('llm_msg', 'Unknown error')}  \n"
            "Please ensure Ollama is running: `ollama serve` then `ollama pull llama3.2`"
        )

    if not status.get("index_ok"):
        st.info(
            "📚 **Knowledge base is empty.**  \n"
            "Go to **Knowledge Base** page and click **Rebuild Knowledge Base**, "
            "or run `python ingest.py` in your terminal."
        )

    # Action buttons row
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col3:
        if st.button("👤 Edit Profile", use_container_width=True):
            st.info("Use the sidebar 👈 to edit your profile.")

    st.markdown("---")

    # Chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                '<div class="info-box" style="text-align:center;">'
                "👋 Hello! I'm your AI College Admission Counselor.<br>"
                "I can help with <strong>courses, eligibility, fees, scholarships, documents, and deadlines</strong>.<br>"
                "Ask me anything, or choose a suggested question below!"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.messages:
                render_chat_message(
                    role=msg["role"],
                    content=msg["content"],
                    agent_name=msg.get("agent_name"),
                )
                if msg.get("sources"):
                    render_sources(msg["sources"])

    st.markdown("---")

    # Suggested questions
    if len(st.session_state.messages) < 2:
        selected = render_suggested_questions(SUGGESTED_QUESTIONS)
        if selected:
            process_query(selected, coordinator, retriever, llm, status)
            st.rerun()

    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Ask your question...",
                placeholder="e.g., What courses are available for a student with 85% in Maths?",
                label_visibility="collapsed",
                key="chat_input",
            )
        with col2:
            submitted = st.form_submit_button("Send 🚀", use_container_width=True, type="primary")

    if submitted and user_input:
        valid, query = sanitize_query(user_input)
        if not valid:
            st.error(query)
        else:
            process_query(query, coordinator, retriever, llm, status)
            st.rerun()


def process_query(
    query: str,
    coordinator,
    retriever,
    llm,
    status: dict,
) -> None:
    """Process a query through the agent system and append to chat history."""
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "agent_name": None,
        "sources": [],
        "timestamp": time.time(),
    })

    # Guard: LLM not available
    if not status.get("llm_ok") or coordinator is None:
        error_msg = (
            "⚠️ The AI assistant is currently unavailable.\n\n"
            f"**Reason**: {status.get('llm_msg', 'LLM not configured.')}\n\n"
            "**How to fix**:\n"
            "1. Install Ollama: https://ollama.com\n"
            "2. Run: `ollama serve`\n"
            "3. Pull model: `ollama pull llama3.2`\n"
            "4. Refresh this page."
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
            "agent_name": "System",
            "sources": [],
        })
        return

    # Guard: Index not available
    if not status.get("index_ok"):
        no_index_msg = (
            "📚 **Knowledge base is empty.** I cannot answer questions without documents.\n\n"
            "**Steps to fix**:\n"
            "1. Add PDF/DOCX/TXT files to the `data/` directory\n"
            "2. Run `python ingest.py` in your terminal\n"
            "OR go to the **Knowledge Base** page and click **Rebuild Knowledge Base**."
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": no_index_msg,
            "agent_name": "System",
            "sources": [],
        })
        return

    # Show processing indicator
    with st.spinner("🤔 Thinking..."):
        status_placeholder = st.empty()

        # Route and generate
        try:
            from agents.router import QueryRouter
            router = QueryRouter()
            category = router.classify(query)
            route_desc = router.route_description(category)

            status_placeholder.markdown(
                f'<div class="processing-step">🔎 Query classified → {route_desc}</div>',
                unsafe_allow_html=True,
            )

            response, category = coordinator.process(
                query=query,
                student_profile=st.session_state.student_profile or None,
                conversation_history=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[-10:]
                    if m["role"] in ("user", "assistant")
                ],
            )
        except Exception as e:
            logger.exception(f"Agent processing error: {e}")
            response = None
            category = None

        status_placeholder.empty()

    if response is None or not response.success:
        error_text = (
            "I encountered an error while processing your question. "
            "Please try rephrasing, or check that the LLM and knowledge base are available."
        )
        if response and response.error:
            logger.error(f"Agent error: {response.error}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_text,
            "agent_name": "System",
            "sources": [],
        })
        return

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.text,
        "agent_name": response.agent_name,
        "sources": response.sources,
        "timestamp": time.time(),
    })


# ---------------------------------------------------------------------------
# Page: Knowledge Base
# ---------------------------------------------------------------------------
def render_knowledge_base_page():
    """Render the knowledge base management page."""
    st.markdown("## 📚 Knowledge Base Manager")
    st.markdown(
        "<p style='color:#64748b;font-size:0.9rem;'>Manage documents in the admission knowledge base.</p>",
        unsafe_allow_html=True,
    )

    # Stats
    idx_status = get_index_status()
    if idx_status["exists"]:
        render_knowledge_base_stats(
            {
                "num_documents": idx_status["num_documents"],
                "num_chunks": idx_status["num_chunks"],
                "dimension": idx_status["dimension"],
                "built_at": idx_status["built_at"],
            }
        )
        st.markdown(
            f'<div style="text-align:right;margin-top:0.3rem;">'
            f'<span class="badge badge-green">✅ Index Active</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="warning-box">⚠️ No knowledge base index found. '
            'Add documents below and click <strong>Rebuild Knowledge Base</strong>.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Upload section
    st.markdown("### 📤 Upload New Document")
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader(
            "Drop a PDF, DOCX, or TXT file here",
            type=["pdf", "docx", "txt"],
            help="Maximum file size: 50 MB",
            key="file_uploader",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬆️ Upload File", use_container_width=True, disabled=uploaded is None):
            if uploaded:
                ok, msg = save_uploaded_file(uploaded)
                if ok:
                    st.success(msg)
                    # Clear the Streamlit cache so the next chat reloads the retriever
                    load_system.clear()
                else:
                    st.error(msg)

    st.markdown("---")

    # Control buttons
    st.markdown("### 🔧 Index Management")
    col1, col2, col3 = st.columns(3)

    with col1:
        rebuild = st.button(
            "🔄 Rebuild Knowledge Base",
            use_container_width=True,
            type="primary",
        )
    with col2:
        clear_idx = st.button(
            "🗑️ Clear Index",
            use_container_width=True,
        )
    with col3:
        refresh = st.button(
            "🔃 Refresh Status",
            use_container_width=True,
        )

    if clear_idx:
        from rag.vector_store import VectorStore
        vs = VectorStore(
            index_path=config.FAISS_INDEX_PATH,
            metadata_path=config.METADATA_PATH,
            info_path=config.INDEX_INFO_PATH,
        )
        vs.clear()
        load_system.clear()
        st.success("Index cleared.")
        st.rerun()

    if refresh:
        st.rerun()

    if rebuild:
        st.markdown("### ⚙️ Rebuilding knowledge base...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        def ui_progress(step: str, pct: float):
            progress_bar.progress(int(pct))
            status_text.markdown(
                f'<div class="processing-step">{step}</div>',
                unsafe_allow_html=True,
            )

        result = ingest_documents(progress_callback=ui_progress)
        load_system.clear()  # Force reload of retriever

        if result["success"]:
            st.success(
                f"✅ Knowledge base rebuilt! "
                f"{result['num_documents']} documents, {result['num_chunks']} chunks."
            )
        else:
            st.error(
                f"❌ Rebuild failed: {'; '.join(result['errors'])}"
            )
            if result.get("num_documents", 0) > 0:
                st.warning("Partial success — some documents may not have been indexed.")

    st.markdown("---")

    # Indexed documents
    if idx_status["exists"] and idx_status.get("documents"):
        st.markdown("### 📄 Indexed Documents")
        docs = idx_status["documents"]
        for i, doc in enumerate(docs, 1):
            st.markdown(
                f'<div class="source-card">'
                f'<span class="source-name">📄 {i}. {doc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Data directory listing
    st.markdown("### 📁 Data Directory Contents")
    data_dir = Path(config.DATA_DIR)
    if data_dir.exists():
        files = [
            f for f in sorted(data_dir.iterdir())
            if f.is_file() and f.suffix.lower() in {".pdf", ".docx", ".txt"}
        ]
        if files:
            for f in files:
                size_kb = f.stat().st_size / 1024
                st.markdown(
                    f'<div style="font-size:0.85rem;padding:4px 8px;border-bottom:1px solid #f1f5f9;">'
                    f'📄 {f.name} <span style="color:#94a3b8;font-size:0.78rem;">({size_kb:.1f} KB)</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info(f"No documents in `{data_dir}`. Upload files above or copy them manually.")
    else:
        st.warning(f"Data directory not found: `{data_dir}`")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        "💡 <strong>Tip:</strong> After uploading new documents, click "
        "<strong>Rebuild Knowledge Base</strong> to make them searchable. "
        "The assistant will automatically use the updated index."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
def render_about_page():
    """Render the About project page."""
    st.markdown("## ℹ️ About This Project")

    _, _, _, status = load_system()
    provider = status.get("provider", "local")
    ibm_active = provider == "ibm"

    st.markdown(
        """
        ### AI College Admission Agent

        The **AI College Admission Agent** is a RAG-based multi-agent system designed to
        simplify the college admission process for students.

        It provides intelligent, grounded guidance on:
        - 🎓 Course discovery and personalised recommendations
        - ✅ Eligibility assessment
        - 💰 Fees and scholarships
        - 📄 Document checklists
        - 📅 Important deadlines
        - 📋 Application procedures

        Every answer is grounded in the actual knowledge base documents — the system
        **never invents** fees, deadlines, eligibility criteria, or scholarship amounts.
        """
    )

    st.markdown("---")

    # Technology stack
    st.markdown("### 🛠️ Technology Stack")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Core Technologies")
        tech_items = [
            ("🤖", "Agentic AI", "Multi-agent architecture with specialised agents"),
            ("📖", "RAG Pipeline", "Retrieval-Augmented Generation for grounded answers"),
            ("🔍", "FAISS", "Fast vector similarity search (Facebook AI)"),
            ("🧬", "Sentence Transformers", "Local dense embeddings (all-MiniLM-L6-v2)"),
            ("🐍", "Python 3.10+", "Primary development language"),
            ("🌐", "Streamlit", "Interactive web interface"),
        ]
        for icon, name, desc in tech_items:
            st.markdown(
                f"**{icon} {name}** — {desc}"
            )

    with col2:
        st.markdown("#### AI / LLM")
        if ibm_active:
            st.markdown(
                '<span class="badge badge-blue">✅ IBM Granite Active</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"- **IBM Granite** ({config.GRANITE_MODEL}) via watsonx.ai\n"
                "- **IBM Bob** — development and orchestration environment"
            )
        else:
            st.markdown(
                '<span class="badge badge-gray">Local Mode (Ollama)</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"- **Ollama** (local LLM: `{config.OLLAMA_MODEL}`)\n"
                "- **IBM Bob** — development and orchestration environment\n"
                "- **IBM Granite** (optional — configure IBM credentials to enable)"
            )

    st.markdown("---")

    # Agent descriptions
    st.markdown("### 🤝 Specialized Agents")
    agents_info = [
        ("🤖", "Admission Knowledge Agent", "General Q&A — admission policies, FAQs, deadlines"),
        ("🎓", "Course Recommendation Agent", "Personalised course advice based on student profile"),
        ("✅", "Eligibility Agent", "Eligibility checking against course requirements"),
        ("📋", "Application Guidance Agent", "Step-by-step application procedures and document checklists"),
        ("💰", "Scholarship & Fee Agent", "Fee structures and financial aid information"),
    ]
    for icon, name, desc in agents_info:
        st.markdown(
            f'<div class="source-card" style="border-left-color:#7c5cd8;">'
            f'<span class="source-name">{icon} {name}</span><br>'
            f'<span class="source-page">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # RAG workflow
    st.markdown("### 🔄 RAG Workflow")
    st.markdown(
        """
        ```
        Student Query
             ↓
        Query Router (classify intent)
             ↓
        Specialized Agent selected
             ↓
        Embedding Model (all-MiniLM-L6-v2)
             ↓
        FAISS Vector Search
             ↓
        Top-K Relevant Chunks retrieved
             ↓
        LLM (IBM Granite / Ollama)
             ↓
        Grounded Response + Source Citations
        ```
        """
    )

    st.markdown("---")

    # Future scope
    st.markdown("### 🚀 Future Scope")
    future = [
        ("🌐", "Real-Time College Portal Integration", "Connect to official APIs for live admission data"),
        ("🎤", "Voice Assistant", "Voice input/output support"),
        ("🌍", "Multilingual Support", "Tamil, Hindi, and other regional languages"),
        ("📱", "Mobile Application", "React Native / Flutter mobile app"),
        ("📊", "Application Tracker", "Track multiple college applications"),
    ]
    for icon, title, desc in future:
        st.markdown(f"- **{icon} {title}** — {desc}")

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#94a3b8;font-size:0.8rem;">'
        "Built with IBM Bob · IBM Granite · Agentic AI · RAG · FAISS · Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main app entry point
# ---------------------------------------------------------------------------
def main():
    """Main application entry point."""
    init_session_state()
    apply_custom_css()
    render_sidebar()

    page = st.session_state.page

    if page == "home":
        render_home_page()
    elif page == "chat":
        render_chat_page()
    elif page == "knowledge":
        render_knowledge_base_page()
    elif page == "about":
        render_about_page()
    else:
        render_home_page()


if __name__ == "__main__":
    main()
