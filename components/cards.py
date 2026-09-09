"""
Card components for the home page feature grid and knowledge-base stats.
"""

import streamlit as st


FEATURE_CARDS = [
    {
        "icon": "🎓",
        "title": "Course Recommendations",
        "desc": "Discover courses matched to your background, interests, and goals.",
    },
    {
        "icon": "✅",
        "title": "Eligibility Checking",
        "desc": "Understand admission requirements and whether you qualify.",
    },
    {
        "icon": "💰",
        "title": "Fees & Scholarships",
        "desc": "Explore fee structures and financial aid opportunities.",
    },
    {
        "icon": "📄",
        "title": "Document Guidance",
        "desc": "Get a complete checklist of documents required for admission.",
    },
    {
        "icon": "📅",
        "title": "Admission Deadlines",
        "desc": "Never miss a key date with deadline reminders and schedules.",
    },
    {
        "icon": "🤖",
        "title": "AI Admission Counselor",
        "desc": "24/7 intelligent guidance powered by RAG and Agentic AI.",
    },
]


def render_feature_cards() -> None:
    """Render the 6 feature cards in a 3-column grid."""
    st.markdown("### What I Can Help You With")
    st.markdown("")

    rows = [FEATURE_CARDS[:3], FEATURE_CARDS[3:]]
    for row in rows:
        cols = st.columns(3)
        for col, card in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class="feature-card">
                        <div class="icon">{card['icon']}</div>
                        <h4>{card['title']}</h4>
                        <p>{card['desc']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("")


def render_knowledge_base_stats(info: dict) -> None:
    """
    Render a stats grid for the knowledge base dashboard.

    Args:
        info: Dict from VectorStore.info property.
    """
    num_docs = info.get("num_documents", 0)
    num_chunks = info.get("num_chunks", 0)
    built_at = info.get("built_at", "—")
    dim = info.get("dimension", "—")

    cols = st.columns(4)
    stats = [
        ("📄", str(num_docs), "Documents"),
        ("🧩", str(num_chunks), "Chunks"),
        ("🧬", str(dim), "Embedding Dim"),
        ("🕐", built_at if built_at else "—", "Last Updated"),
    ]
    for col, (icon, value, label) in zip(cols, stats):
        with col:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div style="font-size:1.4rem">{icon}</div>
                    <div class="stat-value">{value}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
