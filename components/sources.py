"""
Source display components — detailed source citation rendering.
"""

import streamlit as st


def render_source_card(source: dict, show_snippet: bool = True) -> None:
    """
    Render a single source document card.

    Args:
        source: Dict with keys: source, page, section, score, snippet
        show_snippet: Whether to include the text snippet expander.
    """
    source_name = source.get("source", "Unknown")
    page = source.get("page", "")
    section = source.get("section", "")
    score = source.get("score", 0)
    snippet = source.get("snippet", "")

    page_str = f" — Page {page}" if page else ""
    section_str = f" · {section}" if section else ""

    score_html = ""
    if score:
        score_pct = int(score * 100)
        color = "#22c55e" if score_pct >= 60 else "#f59e0b" if score_pct >= 40 else "#94a3b8"
        score_html = (
            f"<span style='font-size:0.74rem;color:{color};margin-left:6px;'>"
            f"▲ {score_pct}% relevance</span>"
        )

    # Build optional snippet block using HTML <details> so this component is
    # safe to call both at top-level AND inside an st.expander without crashing.
    snippet_html = ""
    if show_snippet and snippet:
        safe_snippet = snippet.replace("<", "&lt;").replace(">", "&gt;")
        snippet_html = (
            f"<details style='margin-top:4px;'>"
            f"<summary style='font-size:0.78rem;color:#3b82f6;cursor:pointer;"
            f"list-style:none;'>&#9656; View excerpt</summary>"
            f"<blockquote style='font-size:0.81rem;color:#475569;"
            f"border-left:3px solid #3b82f6;padding-left:8px;margin:4px 0;'>"
            f"<em>&quot;{safe_snippet}&quot;</em>"
            f"</blockquote></details>"
        )

    st.markdown(
        f"""
        <div class="source-card">
            <span class="source-name">📄 {source_name}</span>
            <span class="source-page">{page_str}{section_str}</span>
            {score_html}
            {snippet_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
