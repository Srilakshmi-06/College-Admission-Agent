"""
Chat UI components — renders conversation messages, sources, and agent status.
"""

import streamlit as st
from typing import Optional


def render_chat_message(role: str, content: str, agent_name: Optional[str] = None) -> None:
    """
    Render a single chat message bubble.

    Args:
        role: "user" or "assistant"
        content: Message text (supports markdown)
        agent_name: Name of the responding agent (assistant only)
    """
    if role == "user":
        avatar = "🧑‍🎓"
        label = "You"
        css_class = "chat-user"
    else:
        avatar = "🤖"
        label = agent_name or "AI Admission Counselor"
        css_class = "chat-assistant"

    with st.container():
        st.markdown(
            f"""
            <div class="chat-message {css_class}">
                <div class="chat-meta">
                    <span class="chat-avatar">{avatar}</span>
                    <strong>{label}</strong>
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(content)
        st.markdown("</div>", unsafe_allow_html=True)


def render_sources(sources: list[dict], expanded: bool = False) -> None:
    """
    Render the source citations section for a response.

    Args:
        sources: List of source dicts from RAGRetriever.format_sources()
        expanded: Whether the expander is open by default
    """
    if not sources:
        return

    with st.expander(
        f"📚 Sources ({len(sources)} document{'s' if len(sources) > 1 else ''})",
        expanded=expanded,
    ):
        for src in sources:
            source_name = src.get("source", "Unknown")
            page = src.get("page", "")
            section = src.get("section", "")
            score = src.get("score", 0)
            snippet = src.get("snippet", "")

            page_str = f" — Page {page}" if page else ""
            section_str = f" · {section}" if section else ""
            score_pct = f"{score * 100:.0f}%" if score else ""
            score_html = (
                f"<br><small style='color:#94a3b8'>Relevance: {score_pct}</small>"
                if score_pct else ""
            )

            # Use a plain HTML <details> for the snippet preview so we never
            # nest a Streamlit expander inside another expander (that crashes).
            snippet_html = ""
            if snippet:
                safe_snippet = snippet.replace("<", "&lt;").replace(">", "&gt;")
                snippet_html = (
                    f"<details style='margin-top:4px;'>"
                    f"<summary style='font-size:0.78rem;color:#3b82f6;cursor:pointer;"
                    f"list-style:none;'>"
                    f"&#9656; Preview excerpt</summary>"
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


def render_agent_status(steps: list[str]) -> None:
    """
    Render processing status steps (shown while the agent is working).

    Args:
        steps: List of status strings to display.
    """
    for step in steps:
        st.markdown(
            f'<div class="processing-step">{step}</div>',
            unsafe_allow_html=True,
        )


def render_suggested_questions(questions: list[str]) -> Optional[str]:
    """
    Render a row of clickable suggested question buttons.

    Args:
        questions: List of question strings.

    Returns:
        The selected question string, or None.
    """
    selected: Optional[str] = None
    st.markdown(
        "<p style='font-size:0.82rem;color:#64748b;margin-bottom:0.4rem;'>"
        "💡 <strong>Suggested questions:</strong></p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(min(len(questions), 3))
    for i, question in enumerate(questions[:6]):
        col = cols[i % 3]
        with col:
            if st.button(
                question,
                key=f"sugg_{i}_{hash(question)}",
                use_container_width=True,
            ):
                selected = question
    return selected


def render_error_message(error: str) -> None:
    """Render a user-friendly error message."""
    st.markdown(
        f'<div class="error-box">⚠️ {error}</div>',
        unsafe_allow_html=True,
    )
