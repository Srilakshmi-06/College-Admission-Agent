"""
UI styling and shared layout components.
Applies custom CSS for a professional academic/AI look.
"""

import streamlit as st


def apply_custom_css() -> None:
    """Inject custom CSS into the Streamlit page."""
    st.markdown(
        """
        <style>
        /* ── Global ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        /* ── Main container ── */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            max-width: 1100px;
        }

        /* ── Hero section ── */
        .hero-container {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f4c81 100%);
            border-radius: 16px;
            padding: 3rem 2.5rem;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .hero-subtitle {
            font-size: 1.1rem;
            opacity: 0.85;
            margin-top: 0.6rem;
            font-weight: 400;
        }
        .hero-badge {
            display: inline-block;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.78rem;
            margin-bottom: 1rem;
            letter-spacing: 0.5px;
        }

        /* ── Feature cards ── */
        .feature-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.4rem 1.2rem;
            text-align: center;
            height: 100%;
            transition: box-shadow 0.2s;
        }
        .feature-card:hover {
            box-shadow: 0 4px 16px rgba(15,76,129,0.1);
            border-color: #3b82f6;
        }
        .feature-card .icon {
            font-size: 2rem;
            margin-bottom: 0.6rem;
        }
        .feature-card h4 {
            font-size: 0.95rem;
            font-weight: 600;
            color: #1e293b;
            margin: 0 0 0.4rem 0;
        }
        .feature-card p {
            font-size: 0.82rem;
            color: #64748b;
            margin: 0;
            line-height: 1.4;
        }

        /* ── Chat messages ── */
        .chat-message {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            line-height: 1.6;
            font-size: 0.93rem;
        }
        .chat-user {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            margin-left: 2rem;
        }
        .chat-assistant {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            margin-right: 2rem;
        }
        .chat-avatar {
            font-size: 1.2rem;
            margin-right: 0.5rem;
        }
        .chat-meta {
            font-size: 0.75rem;
            color: #94a3b8;
            margin-bottom: 0.4rem;
        }

        /* ── Source cards ── */
        .source-card {
            background: #f1f5f9;
            border: 1px solid #cbd5e1;
            border-left: 3px solid #3b82f6;
            border-radius: 8px;
            padding: 0.6rem 0.9rem;
            margin: 0.3rem 0;
            font-size: 0.82rem;
        }
        .source-card .source-name {
            font-weight: 600;
            color: #1e40af;
        }
        .source-card .source-page {
            color: #64748b;
            font-size: 0.78rem;
        }

        /* ── Status badges ── */
        .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        .badge-green { background: #dcfce7; color: #166534; }
        .badge-red   { background: #fee2e2; color: #991b1b; }
        .badge-yellow{ background: #fef9c3; color: #854d0e; }
        .badge-blue  { background: #dbeafe; color: #1d4ed8; }
        .badge-gray  { background: #f1f5f9; color: #475569; }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: #0f172a;
        }
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stRadio label {
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: white !important;
        }

        /* ── Info boxes ── */
        .info-box {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin: 0.8rem 0;
            font-size: 0.88rem;
            color: #1e3a5f;
        }
        .warning-box {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin: 0.8rem 0;
            font-size: 0.88rem;
            color: #78350f;
        }
        .error-box {
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin: 0.8rem 0;
            font-size: 0.88rem;
            color: #7f1d1d;
        }

        /* ── Stats grid ── */
        .stat-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }
        .stat-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1e3a5f;
        }
        .stat-label {
            font-size: 0.78rem;
            color: #64748b;
            margin-top: 0.2rem;
        }

        /* ── Processing indicator ── */
        .processing-step {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 8px;
            padding: 0.5rem 0.8rem;
            margin: 0.2rem 0;
            font-size: 0.83rem;
            color: #166534;
        }

        /* ── Dividers ── */
        hr { border-color: #e2e8f0; margin: 1.5rem 0; }

        /* ── Stray buttons ── */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
        }

        /* ── Suggested questions ── */
        .suggestion-chip {
            display: inline-block;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.8rem;
            color: #3b82f6;
            margin: 2px;
            cursor: pointer;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str, badge: str = "") -> None:
    """Render the hero header section."""
    badge_html = f'<div class="hero-badge">{badge}</div>' if badge else ""
    st.markdown(
        f"""
        <div class="hero-container">
            {badge_html}
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(label: str, color: str = "blue") -> str:
    """Return an HTML badge string."""
    return f'<span class="badge badge-{color}">{label}</span>'


def render_info_box(message: str, box_type: str = "info") -> None:
    """Render a styled info/warning/error box."""
    cls = {"info": "info-box", "warning": "warning-box", "error": "error-box"}.get(
        box_type, "info-box"
    )
    st.markdown(f'<div class="{cls}">{message}</div>', unsafe_allow_html=True)
