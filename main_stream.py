# main.py - Streamlit Entry Point

import streamlit as st
import logging
from datetime import datetime
from pathlib import Path

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration Streamlit
st.set_page_config(
    page_title="🎭 Puls-Events - Événements Culturels - RAG",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .source-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    .answer-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #00b4d8;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.search_history = []
    st.session_state.last_result = None
    st.session_state.total_tokens = 0

logger.info(f"Session ID: {st.session_state.session_id}")

# ============================================================================
# SIDEBAR & NAVIGATION
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.title("""Puls-Events
                 📋 Menu""")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Sélectionner une page",
    [
        "🏠 Accueil",
        "🔍 Recherche Simple",
        "💬 Chat Interactif",
        "📊 Analytique"
    ],
    key="page_selection"
)

st.sidebar.markdown("---")

# Configuration snapshot date en sidebar
st.sidebar.markdown("### ⚙️ Configuration")
from config import Config

snapshot_date = st.sidebar.text_input(
    "Date snapshot (YYYY-MM-DD)",
    value=Config.DEV_SNAPSHOT_DATE,
    key="snapshot_date_sidebar"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Version**: v1.0 (Streamlit)")
st.sidebar.markdown(f"**Session**: {st.session_state.session_id}")

# ============================================================================
# PAGE ROUTING
# ============================================================================

if page == "🏠 Accueil":
    from src.ui.pages import home
    home.render(snapshot_date=snapshot_date)

elif page == "🔍 Recherche Simple":
    from src.ui.pages import search
    search.render(snapshot_date=snapshot_date)

elif page == "💬 Chat Interactif":
    from src.ui.pages import chat
    chat.render(snapshot_date=snapshot_date)

elif page == "📊 Analytique":
    from src.ui.pages import analytics
    analytics.render()
