# src/ui/pages/analytics.py
"""
Analytics page - Token tracking and statistics
"""

import streamlit as st
import pandas as pd
import json
import logging
from pathlib import Path
from config import Config
from src.utils.token_accounting import get_accounting

logger = logging.getLogger(__name__)


def render():
    """Render analytics page."""
    
    st.title("📊 Analytique")
    st.markdown("Suivi de la consommation de tokens et statistiques système")
    st.markdown("---")
    
    # Current session stats
    st.markdown("### 📈 Session Actuelle")
    
    try:
        accounting = get_accounting()
        report = accounting.get_session_report()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔤 Tokens Totaux", report['total_tokens'])
        
        with col2:
            vectorization_tokens = report['vectorization']['total_tokens']
            st.metric("📚 Tokens Vectorization", vectorization_tokens)
        
        with col3:
            search_tokens = report['searches']['total_tokens']
            st.metric("🔍 Tokens Recherche", search_tokens)
        
        with col4:
            avg_tokens_per_search = report['searches'].get('avg_tokens_per_search', 0)
            st.metric("⏱️ Moy. Tokens/Recherche", f"{avg_tokens_per_search:.0f}")
    
    except Exception as e:
        st.warning(f"Impossible de charger les statistiques: {e}")
        logger.error(f"Analytics error: {e}", exc_info=True)
    
    st.markdown("---")
    
    st.markdown("### ℹ️ Informations Système")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Configuration")
        st.write(f"""
        - **LLM Provider**: {Config.LLM_PROVIDER}
        - **Region**: {Config.REGION}
        - **Days Back**: {Config.DAYS_BACK}
        """)
    
    with col2:
        st.markdown("#### État des données")
        st.write(f"""
        - **Raw Data**: {Config.RAW_DATA_DIR}
        - **Processed**: {Config.PROCESSED_DATA_DIR}
        - **Indexes**: {Config.INDEXES_DIR}
        """)
