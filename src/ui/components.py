# src/ui/components.py
"""
Composants Streamlit réutilisables pour l'interface RAG
"""

import streamlit as st
from typing import Dict, List, Optional
from datetime import datetime


def render_question_input(
    placeholder: str = "Ex: Concerts ce weekend à Toulouse",
    key: str = "question_input"
) -> str:
    """Render input field for user question."""
    question = st.text_area(
        "❓ Votre question",
        placeholder=placeholder,
        height=100,
        key=key
    )
    return question


def render_advanced_filters() -> Dict:
    """Render advanced filter options."""
    with st.expander("⚙️ Paramètres avancés", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            top_k = st.slider(
                "Nombre de résultats (Top-K)",
                min_value=1,
                max_value=20,
                value=5,
                key="top_k_slider"
            )
        
        with col2:
            temperature = st.slider(
                "Température LLM",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="temperature_slider"
            )
        
    return {
        'top_k': top_k,
        'temperature': temperature,
    }


def render_answer(answer: str, is_chat: bool = False):
    """Render formatted answer."""
    if not answer:
        st.warning("Pas de réponse disponible.")
        return
    
    icon = "💬" if is_chat else "✨"
    st.markdown(f"### {icon} Réponse")
    
    st.markdown(f"""
    <div class="answer-box">
    {answer}
    </div>
    """, unsafe_allow_html=True)


def render_sources(sources: List[Dict]):
    """Render list of source events."""
    if not sources:
        st.info("Aucune source trouvée.")
        return
    
    st.markdown(f"### 📍 Sources ({len(sources)} trouvées)")
    
    cols = st.columns(min(len(sources), 3))
    
    for idx, source in enumerate(sources):
        with cols[idx % len(cols)]:
            render_source_card(source)


def render_source_card(source: Dict):
    """Render single source card."""
    title = source.get('title', 'Sans titre')
    city = source.get('city', 'N/A')
    dates = source.get('dates', 'Date inconnue')
    url = source.get('url', '#')
    distance = source.get('distance')
    
    relevance = ""
    if distance:
        relevance_pct = int((1 - distance) * 100)
        relevance = f"⭐ Pertinence: {relevance_pct}%"
    
    st.markdown(f"""
    <div class="source-card">
    <strong>{title}</strong><br>
    📍 {city}<br>
    📅 {dates}<br>
    {relevance}
    <br><br>
    <a href="{url}" target="_blank">🔗 Voir l'événement</a>
    </div>
    """, unsafe_allow_html=True)


def render_stats(result: Dict):
    """Render statistics about the query."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_tokens = result.get('total_tokens', 0)
        st.metric("📊 Tokens utilisés", total_tokens)
    
    with col2:
        exec_time = result.get('execution_time', 0)
        st.metric("⏱️ Temps d'exécution", f"{exec_time:.2f}s")
    
    with col3:
        sources_count = len(result.get('sources', []))
        st.metric("📍 Résultats", sources_count)
    
    with st.expander("📈 Détails", expanded=False):
        st.json({
            'total_tokens': total_tokens,
            'execution_time': f"{exec_time:.3f}s",
            'sources_count': sources_count,
            'mode': result.get('mode', 'unknown'),
            'timestamp': datetime.now().isoformat()
        })


def render_error(error_message: str):
    """Render error message."""
    st.markdown(f"""
    <div style="background-color: #ffe0e0; padding: 15px; border-radius: 8px; color: #d32f2f; border-left: 4px solid #d32f2f;">
    ❌ <strong>Erreur:</strong> {error_message}
    </div>
    """, unsafe_allow_html=True)


def render_no_index_error():
    """Render error when index is not available."""
    st.error("""
    ❌ **Index non disponible**
    
    L'index Faiss n'a pas été créé pour cette date.
    
    Pour corriger:
    1. Exécutez le script de pipeline: `python scripts/run_pipeline.py`
    2. Ou utilisez une date de snapshot différente
    """)
