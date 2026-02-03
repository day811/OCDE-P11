# src/api/app.py
import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from src.rag.rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Events API", version="1.0.0")

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global RAG engine instance
rag_engine = None

def get_rag_engine(snapshot_date: Optional[str] = None) -> RAGEngine:
    """Get or initialize RAG engine"""
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine(snapshot_date=snapshot_date)
    return rag_engine

# Request/Response models
class SearchRequest(BaseModel):
    question: str
    top_k: int = 5
    snapshot_date: Optional[str] = None

class SearchResponse(BaseModel):
    answer: str
    sources: List[dict]
    constraints: dict
    total_tokens: int
    execution_time: float

class ExampleQuery(BaseModel):
    description: str
    question: str

class ExamplesResponse(BaseModel):
    examples: List[ExampleQuery]

# Routes
@app.get("/")
async def root():
    """Serve HTML interface"""
    html_file = Path(__file__).parent / "static" / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    return {"message": "RAG Events API - Open /api/v1/docs for API docs"}

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/api/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search for events with RAG"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        engine = get_rag_engine(request.snapshot_date)
        result = engine.answer_question(
            question=request.question,
            top_k=request.top_k,
            snapshot_date=request.snapshot_date
        )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat", response_model=SearchResponse)
async def chat_endpoint(request: SearchRequest):
    """Search for events with LangChain RAG"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        import time
        start_time = time.time()
        
        from src.rag.chatbot import ChatBot
        bot = ChatBot(mode = 'UI')
        result = bot.chat(request.question)
        
        execution_time = time.time() - start_time
        
        # ✅ TRANSFORMER LES DOCUMENT LANGCHAIN EN DICT
        sources = []
        if result.get('sources'):
            for doc in result['sources']:
                # doc est un objet Document avec .metadata et .page_content
                sources.append({
                    'title': doc.metadata.get('source', '').split('/')[-1],
                    'city': doc.metadata.get('city', 'NA'),
                    'dates': doc.metadata.get('dates', 'NA'),
                    'url': doc.metadata.get('source', ''),
                    'event_id': doc.metadata.get('event_id', 'NA'),
                    'dept': doc.metadata.get('dept', 'NA'),
                    'distance': 100
                })
        
        # ✅ RETOURNER AVEC TOUS LES CHAMPS REQUIS
        return {
            'answer': result.get('answer', ''),
            'sources': sources,
            'constraints': {
                'date': None,
                'city': None,
                'dept': None
            },
            'execution_time': execution_time,
            'total_tokens': result.get('total_tokens', 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Remplacer l'endpoint :
@app.get("/ui/search", response_class=FileResponse)
async def search_ui():
    """Serve interactive chat interface from file"""
    chat_html_path = Path(__file__).parent / "static" / "search.html"
    if chat_html_path.exists():
        return FileResponse(str(chat_html_path), media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="search.html not found")
    
# Remplacer l'endpoint :
@app.get("/ui/chat", response_class=FileResponse)
async def chat_ui():
    """Serve interactive chat interface from file"""
    chat_html_path = Path(__file__).parent / "static" / "chat.html"
    if chat_html_path.exists():
        return FileResponse(str(chat_html_path), media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="chat.html not found")