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
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

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

@app.get("/api/v1/examples", response_model=ExamplesResponse)
async def get_examples():
    """Get example queries"""
    examples = [
        {
            "description": "Événements ce soir",
            "question": "Quels événements ce soir à Toulouse ?"
        },
        {
            "description": "Concerts demain",
            "question": "Quels concerts demain à Montpellier ?"
        },
        {
            "description": "Spectacles ce weekend",
            "question": "Spectacles ce weekend en Occitanie"
        },
        {
            "description": "Expositions dans 3 jours",
            "question": "Expositions dans 3 jours à Nîmes"
        },
        {
            "description": "Tous les événements",
            "question": "Quels sont les événements disponibles ?"
        }
    ]
    return ExamplesResponse(examples=examples)

@app.get("/api/v1/cities")
async def get_cities():
    """Get available cities"""
    cities = [
        'toulouse', 'montpellier', 'nîmes', 'perpignan', 'albi',
        'rodez', 'cahors', 'figeac', 'auch', 'tarbes',
        'pau', 'biarritz', 'bayonne'
    ]
    return {"cities": cities}
