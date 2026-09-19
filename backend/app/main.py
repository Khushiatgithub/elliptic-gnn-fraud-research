"""
FastAPI Application Entry Point.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import settings
from backend.app.routers import (
    dataset_router,
    figures_router,
    graph_router,
    prediction_router,
    results_router,
)
from backend.app.services.dataset_service import get_dataset_service
from backend.app.services.graph_service import get_graph_service
from backend.app.services.inference_service import get_inference_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly initialize indexed services on startup for instant API responses
    print("[FastAPI] Pre-warming services and indexing data...")
    dataset_srv = get_dataset_service()
    graph_srv = get_graph_service()
    inference_srv = get_inference_service()
    print("[FastAPI] Research demonstration engine is READY.")
    yield
    print("[FastAPI] Shutting down...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Interactive Research Demonstration Platform for Bitcoin Illicit Transaction Detection",
    lifespan=lifespan
)

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(dataset_router, prefix=settings.API_PREFIX)
app.include_router(graph_router, prefix=settings.API_PREFIX)
app.include_router(results_router, prefix=settings.API_PREFIX)
app.include_router(prediction_router, prefix=settings.API_PREFIX)
app.include_router(figures_router, prefix=settings.API_PREFIX)


@app.get("/api/health")
def health_check():
    """Health check endpoint to verify backend operational readiness."""
    return {
        "status": "healthy",
        "project": "Graph Neural Networks vs Tabular Machine Learning for Bitcoin Illicit Detection",
        "dataset": "Elliptic Bitcoin Dataset (203,769 transactions, 234,355 edges)",
        "version": settings.VERSION,
        "mode": "Research Demonstration Platform"
    }
