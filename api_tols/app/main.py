"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db
from .models_store import store
from .routers import classes, datasets, health, jobs, models, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    store.load_production()
    yield
    # Shutdown (nothing to clean up)


app = FastAPI(
    title="Tools Segmentation API",
    version="1.0.0",
    description="YOLOv8-based tool detection and segmentation API",
    lifespan=lifespan,
)

# CORS — allow the frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
PREFIX = "/api/v1"
app.include_router(health.router, prefix=PREFIX, tags=["health"])
app.include_router(classes.router, prefix=PREFIX, tags=["classes"])
app.include_router(predict.router, prefix=PREFIX, tags=["predict"])
app.include_router(datasets.router, prefix=PREFIX, tags=["datasets"])
app.include_router(models.router, prefix=PREFIX, tags=["models"])
app.include_router(jobs.router, prefix=PREFIX, tags=["jobs"])
