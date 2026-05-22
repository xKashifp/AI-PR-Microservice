import os

# PyTorch/OMP thread and memory optimization
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.models.db import init_db
from app.search.faiss_store import get_store
from app.jobs.scheduler import start_scheduler
from app.api import ingest, search, health, digest
from app.utils.logger import get_logger
from app.utils.metrics import MetricsMiddleware


logger = get_logger()
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database")
    init_db()
    logger.info("Loading FAISS index")
    get_store()
    
    logger.info("Pre-loading NLP models to prevent runtime OOM")
    try:
        from app.nlp.embedder import Embedder
        from app.nlp.sentiment import get_sentiment_pipeline
        # Force loading of models into memory
        Embedder.get()
        get_sentiment_pipeline()
        logger.info("NLP models preloaded successfully")
    except Exception as e:
        logger.error(f"Failed to preload models: {e}")
        
    logger.info("Starting scheduler")
    start_scheduler()
    yield
    # Shutdown
    logger.info("Saving FAISS index on shutdown")
    get_store().save()

app = FastAPI(
    title="PR Mention Intelligence API",
    description="AI-powered PR/news mention intelligence microservice with semantic search, Web3 detection, and Slack alerts.",
    version="1.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(MetricsMiddleware)

app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(health.router)
app.include_router(digest.router)

# Dashboard
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
if os.path.exists(templates_dir):
    from fastapi.responses import FileResponse

    @app.get("/", include_in_schema=False)
    async def dashboard():
        return FileResponse(os.path.join(templates_dir, "index.html"))
