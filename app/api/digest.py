from fastapi import APIRouter
from app.jobs.scheduler import run_nightly_digest

router = APIRouter()

@router.post("/run_digest", tags=["Digest"])
async def manual_digest():
    await run_nightly_digest()
    return {"status": "digest_sent"}
