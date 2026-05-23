from fastapi import APIRouter, HTTPException
from app.jobs.scheduler import run_nightly_digest
from app.models.db import db_conn
from app.models.schemas import SlackSubscribeRequest

router = APIRouter()

@router.post("/run_digest", tags=["Digest"])
async def manual_digest():
    await run_nightly_digest()
    return {"status": "digest_sent"}

@router.post("/slack/subscribe", tags=["Slack Integration"])
async def subscribe_slack(req: SlackSubscribeRequest):
    url = req.webhook_url.strip()
    if not url.startswith("https://hooks.slack.com/services/"):
        raise HTTPException(status_code=400, detail="Invalid Slack webhook URL format. Must start with https://hooks.slack.com/services/")
    
    with db_conn() as conn:
        try:
            conn.execute("INSERT INTO slack_subscribers (webhook_url) VALUES (?)", (url,))
            return {"status": "subscribed", "webhook_url": url}
        except Exception:
            # unique constraint failed -> already subscribed
            return {"status": "already_subscribed", "webhook_url": url}

@router.post("/slack/unsubscribe", tags=["Slack Integration"])
async def unsubscribe_slack(req: SlackSubscribeRequest):
    url = req.webhook_url.strip()
    with db_conn() as conn:
        res = conn.execute("DELETE FROM slack_subscribers WHERE webhook_url = ?", (url,))
        if res.rowcount > 0:
            return {"status": "unsubscribed", "webhook_url": url}
        else:
            return {"status": "not_found", "webhook_url": url}
