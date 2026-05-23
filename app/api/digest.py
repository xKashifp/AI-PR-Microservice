from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
import httpx
from app.config import settings
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

@router.get("/slack/install", tags=["Slack OAuth"])
async def slack_install():
    if not settings.SLACK_CLIENT_ID or not settings.SLACK_REDIRECT_URI:
        raise HTTPException(status_code=400, detail="Slack Client credentials not configured in backend settings.")
    
    install_url = (
        f"https://slack.com/oauth/v2/authorize"
        f"?client_id={settings.SLACK_CLIENT_ID}"
        f"&scope=incoming-webhook"
        f"&redirect_uri={settings.SLACK_REDIRECT_URI}"
    )
    return RedirectResponse(url=install_url)

@router.get("/slack/oauth/callback", tags=["Slack OAuth"])
async def slack_oauth_callback(code: str = None, error: str = None):
    if error:
        return RedirectResponse(url=f"/?slack_error={error}")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")
        
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post(
                "https://slack.com/api/oauth.v2.access",
                data={
                    "client_id": settings.SLACK_CLIENT_ID,
                    "client_secret": settings.SLACK_CLIENT_SECRET,
                    "code": code,
                    "redirect_uri": settings.SLACK_REDIRECT_URI
                }
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return RedirectResponse(url=f"/?slack_error=OAuth exchange failed: {str(e)}")
            
    if not data.get("ok"):
        err_msg = data.get("error", "Unknown error")
        return RedirectResponse(url=f"/?slack_error={err_msg}")
        
    team = data.get("team", {})
    team_name = team.get("name", "Unknown Team")
    team_id = team.get("id", "")
    
    webhook_info = data.get("incoming_webhook", {})
    webhook_url = webhook_info.get("url", "")
    channel_name = webhook_info.get("channel", "Unknown Channel")
    channel_id = webhook_info.get("channel_id", "")
    access_token = data.get("access_token", "")
    
    if not webhook_url:
        return RedirectResponse(url="/?slack_error=No webhook URL returned in OAuth response")
        
    with db_conn() as conn:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO slack_oauth_installations 
                (team_id, team_name, channel_id, channel_name, webhook_url, access_token)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (team_id, team_name, channel_id, channel_name, webhook_url, access_token))
        except Exception as e:
            return RedirectResponse(url=f"/?slack_error=Database error: {str(e)}")
            
    return RedirectResponse(url="/?slack_success=connected")

@router.get("/slack/oauth/installations", tags=["Slack OAuth"])
async def list_oauth_installations():
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT id, team_name, channel_name, created_at 
            FROM slack_oauth_installations
        """).fetchall()
        return [
            {
                "id": r["id"],
                "team_name": r["team_name"],
                "channel_name": r["channel_name"],
                "created_at": r["created_at"]
            }
            for r in rows
        ]

@router.post("/slack/oauth/disconnect/{install_id}", tags=["Slack OAuth"])
async def disconnect_oauth_installation(install_id: int):
    with db_conn() as conn:
        row = conn.execute("""
            SELECT access_token FROM slack_oauth_installations WHERE id = ?
        """, (install_id,)).fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Installation not found")
            
        token = row["access_token"]
        
        if token:
            async with httpx.AsyncClient(timeout=5) as client:
                try:
                    await client.post(
                        "https://slack.com/api/auth.revoke",
                        headers={"Authorization": f"Bearer {token}"}
                    )
                except Exception:
                    pass
                    
        conn.execute("DELETE FROM slack_oauth_installations WHERE id = ?", (install_id,))
        
    return {"status": "disconnected"}
