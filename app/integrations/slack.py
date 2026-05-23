import httpx
from app.config import settings

async def post_digest(mentions: list):
    # Fetch all registered webhooks from SQLite
    webhooks = []
    try:
        from app.models.db import db_conn
        with db_conn() as conn:
            # Fetch manual subscribers
            rows = conn.execute("SELECT webhook_url FROM slack_subscribers").fetchall()
            webhooks.extend([r["webhook_url"] for r in rows])
            
            # Fetch OAuth installations
            oauth_rows = conn.execute("SELECT webhook_url FROM slack_oauth_installations").fetchall()
            webhooks.extend([r["webhook_url"] for r in oauth_rows])
    except Exception:
        pass

    # Always include settings.SLACK_WEBHOOK_URL by default if configured
    if settings.SLACK_WEBHOOK_URL and "..." not in settings.SLACK_WEBHOOK_URL:
        if settings.SLACK_WEBHOOK_URL not in webhooks:
            webhooks.append(settings.SLACK_WEBHOOK_URL)

    if not webhooks:
        from app.utils.logger import get_logger
        get_logger().warning("No Slack webhook URLs configured, skipping digest post")
        return

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "🔗 Web3 PR Intelligence Digest"}
        },
        {"type": "divider"}
    ]

    for i, m in enumerate(mentions[:3], 1):
        web3 = m.get("web3_signals", {})
        if isinstance(web3, str):
            import json
            try:
                web3 = json.loads(web3)
            except Exception:
                web3 = {}
        tickers = ", ".join(web3.get("tickers", [])) if web3 else ""
        ens = ", ".join(
            [e["ens"] for e in web3.get("ens_names", []) if isinstance(e, dict) and e.get("valid")]
        ) if web3 else ""

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{i}. {m['title']}*\n"
                    f"{m.get('summary', 'No summary available.')}\n"
                    f"Source: {m.get('source', 'Unknown')} | "
                    f"Published: {m.get('published_at', 'N/A')} | "
                    f"Reach: {m.get('reach', 0):,} | "
                    f"Sentiment: {m.get('sentiment', 'N/A')}\n"
                    + (f"Tickers: `{tickers}` " if tickers else "")
                    + (f"ENS: `{ens}`" if ens else "")
                )
            }
        })
        blocks.append({"type": "divider"})

    payload = {"blocks": blocks}

    from app.utils.logger import get_logger
    logger = get_logger()
    async with httpx.AsyncClient(timeout=10) as client:
        for url in webhooks:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            except Exception as e:
                logger.error("Failed to post digest to Slack webhook", error=str(e), url=url)
