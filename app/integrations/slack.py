import httpx
from app.config import settings

async def post_digest(mentions: list):
    if not settings.SLACK_WEBHOOK_URL or "..." in settings.SLACK_WEBHOOK_URL:
        from app.utils.logger import get_logger
        get_logger().warning("Slack webhook URL is empty or placeholder, skipping post")
        return  # Slack not configured, skip silently

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
        try:
            resp = await client.post(settings.SLACK_WEBHOOK_URL, json=payload)
            resp.raise_for_status()
        except Exception as e:
            logger.error("Failed to post digest to Slack", error=str(e), url=settings.SLACK_WEBHOOK_URL)
