from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.models.db import db_conn
from app.integrations.slack import post_digest
import json

scheduler = AsyncIOScheduler()

async def run_nightly_digest():
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT id, title, source, reach, sentiment, summary, web3_signals, published_at
            FROM mentions
            WHERE web3_signals IS NOT NULL
              AND web3_signals != '{}'
              AND (
                web3_signals LIKE '%eth_addresses%'
                OR web3_signals LIKE '%tickers%'
                OR web3_signals LIKE '%ens_names%'
              )
            ORDER BY (reach * 1.0 / MAX(1, JULIANDAY('now') - JULIANDAY(published_at))) DESC
            LIMIT 10
        """).fetchall()
        mentions = [dict(r) for r in rows]

    for m in mentions:
        raw = m.get("web3_signals") or "{}"
        try:
            m["web3_signals"] = json.loads(raw)
        except Exception:
            m["web3_signals"] = {}

    if mentions:
        await post_digest(mentions)

def start_scheduler():
    scheduler.add_job(
        run_nightly_digest,
        "cron",
        hour=8,
        minute=0,
        id="nightly_digest",
        replace_existing=True
    )
    scheduler.start()
