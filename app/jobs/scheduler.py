from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.models.db import db_conn
from app.integrations.slack import post_digest
import json
import httpx
import xml.etree.ElementTree as ET
import hashlib
import re
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from app.models.schemas import MentionIn
from app.utils.logger import get_logger

logger = get_logger()
scheduler = AsyncIOScheduler()

DEFAULT_FEEDS = {
    "TechCrunch": "https://techcrunch.com/feed/",
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeed/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss"
}

def clean_html(text: str) -> str:
    if not text:
        return ""
    clean = re.sub(r'<[^>]+>', '', text)
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()

async def fetch_and_parse_rss() -> list[MentionIn]:
    all_mentions = []
    headers = {"User-Agent": "Mozilla/5.0"}
    
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        for source_name, feed_url in DEFAULT_FEEDS.items():
            logger.info(f"Background Job: Fetching {source_name} feed...")
            try:
                resp = await client.get(feed_url, headers=headers)
                resp.raise_for_status()
            except Exception as e:
                logger.error(f"Failed to fetch {source_name}: {e}")
                continue

            try:
                root = ET.fromstring(resp.content)
            except Exception as e:
                logger.error(f"Failed to parse XML for {source_name}: {e}")
                continue

            count = 0
            for item in root.findall(".//item"):
                title_el = item.find("title")
                link_el = item.find("link")
                desc_el = item.find("description")
                pub_date_el = item.find("pubDate")

                title = title_el.text if title_el is not None else ""
                link = link_el.text if link_el is not None else ""
                description = desc_el.text if desc_el is not None else ""
                pub_date_str = pub_date_el.text if pub_date_el is not None else ""

                if not title or not link:
                    continue

                text = clean_html(description)
                if not text:
                    text = title

                published_at = date.today().isoformat()
                if pub_date_str:
                    try:
                        dt = parsedate_to_datetime(pub_date_str)
                        published_at = dt.date().isoformat()
                    except Exception:
                        pass

                doc_id = f"rss-{hashlib.md5(link.encode('utf-8')).hexdigest()}"

                all_mentions.append(MentionIn(
                    id=doc_id,
                    title=title.strip(),
                    text=text,
                    source=source_name,
                    published_at=published_at,
                    reach=15000,
                    labels=[]
                ))
                count += 1
            logger.info(f"Background Job: Parsed {count} items from {source_name}")
            
    return all_mentions

async def run_periodic_ingestion():
    logger.info("Starting background RSS ingestion job")
    try:
        mentions = await fetch_and_parse_rss()
        if not mentions:
            logger.info("No new mentions found to ingest")
            return
        
        # Import process_and_ingest_mentions dynamically to avoid circular import issues
        from app.api.ingest import process_and_ingest_mentions
        res = await process_and_ingest_mentions(mentions)
        logger.info(
            "Background RSS Ingestion completed",
            inserted=res.get("inserted", 0),
            updated=res.get("updated", 0),
            errors=len(res.get("errors", []))
        )
    except Exception as e:
        logger.error(f"Error in background RSS ingestion job: {e}")

async def run_nightly_digest():
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT id, title, source, reach, sentiment, summary, web3_signals, published_at
            FROM mentions
            WHERE sent_to_slack = 0
              AND web3_signals IS NOT NULL
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
        
        # Mark these mentions as sent to prevent resending them in the next digest
        mention_ids = [m["id"] for m in mentions]
        with db_conn() as conn:
            conn.executemany(
                "UPDATE mentions SET sent_to_slack = 1 WHERE id = ?",
                [(m_id,) for m_id in mention_ids]
            )

def start_scheduler():
    # Nightly digest
    scheduler.add_job(
        run_nightly_digest,
        "cron",
        hour=8,
        minute=0,
        id="nightly_digest",
        replace_existing=True
    )
    
    # RSS Ingestion job (every hour)
    scheduler.add_job(
        run_periodic_ingestion,
        "interval",
        minutes=60,
        id="periodic_rss_ingestion",
        replace_existing=True
    )
    
    # Also trigger an immediate run of ingestion 10 seconds after startup
    scheduler.add_job(
        run_periodic_ingestion,
        "date",
        run_date=datetime.now() + timedelta(seconds=10),
        id="startup_rss_ingestion"
    )
    
    scheduler.start()
