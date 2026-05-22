import xml.etree.ElementTree as ET
import hashlib
import re
import argparse
from email.utils import parsedate_to_datetime
from datetime import date
import httpx

# Default RSS feeds to scrape
DEFAULT_FEEDS = {
    "TechCrunch": "https://techcrunch.com/feed/",
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeed/rss",
    "CoinTelegraph": "https://cointelegraph.com/rss"
}

def clean_html(text: str) -> str:
    """Strip HTML tags from text."""
    if not text:
        return ""
    # Strip HTML tags
    clean = re.sub(r'<[^>]+>', '', text)
    # Normalize whitespaces
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()

def parse_rss_feed(source_name: str, feed_url: str) -> list:
    """Fetch and parse RSS feed items."""
    print(f"Fetching {source_name} feed...")
    try:
        resp = httpx.get(feed_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {source_name}: {e}")
        return []

    try:
        root = ET.fromstring(resp.content)
    except Exception as e:
        print(f"Failed to parse XML for {source_name}: {e}")
        return []

    items = []
    # Find all <item> tags in the RSS feed
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

        # Clean HTML/whitespaces
        text = clean_html(description)
        if not text:
            text = title

        # Parse date to YYYY-MM-DD
        published_at = date.today().isoformat()
        if pub_date_str:
            try:
                dt = parsedate_to_datetime(pub_date_str)
                published_at = dt.date().isoformat()
            except Exception:
                pass

        # Generate deterministic unique ID from link
        doc_id = f"rss-{hashlib.md5(link.encode('utf-8')).hexdigest()}"

        items.append({
            "id": doc_id,
            "title": title.strip(),
            "text": text,
            "source": source_name,
            "published_at": published_at,
            "reach": 15000  # standard fallback reach
        })

    print(f"Parsed {len(items)} items from {source_name}")
    return items

def run_sync(api_url: str):
    """Sync all parsed feed items to the microservice API."""
    all_mentions = []
    for source, url in DEFAULT_FEEDS.items():
        items = parse_rss_feed(source, url)
        all_mentions.extend(items)

    if not all_mentions:
        print("No mentions parsed. Exiting.")
        return

    print(f"Syncing total of {len(all_mentions)} mentions to {api_url}/ingest ...")

    # Ingest in batches of 50 to avoid payload size timeouts
    batch_size = 50
    inserted = 0
    updated = 0
    errors = 0

    for i in range(0, len(all_mentions), batch_size):
        batch = all_mentions[i:i + batch_size]
        payload = {"mentions": batch}
        
        try:
            r = httpx.post(
                f"{api_url.rstrip('/')}/ingest",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            r.raise_for_status()
            res = r.json()
            inserted += res.get("inserted", 0)
            updated += res.get("updated", 0)
            errs = res.get("errors", [])
            errors += len(errs)
            if errs:
                print(f"Batch errors: {errs}")
        except Exception as e:
            print(f"Failed to post batch {i//batch_size + 1}: {e}")

    print("\n--- Sync Summary ---")
    print(f"Successfully Inserted: {inserted}")
    print(f"Successfully Updated:  {updated}")
    print(f"Failed / Errors:       {errors}")
    print("--------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated RSS Ingestion Client")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Target FastAPI microservice URL (default: http://localhost:8000)"
    )
    args = parser.parse_args()
    run_sync(args.url)
