import os
import json
import hashlib
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time

BASE_URL = "https://www.founderstribune.org/"
RAW_DIR = "data/raw/founderstribune"
SEEN_FILE = "data/ft_seen.json"

os.makedirs(RAW_DIR, exist_ok=True)

if os.path.exists(SEEN_FILE):
    seen = set(json.load(open(SEEN_FILE)))
else:
    seen = set()

def url_to_filename(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest() + ".json"

def save_seen():
    with open(SEEN_FILE, "w") as f:
        json.dump(list(seen), f, indent=2)

def fetch_article(url):
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception as e:
        print(f"⚠️ Failed {url}: {e}")
        return

    soup = BeautifulSoup(r.text, "html.parser")

    title = soup.find("h1")
    title = title.get_text(strip=True) if title else ""

    author = soup.find("a", rel="author")
    author = author.get_text(strip=True) if author else ""

    date = soup.find("time")
    date = date.get("datetime") if date else ""

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n\n".join(paragraphs)

    doc = {
        "url": url,
        "title": title,
        "author": author,
        "date": date,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "text": text
    }

    outdir = os.path.join(RAW_DIR, datetime.utcnow().strftime("%Y-%m-%d"))
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, url_to_filename(url))
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    seen.add(url)
    print(f"✅ Saved: {title} ({url})")

def crawl_archive_pages():
    page = 1
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/117.0.0.0 Safari/537.36"
    }

    while True:
        url = f"{BASE_URL}archive?page={page}"
        print(f"📄 Scraping archive page {page}...")
        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            print("❌ No more pages or blocked. Stopping.")
            break

        soup = BeautifulSoup(r.text, "html.parser")
        links = [a["href"] for a in soup.find_all("a", href=True) if "/p/" in a["href"]]

        if not links:
            print("✅ No more links found.")
            break

        for link in links:
            if not link.startswith("http"):
                link = BASE_URL.rstrip("/") + link
            if link in seen:
                continue
            fetch_article(link)
            time.sleep(0.3)

        page += 1  # Next page

    save_seen()
    print(f"✅ Finished crawling. Total seen: {len(seen)}")

if __name__ == "__main__":
    crawl_archive_pages()
