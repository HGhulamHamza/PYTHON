#!/usr/bin/env python3
"""
scrape_conferences.py

Purpose:
    - This script demonstrates how to scrape conference listings and details from a category page on WikiCFP.
    - Extracts structured information: title, location, start/end dates, submission deadlines, and emails.
    - Saves output to a CSV file.
    - Respects ethical scraping principles: delay, user-agent, no spam.
    - This is intended for research/educational purposes only.

Usage:
    python scrape_conferences.py --output conferences.csv --max 50

Important:
    - This script should be used for educational and research purposes.
    - Do not use collected emails for spam or commercial purposes.
    - Ensure compliance with the target website's robots.txt and Terms of Service.
"""

import argparse
import csv
import random
import re
import sys
import time
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry
from urllib import robotparser

# ----------------------------
# Configuration & Constants
# ----------------------------
DEFAULT_USER_AGENT = (
    "ResearcherScraper/1.0 (+https://your-institution.example/; "
    "contact: your.email@example.com) - Educational use only"
)

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zAZ0-9\-.]+")
REQUEST_TIMEOUT = 15  # seconds
MIN_DELAY = 1.0  # seconds between requests
MAX_DELAY = 3.0  # seconds between requests
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.0

CSV_COLUMNS = [
    "scrape_datetime",
    "source_domain",
    "category_url",
    "conference_title",
    "conference_url",
    "location",
    "start_date",
    "end_date",
    "submission_deadline",
    "emails_found",
    "raw_text_snippet",
    "notes",
]

CATEGORY_URL = "http://www.wikicfp.com/cfp/call?conference=artificial%20intelligence"


# ----------------------------
# Utilities
# ----------------------------
def setup_session(user_agent: str = DEFAULT_USER_AGENT) -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": user_agent})
    retries = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET", "HEAD"],
    )
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.mount("http://", HTTPAdapter(max_retries=retries))
    return sess


def is_allowed_by_robots(target_url: str, user_agent: str = DEFAULT_USER_AGENT) -> bool:
    """
    Check robots.txt for the target domain and see if crawling the given URL is allowed.
    """
    parsed = urlparse(target_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, target_url)
    except Exception:
        return False


def polite_sleep(min_delay: float = MIN_DELAY, max_delay: float = MAX_DELAY):
    delay = random.uniform(min_delay, max_delay)
    time.sleep(delay)


def find_emails_in_text(text: str) -> List[str]:
    return sorted(set(EMAIL_REGEX.findall(text)))


# ----------------------------
# Parsers (WikiCFP specific)
# ----------------------------
def extract_conference_links_from_category(html_soup: BeautifulSoup, base_url: str) -> List[Dict]:
    """
    Extract a list of conference links from a category page on WikiCFP.
    Returns a list of dicts: {"title": ..., "url": ...}
    """
    results = []

    # Extract all anchor tags that are conference links (heuristic based on 'cfp', 'call', etc. in URL)
    for a in html_soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        if len(text) < 3:
            continue
        if any(k in href.lower() for k in ("cfp", "event", "conference", "conf")):
            full = urljoin(base_url, href)
            results.append({"title": text, "url": full})

    # Deduplicate preserving order
    seen = set()
    unique = []
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)
    return unique


def parse_conference_detail(html_soup: BeautifulSoup, conference_url: str) -> Dict:
    """
    Extract structured info from a conference detail page.
    Heuristics: title, dates, location, deadlines, emails.
    """
    data = {
        "conference_title": None,
        "conference_url": conference_url,
        "location": None,
        "start_date": None,
        "end_date": None,
        "submission_deadline": None,
        "emails_found": [],
        "raw_text_snippet": None,
        "notes": "",
    }

    # Title: try h1, h2, or title tag
    title_tag = html_soup.find(["h1", "h2"])
    if title_tag:
        data["conference_title"] = title_tag.get_text(strip=True)
    else:
        data["conference_title"] = html_soup.title.string.strip() if html_soup.title else None

    # Large text block to search for dates and location
    body_text = html_soup.get_text(separator="\n", strip=True)
    data["raw_text_snippet"] = body_text[:1000]  # first 1000 chars as a snippet

    # Emails
    data["emails_found"] = find_emails_in_text(body_text)

    # Very simple date heuristics (YYYY or Month Day Year)
    date_matches = re.findall(
        r"(\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4})",
        body_text,
    )
    if date_matches:
        data["start_date"] = date_matches[0]
        if len(date_matches) > 1:
            data["end_date"] = date_matches[-1]

    # location heuristic: look for "Location:" or "Venue:"
    loc = None
    for label in ("Location:", "Venue:", "Place:"):
        idx = body_text.find(label)
        if idx != -1:
            snippet = body_text[idx : idx + 200]
            loc_match = re.split(r"[\n\r]", snippet)
            if len(loc_match) > 1:
                loc = loc_match[0].replace(label, "").strip()
            else:
                loc = snippet.replace(label, "").strip()
            break
    data["location"] = loc

    # submission deadline heuristic
    deadline_match = re.search(r"(Submission deadline|Paper submission deadline|Deadline)[:\s]*([A-Za-z0-9 ,.-]+)", body_text, re.IGNORECASE)
    if deadline_match:
        data["submission_deadline"] = deadline_match.group(2).strip()

    return data


# ----------------------------
# Main scraping workflow
# ----------------------------
def scrape_category(category_url: str, output_csv: str, max_items: Optional[int] = None, user_agent: str = DEFAULT_USER_AGENT):
    sess = setup_session(user_agent=user_agent)
    parsed = urlparse(category_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    print(f"[{datetime.utcnow().isoformat()}] Starting scrape for: {category_url}")
    # robots.txt check
    allowed = is_allowed_by_robots(category_url, user_agent=user_agent)
    if not allowed:
        print("ERROR: Crawling disallowed by robots.txt or robots.txt not accessible. Aborting.")
        return

    # Download category page
    try:
        r = sess.get(category_url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR fetching category page: {e}")
        return

    soup = BeautifulSoup(r.text, "html.parser")
    conf_links = extract_conference_links_from_category(soup, base_url=base_url)
    if max_items:
        conf_links = conf_links[:max_items]
    print(f"Found {len(conf_links)} candidate links (after heuristics).")

    results = []
    for idx, conf in enumerate(conf_links, 1):
        conf_url = conf["url"]
        conf_title_hint = conf.get("title") or ""
        print(f"[{idx}/{len(conf_links)}] Fetching: {conf_url}  (hint: {conf_title_hint})")

        # robots check per conference page
        if not is_allowed_by_robots(conf_url, user_agent=user_agent):
            print(f" - Skipping {conf_url}: disallowed by robots.txt")
            continue

        try:
            conf_resp = sess.get(conf_url, timeout=REQUEST_TIMEOUT)
            conf_resp.raise_for_status()
        except Exception as e:
            print(f" - Failed to fetch {conf_url}: {e}")
            continue

        conf_soup = BeautifulSoup(conf_resp.text, "html.parser")
        parsed_data = parse_conference_detail(conf_soup, conf_url)

        row = {
            "scrape_datetime": datetime.utcnow().isoformat(),
            "source_domain": parsed.netloc,
            "category_url": category_url,
            "conference_title": parsed_data.get("conference_title"),
            "conference_url": conf_url,
            "location": parsed_data.get("location"),
            "start_date": parsed_data.get("start_date"),
            "end_date": parsed_data.get("end_date"),
            "submission_deadline": parsed_data.get("submission_deadline"),
            "emails_found": ";".join(parsed_data.get("emails_found", [])),
            "raw_text_snippet": parsed_data.get("raw_text_snippet"),
            "notes": parsed_data.get("notes"),
        }
        results.append(row)

        # polite delay between requests
        polite_sleep()

    # Save to CSV (via pandas for convenience)
    if results:
        df = pd.DataFrame(results, columns=CSV_COLUMNS)
        df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Saved {len(df)} records to {output_csv}")
    else:
        print("No results to save.")


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Educational scraper for conference listings (ethical).")
    parser.add_argument("--output", default="conferences.csv", help="Output CSV filename. Default: conferences.csv")
    parser.add_argument("--max", type=int, default=30, help="Maximum number of conference links to attempt (for safety/testing).")
    parser.add_argument("--user_agent", default=DEFAULT_USER_AGENT, help="User-Agent to use.")
    args = parser.parse_args()

    # Quick sanity checks
    scrape_category(CATEGORY_URL, args.output, max_items=args.max, user_agent=args.user_agent)


if __name__ == "__main__":
    main()
