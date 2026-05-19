"""
Scrape Wikipedia infobox photos for takers and keepers.

Strategy (strict, prefers no photo over wrong photo):
  1. Skip if assets/photos/{player_id}.jpg already exists.
  2. Search Wikipedia for the player's FULL StatsBomb name.
  3. Take only the FIRST search result — never fall through to other matches.
  4. Fetch that page's categories.
  5. If no category mentions football → log + skip.
  6. Fetch the main image (pithumbsize=400).
  7. If no image → log + skip.
  8. Download, center-crop to a square, save as JPEG.
  9. Log every attempt to data/processed/wikipedia_scrape_log.csv.

Polite to Wikipedia:
  - Custom User-Agent identifying the project.
  - 1 second between requests.
  - Graceful 429 backoff.

Usage:
    python -m src.data.scrape_wikipedia_photos

Resumable — re-running skips players that already have a photo file.
"""

import csv
import io
import os
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

TAKERS_FILE = "data/processed/taker_profiles.csv"
KEEPERS_FILE = "data/processed/keeper_profiles.csv"
PHOTOS_DIR = "assets/photos"
LOG_FILE = "data/processed/wikipedia_scrape_log.csv"

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "PenaltySimulator/1.0 "
    "(https://github.com/Nabarun-Kalita/penalty-simulator; portfolio project) "
    "python-requests"
)

REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT = 15
IMAGE_SIZE = 400  # final cropped square size

# Category keywords that indicate "this person is a footballer"
FOOTBALL_KEYWORDS = [
    'footballer',
    'football player',
    'soccer player',
    'goalkeeper',
    'association football',
]


# ============================================================
# DATA TYPES
# ============================================================

@dataclass
class ScrapeResult:
    player_id: int
    player_name: str
    role: str  # 'taker' or 'keeper'
    status: str  # 'success', 'already_exists', 'no_search_results',
                 #  'not_a_footballer', 'no_image', 'download_failed', 'crop_failed'
    wikipedia_title: str = ""
    image_url: str = ""
    notes: str = ""


# ============================================================
# HTTP HELPERS
# ============================================================

_session = requests.Session()
_session.headers.update({'User-Agent': USER_AGENT})


def _api_get(params: dict) -> Optional[dict]:
    """Hit the Wikipedia API with polite rate limiting and 429 handling."""
    params = dict(params)
    params['format'] = 'json'

    for attempt in range(3):
        try:
            response = _session.get(WIKI_API, params=params, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as e:
            print(f"  ! Network error: {e}")
            return None

        if response.status_code == 429:
            wait = 5 * (attempt + 1)
            print(f"  ! Rate-limited (429), waiting {wait}s")
            time.sleep(wait)
            continue

        if not response.ok:
            print(f"  ! HTTP {response.status_code}: {response.text[:200]}")
            return None

        try:
            return response.json()
        except ValueError:
            return None

    return None


# ============================================================
# WIKIPEDIA STEPS
# ============================================================

def search_wikipedia(query: str) -> Optional[str]:
    """Return the title of the first search result, or None."""
    data = _api_get({
        'action': 'query',
        'list': 'search',
        'srsearch': query,
        'srlimit': 1,
    })
    if not data:
        return None

    results = data.get('query', {}).get('search', [])
    if not results:
        return None
    return results[0]['title']


def is_footballer_page(page_title: str) -> tuple[bool, str]:
    """
    Check whether a Wikipedia page is about a footballer.
    Returns (is_footballer, joined_category_string_for_logging).
    """
    data = _api_get({
        'action': 'query',
        'prop': 'categories',
        'titles': page_title,
        'cllimit': 50,
    })
    if not data:
        return False, ""

    pages = data.get('query', {}).get('pages', {})
    if not pages:
        return False, ""

    # Pages dict is keyed by pageid; grab the only value
    page = next(iter(pages.values()))
    categories = page.get('categories', [])
    if not categories:
        return False, ""

    # Category titles look like "Category:Argentine footballers"
    category_names = [c.get('title', '') for c in categories]
    joined = " | ".join(category_names)
    joined_lower = joined.lower()

    for keyword in FOOTBALL_KEYWORDS:
        if keyword in joined_lower:
            return True, joined

    return False, joined


def fetch_image_url(page_title: str) -> Optional[str]:
    """Return the main thumbnail URL for the page, or None."""
    data = _api_get({
        'action': 'query',
        'prop': 'pageimages',
        'titles': page_title,
        'pithumbsize': IMAGE_SIZE,
    })
    if not data:
        return None

    pages = data.get('query', {}).get('pages', {})
    if not pages:
        return None

    page = next(iter(pages.values()))
    thumbnail = page.get('thumbnail', {})
    return thumbnail.get('source')


def download_and_save_image(url: str, output_path: str) -> bool:
    """Download image from URL, center-crop to a square, save as JPEG."""
    try:
        response = _session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  ! Image download failed: {e}")
        return False

    try:
        img = Image.open(io.BytesIO(response.content))
        # Convert to RGB if it's RGBA, palette, or grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Center-crop to a square
        width, height = img.size
        side = min(width, height)
        left = (width - side) // 2
        top = 0
        right = left + side
        bottom = side
        img_cropped = img.crop((left, top, right, bottom))

        # Resize to IMAGE_SIZE
        img_final = img_cropped.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

        img_final.save(output_path, 'JPEG', quality=85, optimize=True)
        return True
    except Exception as e:
        print(f"  ! Image processing failed: {e}")
        return False


# ============================================================
# PER-PLAYER PIPELINE
# ============================================================

def process_player(player_id: int, name: str, role: str) -> ScrapeResult:
    """Full pipeline for one player. Returns a ScrapeResult."""
    photo_path = os.path.join(PHOTOS_DIR, f"{player_id}.jpg")

    if os.path.exists(photo_path):
        return ScrapeResult(player_id, name, role, 'already_exists')

    # ----- Step 1: search -----
    title = search_wikipedia(name)
    time.sleep(REQUEST_DELAY_SECONDS)
    if not title:
        return ScrapeResult(player_id, name, role, 'no_search_results',
                            notes="Wikipedia returned 0 results")

    # ----- Step 2: verify footballer -----
    is_footballer, categories_str = is_footballer_page(title)
    time.sleep(REQUEST_DELAY_SECONDS)
    if not is_footballer:
        return ScrapeResult(player_id, name, role, 'not_a_footballer',
                            wikipedia_title=title,
                            notes=f"Categories: {categories_str[:200]}")

    # ----- Step 3: fetch image URL -----
    image_url = fetch_image_url(title)
    time.sleep(REQUEST_DELAY_SECONDS)
    if not image_url:
        return ScrapeResult(player_id, name, role, 'no_image',
                            wikipedia_title=title,
                            notes="Page exists but has no infobox image")

    # ----- Step 4: download and save -----
    success = download_and_save_image(image_url, photo_path)
    time.sleep(REQUEST_DELAY_SECONDS)
    if not success:
        return ScrapeResult(player_id, name, role, 'download_failed',
                            wikipedia_title=title, image_url=image_url,
                            notes="Image URL found but download or processing failed")

    return ScrapeResult(player_id, name, role, 'success',
                        wikipedia_title=title, image_url=image_url)


# ============================================================
# LOG WRITING
# ============================================================

def append_to_log(result: ScrapeResult):
    """Append a row to the log CSV. Creates with header if not present."""
    file_exists = os.path.exists(LOG_FILE)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    with open(LOG_FILE, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'player_id', 'player_name', 'role',
                'status', 'wikipedia_title', 'image_url', 'notes',
            ])
        writer.writerow([
            result.player_id, result.player_name, result.role,
            result.status, result.wikipedia_title, result.image_url, result.notes,
        ])


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    print("Loading player profiles...")
    takers = pd.read_csv(TAKERS_FILE)
    keepers = pd.read_csv(KEEPERS_FILE)
    print(f"  {len(takers)} takers, {len(keepers)} keepers")

    # Build the work list
    work = []
    for _, row in takers.iterrows():
        work.append((int(row['taker_id']), row['taker_name'], 'taker'))
    for _, row in keepers.iterrows():
        work.append((int(row['keeper_id']), row['keeper_name'], 'keeper'))

    # Dedupe player_ids that appear in both lists (player who is both taker and keeper — rare)
    seen = set()
    deduped = []
    for player_id, name, role in work:
        if player_id in seen:
            continue
        seen.add(player_id)
        deduped.append((player_id, name, role))
    work = deduped

    print(f"  {len(work)} unique players to process")
    print()

    # Tally counters
    counts = {
        'success': 0,
        'already_exists': 0,
        'no_search_results': 0,
        'not_a_footballer': 0,
        'no_image': 0,
        'download_failed': 0,
    }

    start = time.time()
    for idx, (player_id, name, role) in enumerate(work, 1):
        prefix = f"[{idx:>4}/{len(work)}]"
        # Pre-check to avoid printing for already-existing photos in noisy fashion
        photo_path = os.path.join(PHOTOS_DIR, f"{player_id}.jpg")
        if os.path.exists(photo_path):
            counts['already_exists'] += 1
            # Quietly continue; don't even log re-attempts
            continue

        print(f"{prefix} {name} ({role})")

        result = process_player(player_id, name, role)
        counts[result.status] = counts.get(result.status, 0) + 1

        status_marker = "✓" if result.status == 'success' else "✗"
        print(f"      {status_marker} {result.status}", end="")
        if result.wikipedia_title:
            print(f" — {result.wikipedia_title}", end="")
        print()

        append_to_log(result)

    elapsed = time.time() - start

    # ----- Summary -----
    print()
    print("=" * 60)
    print("SCRAPE SUMMARY")
    print("=" * 60)
    print(f"  Total players:        {len(work)}")
    for status, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {status:22} {count}")
    print(f"  Elapsed: {elapsed/60:.1f} minutes")
    print()
    print(f"  Photos saved in:      {PHOTOS_DIR}/")
    print(f"  Full log:             {LOG_FILE}")


if __name__ == "__main__":
    main()
