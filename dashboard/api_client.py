"""
Wrapper around the Penalty Simulator API.

Centralizes HTTP calls so the dashboard doesn't deal with requests directly.
Reads API_URL from environment (defaults to localhost for dev).
"""

import os
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()  # Load .env file if present

API_URL = os.environ.get('API_URL', 'http://localhost:8000')
REQUEST_TIMEOUT = 30


class APIError(Exception):
    """Raised when the API returns an error or is unreachable."""
    pass


def _request(method: str, endpoint: str, **kwargs):
    """Single place for HTTP handling + error translation."""
    url = f"{API_URL}{endpoint}"
    try:
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.exceptions.ConnectionError:
        raise APIError(f"Cannot reach API at {API_URL}. Is the server running?")
    except requests.exceptions.Timeout:
        raise APIError(f"Request to {url} timed out after {REQUEST_TIMEOUT}s")

    if not response.ok:
        try:
            detail = response.json().get('detail', response.text)
        except Exception:
            detail = response.text
        raise APIError(f"API error {response.status_code}: {detail}")

    return response.json()


# ============================================================
# Endpoints
# ============================================================

def health_check() -> dict:
    """Hit / to verify the API is alive."""
    return _request('GET', '/')


def search_takers(query: str) -> list:
    """Find takers by name substring."""
    return _request('GET', '/takers/search', params={'q': query})


def search_keepers(query: str) -> list:
    """Find keepers by name substring."""
    return _request('GET', '/keepers/search', params={'q': query})


def list_takers(limit: int = 100, reliability: Optional[str] = None) -> list:
    """Top N takers, optionally filtered by reliability."""
    params = {'limit': limit}
    if reliability:
        params['reliability'] = reliability
    return _request('GET', '/takers', params=params)


def list_keepers(limit: int = 100, reliability: Optional[str] = None) -> list:
    params = {'limit': limit}
    if reliability:
        params['reliability'] = reliability
    return _request('GET', '/keepers', params=params)


def get_taker_profile(taker_id: int) -> dict:
    return _request('GET', f'/taker/{taker_id}')


def get_keeper_profile(keeper_id: int) -> dict:
    return _request('GET', f'/keeper/{keeper_id}')


def simulate(payload: dict) -> dict:
    """Run a penalty simulation. payload must match SimulateRequest schema."""
    return _request('POST', '/simulate', json=payload)
