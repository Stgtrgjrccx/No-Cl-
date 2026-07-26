# Poster Art (No-Signup) + Free-Tier Cost Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real poster art to No Clú's results with zero account creation required, and protect the shared free Gemini key with a daily cap that fails gracefully instead of breaking.

**Architecture:** Two independent additions to `server/main.py`. Poster art is a two-step fallback chain (Apple's iTunes Search API, then Wikipedia's page image) that runs regardless of whether a TMDB key is configured, replacing the TMDB-only poster support added earlier. The cost guard is a small in-process `DailyCap` counter checked before every `/identify` call.

**Tech Stack:** Python 3.9, FastAPI, httpx (async), pytest + pytest-httpx for tests (new dev dependencies — this project has no existing test suite).

## Global Constraints

- The app must stay **free for end users, permanently** — no user-facing subscriptions, no "bring your own key." (spec: hard requirement)
- Poster art must require **zero account/API key creation**, for any of movie/tv_show/anime. (spec: Design — poster art)
- Both poster lookup and the cost guard must be **fail-soft**: a failure must never surface as a raw error or block the core recognition result. (spec: Fail-soft requirement / Error handling summary)
- `DAILY_SCAN_CAP` env var, **default 100/day**. (spec: Design — free-tier cost guard)
- TMDB-based "where to watch" stays unchanged and optional, gated on `TMDB_API_KEY`. Poster art no longer depends on TMDB at all. (spec: Relationship to TMDB)

---

### Task 1: iTunes artwork URL upsizing helper

**Files:**
- Modify: `server/main.py` (add `_itunes_upsize` near the top-level helper functions, after `shrink`)
- Test: `server/tests/test_poster.py` (new file)
- Modify: `server/requirements.txt` (add test dependencies)

**Interfaces:**
- Produces: `_itunes_upsize(url: str, box: int = 1200) -> str` — used by Task 2.

- [ ] **Step 1: Add test dependencies**

Add to `server/requirements.txt` (append after the existing `anthropic>=0.69` line):

```
# Test-only dependencies (not needed to run the server).
pytest>=8.0
pytest-httpx>=0.30
```

- [ ] **Step 2: Install dependencies**

Run: `cd server && ./.venv/bin/pip install -r requirements.txt`
Expected: `pytest` and `pytest-httpx` install without errors.

- [ ] **Step 3: Write the failing test**

Create `server/tests/test_poster.py`:

```python
from main import _itunes_upsize


def test_itunes_upsize_replaces_100x100_with_larger_box():
    url = "https://is1-ssl.mzstatic.com/image/thumb/Music/v4/ab/cd/ef/100x100bb.jpg"
    assert _itunes_upsize(url) == (
        "https://is1-ssl.mzstatic.com/image/thumb/Music/v4/ab/cd/ef/1200x1200bb.jpg"
    )


def test_itunes_upsize_respects_custom_box_size():
    url = "https://example.com/100x100bb.jpg"
    assert _itunes_upsize(url, box=600) == "https://example.com/600x600bb.jpg"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd server && ./.venv/bin/pytest tests/test_poster.py -v`
Expected: FAIL — `ImportError: cannot import name '_itunes_upsize' from 'main'`

- [ ] **Step 5: Write minimal implementation**

In `server/main.py`, add immediately after the `shrink()` function (currently ending at the line before `def _loads_tolerant`):

```python
def _itunes_upsize(url: str, box: int = 1200) -> str:
    """Swap Apple's default 100x100 thumbnail for a much larger box.

    Apple's iTunes Search API always returns "100x100bb" in artworkUrl100
    regardless of the source image's real resolution; requesting a bigger
    box gets back the best resolution Apple actually has, up to `box`.
    """
    return url.replace("100x100bb", f"{box}x{box}bb")
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd server && ./.venv/bin/pytest tests/test_poster.py -v`
Expected: 2 passed

- [ ] **Step 7: Commit**

```bash
cd server && git add main.py requirements.txt tests/test_poster.py
git commit -m "feat: add iTunes artwork URL upsizing helper"
```

(If this isn't a git repo yet, skip this step — note it in the task's completion summary instead of failing on it.)

---

### Task 2: iTunes poster lookup

**Files:**
- Modify: `server/main.py` (add `itunes_poster`, after `_itunes_upsize`)
- Test: `server/tests/test_poster.py` (append)

**Interfaces:**
- Consumes: `_itunes_upsize(url, box=1200) -> str` (Task 1); `ScreenContent` (existing model, fields: `content_type`, `title`, `year`).
- Produces: `async itunes_poster(content: ScreenContent) -> Optional[str]` — used by Task 4.

- [ ] **Step 1: Write the failing tests**

Append to `server/tests/test_poster.py`:

```python
import asyncio

from main import ScreenContent, itunes_poster


def test_itunes_poster_returns_upsized_artwork(httpx_mock):
    httpx_mock.add_response(
        url="https://itunes.apple.com/search?term=Interstellar&media=movie&limit=5",
        json={"results": [{
            "releaseDate": "2014-11-05T00:00:00Z",
            "artworkUrl100": "https://example.com/is1/100x100bb.jpg",
        }]},
    )
    content = ScreenContent(content_type="movie", title="Interstellar", year=2014,
                             confidence="high", detail="")
    poster = asyncio.run(itunes_poster(content))
    assert poster == "https://example.com/is1/1200x1200bb.jpg"


def test_itunes_poster_prefers_matching_release_year(httpx_mock):
    httpx_mock.add_response(
        url="https://itunes.apple.com/search?term=Dune&media=movie&limit=5",
        json={"results": [
            {"releaseDate": "2021-10-22T00:00:00Z",
             "artworkUrl100": "https://example.com/2021/100x100bb.jpg"},
            {"releaseDate": "1984-12-14T00:00:00Z",
             "artworkUrl100": "https://example.com/1984/100x100bb.jpg"},
        ]},
    )
    content = ScreenContent(content_type="movie", title="Dune", year=1984,
                             confidence="high", detail="")
    poster = asyncio.run(itunes_poster(content))
    assert poster == "https://example.com/1984/1200x1200bb.jpg"


def test_itunes_poster_returns_none_when_no_results(httpx_mock):
    httpx_mock.add_response(
        url="https://itunes.apple.com/search?term=Totally+Made+Up+Title&media=movie&limit=5",
        json={"results": []},
    )
    content = ScreenContent(content_type="movie", title="Totally Made Up Title",
                             confidence="high", detail="")
    assert asyncio.run(itunes_poster(content)) is None


def test_itunes_poster_returns_none_on_network_failure(httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError("boom"))
    content = ScreenContent(content_type="movie", title="Interstellar",
                             confidence="high", detail="")
    assert asyncio.run(itunes_poster(content)) is None
```

Add `import httpx` near the top of `server/tests/test_poster.py` (needed for the last test).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd server && ./.venv/bin/pytest tests/test_poster.py -v`
Expected: 4 new FAILs — `ImportError: cannot import name 'itunes_poster' from 'main'`

- [ ] **Step 3: Write minimal implementation**

In `server/main.py`, add immediately after `_itunes_upsize`:

```python
async def itunes_poster(content: ScreenContent) -> Optional[str]:
    """Official cover art via Apple's public iTunes Search API. No key, no signup."""
    itunes_media = "movie" if content.content_type == "movie" else "tvShow"
    params = {"term": content.title, "media": itunes_media, "limit": 5}
    try:
        async with httpx.AsyncClient(timeout=4.0) as http:
            r = await http.get("https://itunes.apple.com/search", params=params)
            results = r.json().get("results", [])
    except Exception:
        return None

    if not results:
        return None
    if content.year:
        for result in results:
            if str(result.get("releaseDate", ""))[:4] == str(content.year):
                artwork = result.get("artworkUrl100")
                if artwork:
                    return _itunes_upsize(artwork)
    artwork = results[0].get("artworkUrl100")
    return _itunes_upsize(artwork) if artwork else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd server && ./.venv/bin/pytest tests/test_poster.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
cd server && git add main.py tests/test_poster.py
git commit -m "feat: add iTunes poster lookup with year-matching and fail-soft errors"
```

---

### Task 3: Wikipedia poster fallback

**Files:**
- Modify: `server/main.py` (add `wikipedia_poster`, after `itunes_poster`)
- Test: `server/tests/test_poster.py` (append)

**Interfaces:**
- Consumes: `ScreenContent` (existing model).
- Produces: `async wikipedia_poster(content: ScreenContent) -> Optional[str]` — used by Task 4.

- [ ] **Step 1: Write the failing tests**

Append to `server/tests/test_poster.py`:

```python
from main import wikipedia_poster


def test_wikipedia_poster_returns_thumbnail_source(httpx_mock):
    httpx_mock.add_response(
        url=(
            "https://en.wikipedia.org/w/api.php?action=query&format=json"
            "&prop=pageimages&piprop=thumbnail&pithumbsize=1000&redirects=1"
            "&titles=Jujutsu+Kaisen"
        ),
        json={"query": {"pages": {"123": {
            "title": "Jujutsu Kaisen",
            "thumbnail": {"source": "https://upload.wikimedia.org/wp/jjk.jpg"},
        }}}},
    )
    content = ScreenContent(content_type="anime", title="Jujutsu Kaisen",
                             confidence="high", detail="")
    poster = asyncio.run(wikipedia_poster(content))
    assert poster == "https://upload.wikimedia.org/wp/jjk.jpg"


def test_wikipedia_poster_returns_none_when_page_has_no_thumbnail(httpx_mock):
    httpx_mock.add_response(
        url=(
            "https://en.wikipedia.org/w/api.php?action=query&format=json"
            "&prop=pageimages&piprop=thumbnail&pithumbsize=1000&redirects=1"
            "&titles=Some+Obscure+Show"
        ),
        json={"query": {"pages": {"456": {"title": "Some Obscure Show"}}}},
    )
    content = ScreenContent(content_type="tv_show", title="Some Obscure Show",
                             confidence="low", detail="")
    assert asyncio.run(wikipedia_poster(content)) is None


def test_wikipedia_poster_returns_none_on_network_failure(httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError("boom"))
    content = ScreenContent(content_type="anime", title="Jujutsu Kaisen",
                             confidence="high", detail="")
    assert asyncio.run(wikipedia_poster(content)) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd server && ./.venv/bin/pytest tests/test_poster.py -v`
Expected: 3 new FAILs — `ImportError: cannot import name 'wikipedia_poster' from 'main'`

- [ ] **Step 3: Write minimal implementation**

In `server/main.py`, add immediately after `itunes_poster`:

```python
async def wikipedia_poster(content: ScreenContent) -> Optional[str]:
    """Fall back to a Wikipedia page's lead image. No key, no signup.

    Broader coverage than iTunes for anime, regional, and older titles —
    though the image isn't guaranteed to be an official poster.
    """
    params = {
        "action": "query", "format": "json", "prop": "pageimages",
        "piprop": "thumbnail", "pithumbsize": "1000", "redirects": 1,
        "titles": content.title,
    }
    try:
        async with httpx.AsyncClient(timeout=4.0) as http:
            r = await http.get("https://en.wikipedia.org/w/api.php", params=params)
            pages = r.json().get("query", {}).get("pages", {})
    except Exception:
        return None

    for page in pages.values():
        thumbnail = page.get("thumbnail", {}).get("source")
        if thumbnail:
            return thumbnail
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd server && ./.venv/bin/pytest tests/test_poster.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
cd server && git add main.py tests/test_poster.py
git commit -m "feat: add Wikipedia poster fallback for anime/regional titles"
```

---

### Task 4: Combine into `fetch_poster`, remove dead TMDB poster code, wire into `/identify`

**Files:**
- Modify: `server/main.py:256-305` (remove `TMDB_IMAGE_BASE` and poster fields from `tmdb_where_to_watch`)
- Modify: `server/main.py` (add `fetch_poster`, after `wikipedia_poster`)
- Modify: `server/main.py:499-515` (wire `poster` into `/identify` response from `fetch_poster` instead of `watch.get("poster")`)
- Test: `server/tests/test_poster.py` (append)

**Interfaces:**
- Consumes: `itunes_poster(content) -> Optional[str]` (Task 2); `wikipedia_poster(content) -> Optional[str]` (Task 3).
- Produces: `async fetch_poster(content: ScreenContent) -> Optional[str]` — used by the `/identify` route.

- [ ] **Step 1: Write the failing tests**

Append to `server/tests/test_poster.py`:

```python
from unittest.mock import AsyncMock, patch

from main import fetch_poster


def test_fetch_poster_uses_itunes_result_when_found():
    content = ScreenContent(content_type="movie", title="Interstellar", year=2014,
                             confidence="high", detail="")
    with patch("main.itunes_poster", AsyncMock(return_value="https://itunes.example/poster.jpg")), \
         patch("main.wikipedia_poster", AsyncMock(return_value="https://wikipedia.example/poster.jpg")):
        poster = asyncio.run(fetch_poster(content))
    assert poster == "https://itunes.example/poster.jpg"


def test_fetch_poster_falls_back_to_wikipedia_when_itunes_empty():
    content = ScreenContent(content_type="anime", title="Some Anime",
                             confidence="high", detail="")
    with patch("main.itunes_poster", AsyncMock(return_value=None)), \
         patch("main.wikipedia_poster", AsyncMock(return_value="https://wikipedia.example/poster.jpg")):
        poster = asyncio.run(fetch_poster(content))
    assert poster == "https://wikipedia.example/poster.jpg"


def test_fetch_poster_returns_none_when_both_sources_empty():
    content = ScreenContent(content_type="movie", title="Nonexistent",
                             confidence="low", detail="")
    with patch("main.itunes_poster", AsyncMock(return_value=None)), \
         patch("main.wikipedia_poster", AsyncMock(return_value=None)):
        assert asyncio.run(fetch_poster(content)) is None


def test_fetch_poster_skips_lookup_for_non_poster_content_types():
    content = ScreenContent(content_type="sports", title="Big Match",
                             confidence="medium", detail="")
    with patch("main.itunes_poster", AsyncMock(return_value="should not be called")) as mock_itunes:
        assert asyncio.run(fetch_poster(content)) is None
    mock_itunes.assert_not_called()


def test_fetch_poster_never_raises_even_if_a_source_blows_up():
    content = ScreenContent(content_type="movie", title="Interstellar",
                             confidence="high", detail="")
    with patch("main.itunes_poster", AsyncMock(side_effect=RuntimeError("boom"))):
        assert asyncio.run(fetch_poster(content)) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd server && ./.venv/bin/pytest tests/test_poster.py -v`
Expected: 5 new FAILs — `ImportError: cannot import name 'fetch_poster' from 'main'`

- [ ] **Step 3: Write minimal implementation**

In `server/main.py`, add immediately after `wikipedia_poster`:

```python
async def fetch_poster(content: ScreenContent) -> Optional[str]:
    """Poster art for any recognized movie/TV/anime — no account required, ever.

    Tries Apple's iTunes catalog first (best quality), falls back to
    Wikipedia's page image for anything iTunes doesn't have. Never raises:
    a poster lookup failing must never break the recognition result.
    """
    if content.content_type not in ("movie", "tv_show", "anime"):
        return None
    try:
        poster = await itunes_poster(content)
        if poster:
            return poster
        return await wikipedia_poster(content)
    except Exception:
        return None
```

Now remove the TMDB-based poster code, since poster art no longer depends on TMDB. Replace this block in `tmdb_where_to_watch` (currently lines 256-305):

```python
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


async def tmdb_where_to_watch(content: ScreenContent, country: str) -> Optional[dict]:
    """Look up streaming availability + poster art for movies/TV in the given country."""
    if not TMDB_API_KEY or content.content_type not in ("movie", "tv_show", "anime"):
        return None

    media = "movie" if content.content_type == "movie" else "tv"
    params: dict = {"api_key": TMDB_API_KEY, "query": content.title}
    if content.year:
        params["primary_release_year" if media == "movie" else "first_air_date_year"] = content.year

    try:
        async with httpx.AsyncClient(timeout=4.0) as http:
            r = await http.get(f"https://api.themoviedb.org/3/search/{media}", params=params)
            results = r.json().get("results", [])
            if not results and ("primary_release_year" in params or "first_air_date_year" in params):
                # Retry without the year — the AI's year guess can be off by one.
                params.pop("primary_release_year", None)
                params.pop("first_air_date_year", None)
                r = await http.get(f"https://api.themoviedb.org/3/search/{media}", params=params)
                results = r.json().get("results", [])
            if not results:
                return None

            match = results[0]
            poster_path = match.get("poster_path")
            poster = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None

            tmdb_id = match["id"]
            r = await http.get(
                f"https://api.themoviedb.org/3/{media}/{tmdb_id}/watch/providers",
                params={"api_key": TMDB_API_KEY},
            )
            region = r.json().get("results", {}).get(country)
            if not region:
                return {"stream": [], "rent": [], "buy": [], "link": None, "poster": poster,
                        "note": f"No providers listed for {country}"}

            def names(key: str) -> list:
                return [p["provider_name"] for p in region.get(key, [])]

            return {
                "stream": names("flatrate"),
                "rent": names("rent"),
                "buy": names("buy"),
                "link": region.get("link"),
                "poster": poster,
            }
    except Exception:
        return None
```

with:

```python
async def tmdb_where_to_watch(content: ScreenContent, country: str) -> Optional[dict]:
    """Look up streaming availability for movies/TV in the given country."""
    if not TMDB_API_KEY or content.content_type not in ("movie", "tv_show", "anime"):
        return None

    media = "movie" if content.content_type == "movie" else "tv"
    params: dict = {"api_key": TMDB_API_KEY, "query": content.title}
    if content.year:
        params["primary_release_year" if media == "movie" else "first_air_date_year"] = content.year

    try:
        async with httpx.AsyncClient(timeout=4.0) as http:
            r = await http.get(f"https://api.themoviedb.org/3/search/{media}", params=params)
            results = r.json().get("results", [])
            if not results and ("primary_release_year" in params or "first_air_date_year" in params):
                # Retry without the year — the AI's year guess can be off by one.
                params.pop("primary_release_year", None)
                params.pop("first_air_date_year", None)
                r = await http.get(f"https://api.themoviedb.org/3/search/{media}", params=params)
                results = r.json().get("results", [])
            if not results:
                return None

            tmdb_id = results[0]["id"]
            r = await http.get(
                f"https://api.themoviedb.org/3/{media}/{tmdb_id}/watch/providers",
                params={"api_key": TMDB_API_KEY},
            )
            region = r.json().get("results", {}).get(country)
            if not region:
                return {"stream": [], "rent": [], "buy": [], "link": None,
                        "note": f"No providers listed for {country}"}

            def names(key: str) -> list:
                return [p["provider_name"] for p in region.get(key, [])]

            return {
                "stream": names("flatrate"),
                "rent": names("rent"),
                "buy": names("buy"),
                "link": region.get("link"),
            }
    except Exception:
        return None
```

Finally, wire `fetch_poster` into `/identify`. Replace (currently lines 499-515):

```python
    watch = await tmdb_where_to_watch(content, resolved_country)

    return {
        "identified": True,
        "type": content.content_type,
        "title": content.title,
        "year": content.year,
        "season": content.season,
        "episode": content.episode,
        "confidence": content.confidence,
        "detail": content.detail,
        "country": resolved_country,
        "watch": watch,
        "poster": watch.get("poster") if watch else None,
        "summary": build_summary(content, watch, resolved_country),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
```

with:

```python
    watch = await tmdb_where_to_watch(content, resolved_country)
    poster = await fetch_poster(content)

    return {
        "identified": True,
        "type": content.content_type,
        "title": content.title,
        "year": content.year,
        "season": content.season,
        "episode": content.episode,
        "confidence": content.confidence,
        "detail": content.detail,
        "country": resolved_country,
        "watch": watch,
        "poster": poster,
        "summary": build_summary(content, watch, resolved_country),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd server && ./.venv/bin/pytest tests/test_poster.py -v`
Expected: 14 passed

- [ ] **Step 5: Sanity-check the whole server still imports cleanly**

Run: `cd server && ./.venv/bin/python -c "import main; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 6: Commit**

```bash
cd server && git add main.py tests/test_poster.py
git commit -m "feat: wire iTunes/Wikipedia poster chain into /identify, drop TMDB poster code"
```

---

### Task 5: Daily scan cap

**Files:**
- Modify: `server/main.py` (add `from datetime import date, datetime` import, `DailyCap` class + module-level instance, gate in `/identify`)
- Modify: `server/.env.example` (document `DAILY_SCAN_CAP`)
- Test: `server/tests/test_daily_cap.py` (new file)

**Interfaces:**
- Produces: `DailyCap(limit: int, now_fn: Callable[[], datetime] = None)` with `.record() -> None` and `.exceeded -> bool` — used by the `/identify` route.

- [ ] **Step 1: Write the failing tests**

Create `server/tests/test_daily_cap.py`:

```python
from datetime import datetime

from main import DailyCap


def test_not_exceeded_before_any_calls():
    cap = DailyCap(limit=2)
    assert cap.exceeded is False


def test_exceeded_once_limit_is_reached():
    cap = DailyCap(limit=2)
    cap.record()
    cap.record()
    assert cap.exceeded is True


def test_not_exceeded_one_below_limit():
    cap = DailyCap(limit=2)
    cap.record()
    assert cap.exceeded is False


def test_resets_when_the_day_rolls_over():
    day_one = [datetime(2026, 7, 22, 23, 59)]
    cap = DailyCap(limit=1, now_fn=lambda: day_one[0])
    cap.record()
    assert cap.exceeded is True

    day_one[0] = datetime(2026, 7, 23, 0, 1)  # next day
    assert cap.exceeded is False, "should reset once the calendar date changes"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && ./.venv/bin/pytest tests/test_daily_cap.py -v`
Expected: FAIL — `ImportError: cannot import name 'DailyCap' from 'main'`

- [ ] **Step 3: Write minimal implementation**

In `server/main.py`, change the import line (near the top):

```python
from base64 import standard_b64encode
```

to:

```python
from base64 import standard_b64encode
from datetime import datetime
```

Then add this class immediately after the `ProviderError` class definition:

```python
class DailyCap:
    """In-process counter for /identify calls today. Resets when the date
    rolls over. `now_fn` is injectable so tests don't depend on real time."""

    def __init__(self, limit: int, now_fn=None):
        self.limit = limit
        self._now_fn = now_fn or datetime.now
        self._day = self._now_fn().date()
        self._count = 0

    def _maybe_reset(self):
        today = self._now_fn().date()
        if today != self._day:
            self._day = today
            self._count = 0

    @property
    def exceeded(self) -> bool:
        self._maybe_reset()
        return self._count >= self.limit

    def record(self):
        self._maybe_reset()
        self._count += 1


daily_cap = DailyCap(limit=int(os.getenv("DAILY_SCAN_CAP", "100")))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && ./.venv/bin/pytest tests/test_daily_cap.py -v`
Expected: 4 passed

- [ ] **Step 5: Wire the cap into `/identify`**

In `server/main.py`, replace (currently lines 487-491):

```python
@app.post("/identify")
async def identify(request: Request, image: UploadFile = File(...), country: Optional[str] = None):
    started = time.monotonic()
    resolved_country = await resolve_country(request, country)
    b64 = standard_b64encode(shrink(await image.read())).decode()

    try:
        content = await identify_content(b64)
```

with:

```python
@app.post("/identify")
async def identify(request: Request, image: UploadFile = File(...), country: Optional[str] = None):
    started = time.monotonic()
    if daily_cap.exceeded:
        return {"identified": False,
                "summary": "🌙 No Clú's free daily limit is used up — try again after midnight!",
                "elapsed_seconds": round(time.monotonic() - started, 2)}
    daily_cap.record()

    resolved_country = await resolve_country(request, country)
    b64 = standard_b64encode(shrink(await image.read())).decode()

    try:
        content = await identify_content(b64)
```

- [ ] **Step 6: Document the new env var**

Append to `server/.env.example` (after the `GEMINI_MODEL`/`CLAUDE_MODEL` comment block at the end):

```
# Daily safety cap on /identify calls, shared across everyone who uses this
# server. Keeps the app free and working (never a broken error) instead of
# silently exhausting Google's free quota. Raise it if you check your real
# limit at https://aistudio.google.com/rate-limit and it's higher than this.
DAILY_SCAN_CAP=100
```

- [ ] **Step 7: Verify end-to-end with a tiny cap**

Run: `cd server && DAILY_SCAN_CAP=1 ./.venv/bin/python -c "
from main import daily_cap
print('exceeded before any calls:', daily_cap.exceeded)
daily_cap.record()
print('exceeded after 1 call (cap=1):', daily_cap.exceeded)
"`
Expected:
```
exceeded before any calls: False
exceeded after 1 call (cap=1): True
```

- [ ] **Step 8: Run the full test suite**

Run: `cd server && ./.venv/bin/pytest tests/ -v`
Expected: 18 passed (14 from `test_poster.py` + 4 from `test_daily_cap.py`)

- [ ] **Step 9: Commit**

```bash
cd server && git add main.py .env.example tests/test_daily_cap.py
git commit -m "feat: add daily scan cap with graceful limit message"
```

---

## Self-Review

**Spec coverage:**
- Poster art, no signup, any of movie/tv_show/anime → Tasks 1–4 ✅
- Fail-soft on poster lookup → Task 4 (`fetch_poster` try/except + tests) ✅
- Poster no longer depends on TMDB → Task 4 (removes `poster` from `tmdb_where_to_watch`) ✅
- Daily cap with configurable ceiling, default 100 → Task 5 ✅
- Graceful message instead of raw error when cap hit → Task 5, Step 5 ✅
- Free-for-users hard requirement → no task introduces any user-facing payment or key requirement; `DailyCap` only ever produces a "try again tomorrow" message, never a paywall ✅
- Verification plan (known movie / anime / nonsense title / cap exhaustion) → covered by the automated tests in Tasks 2–5, which are more repeatable than the spec's manual-curl suggestion and don't burn real API quota ✅

**Placeholder scan:** No TBD/TODO; every step has complete, runnable code.

**Type consistency:** `ScreenContent` fields (`content_type`, `title`, `year`, `confidence`, `detail`) used consistently across all task's test code, matching the existing model in `main.py`. `fetch_poster`/`itunes_poster`/`wikipedia_poster` all take `ScreenContent` and return `Optional[str]` consistently across Tasks 2–4.
