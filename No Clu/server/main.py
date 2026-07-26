"""No Clú — one-tap "Shazam for your screen".

POST /identify with a screenshot -> a vision AI identifies what's playing,
TMDB tells you where to stream it in your country.

The vision "brain" is pluggable via the PROVIDER setting in .env:
  PROVIDER=gemini     -> Google Gemini free tier (no credit card)  [default]
  PROVIDER=anthropic  -> Anthropic Claude (paid, sharper on obscure content)
Switching later is a one-word change plus the matching API key.
"""

import io
import ipaddress
import json
import os
import re
import time
from base64 import standard_b64encode
from datetime import datetime
from typing import Literal, Optional
from urllib.parse import quote_plus

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from PIL import Image, ImageDraw
from pydantic import BaseModel, ValidationError

import auth
import db

load_dotenv()

app = FastAPI(title="No Clú")


@app.on_event("startup")
def _startup():
    db.init_db()


def current_user_id(request: Request) -> Optional[int]:
    """The signed-in user's id from their session cookie, or None."""
    return auth.read_session(request.cookies.get(auth.SESSION_COOKIE, ""))


def _set_session_cookie(request: Request, response: Response, user_id: int) -> None:
    # httponly so page scripts can't read it; 30-day lifetime.
    # Secure flag from the ACTUAL request scheme (honoring a proxy's
    # X-Forwarded-Proto) — never from an unrelated setting, so a Secure cookie
    # is only sent when the connection is really HTTPS.
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    response.set_cookie(
        auth.SESSION_COOKIE, auth.make_session(user_id),
        max_age=60 * 60 * 24 * 30, httponly=True, samesite="lax",
        secure=(scheme == "https"),
    )

PROVIDER = os.getenv("PROVIDER", "gemini").lower().strip()
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
DEFAULT_COUNTRY = os.getenv("DEFAULT_COUNTRY", "US")
# iCloud share link of the ready-made "No Clú" Shortcut template. Set this in the
# host's env once the template is shared; the /shortcut page then offers one-tap
# install. Left blank => the page falls back to short manual build instructions.
ICLOUD_SHORTCUT_URL = os.getenv("ICLOUD_SHORTCUT_URL", "").strip()

# --- Provider settings -------------------------------------------------------
# Gemini: gemini-flash-latest is fast, multimodal, and free-tier friendly (some
# accounts have zero free quota on the pinned gemini-2.0-flash, so "latest" is a
# safer default). Get a key (no card) at https://aistudio.google.com/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
# Anthropic: claude-opus-4-8 is the most accurate; claude-haiku-4-5 the fastest.
ANTHROPIC_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")

# Vision models read a ~1.15MP image at full fidelity; anything bigger just adds
# upload + processing latency, so downscale before sending.
MAX_IMAGE_EDGE = 1568
JPEG_QUALITY = 80

IDENTIFY_PROMPT = """\
This is a single frame captured from someone's phone or TV screen. Identify what \
content is being watched. Look at everything: the video frame itself, actors' faces, \
UI elements (Netflix/YouTube/Prime player chrome, titles, captions, channel names, \
progress bars, watermarks, scoreboards).

Rules:
- Give the official title, not a description of the scene.
- For TV shows and anime, include season/episode if the UI shows it or you can tell \
from the scene; otherwise leave them null.
- For YouTube, the title is the video title and detail should name the channel if visible.
- For live sports, title is the matchup/event (e.g. "India vs Australia — 3rd ODI").
- If you genuinely cannot identify it, set content_type to "other", title to your best \
guess of what kind of content it is, and confidence to "low".
- detail: one short sentence of context (what scene / why you're confident / what gave it away).

Respond with ONLY a JSON object (no markdown, no code fences) with exactly these keys:
  content_type: one of "movie","tv_show","anime","youtube_video","sports","music_video","game","other"
  title: string
  year: integer or null
  season: integer or null
  episode: integer or null
  confidence: one of "high","medium","low"
  detail: string
"""


class ScreenContent(BaseModel):
    content_type: Literal[
        "movie", "tv_show", "anime", "youtube_video",
        "sports", "music_video", "game", "other",
    ] = "other"
    title: str = "Unknown"
    year: Optional[int] = None
    season: Optional[int] = None
    episode: Optional[int] = None
    confidence: Literal["high", "medium", "low"] = "low"
    detail: str = ""


# Strict shape Gemini must return — prevents malformed/truncated JSON.
GEMINI_SCHEMA = {
    "type": "object",
    "properties": {
        "content_type": {"type": "string", "enum": [
            "movie", "tv_show", "anime", "youtube_video",
            "sports", "music_video", "game", "other"]},
        "title": {"type": "string"},
        "year": {"type": "integer", "nullable": True},
        "season": {"type": "integer", "nullable": True},
        "episode": {"type": "integer", "nullable": True},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "detail": {"type": "string"},
    },
    "required": ["content_type", "title", "confidence", "detail"],
    "propertyOrdering": ["content_type", "title", "year", "season",
                         "episode", "confidence", "detail"],
}


class ProviderError(Exception):
    """Raised with a user-facing message when the vision call fails."""


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


def shrink(raw: bytes) -> bytes:
    img = Image.open(io.BytesIO(raw))
    img.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
    if img.mode != "RGB":
        img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


def _itunes_upsize(url: str, box: int = 1200) -> str:
    """Swap Apple's default 100x100 thumbnail for a much larger box.

    Apple's iTunes Search API always returns "100x100bb" in artworkUrl100
    regardless of the source image's real resolution; requesting a bigger
    box gets back the best resolution Apple actually has, up to `box`.
    """
    return url.replace("100x100bb", f"{box}x{box}bb")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _title_qualifies(kind: str, track: str, collection: str, title: str) -> bool:
    """True only when a catalog entry is confidently the title we're after.

    Kind-aware on purpose:
    - Movies: the track name must match *exactly* (normalized), so "Oppenheimer"
      never latches onto "Oppenheimer: The Real Story".
    - TV/anime: the SHOW name lives in the collection (and a season's track) —
      NOT the episode's track. An episode of an unrelated show can be titled
      "Stranger Things", so matching an episode's track name would attach the
      wrong poster. We match the show via a collection prefix instead.

    A missing poster is a graceful non-issue; a mismatched one is a visible bug.
    """
    t = _norm(title)
    if not t:
        return False
    if kind == "feature-movie":
        return _norm(track) == t
    # tv-season / tv-episode
    if _norm(collection).startswith(t):
        return True
    if kind == "tv-season" and _norm(track).startswith(t):
        return True
    return False


async def itunes_poster(content: ScreenContent) -> Optional[str]:
    """Official cover art via Apple's public iTunes Search API. No key, no signup.

    Apple's `media=movie` filter is unreliable (returns 0 results at times),
    so we search unfiltered and select client-side by `kind` + strict title
    match.
    """
    movie_kinds = {"feature-movie"}
    tv_kinds = {"tv-season", "tv-episode"}
    wanted = movie_kinds if content.content_type == "movie" else (movie_kinds | tv_kinds)

    try:
        async with httpx.AsyncClient(timeout=4.0) as http:
            r = await http.get("https://itunes.apple.com/search",
                               params={"term": content.title, "limit": 25})
            results = r.json().get("results", [])
    except Exception:
        return None

    candidates = [
        res for res in results
        if res.get("kind") in wanted
        and res.get("artworkUrl100")
        and _title_qualifies(res.get("kind"), res.get("trackName") or "",
                             res.get("collectionName") or "", content.title)
    ]

    if not candidates:
        return None
    if content.year:
        for res in candidates:
            if str(res.get("releaseDate", ""))[:4] == str(content.year):
                return _itunes_upsize(res["artworkUrl100"])
    return _itunes_upsize(candidates[0]["artworkUrl100"])


async def fetch_poster(content: ScreenContent) -> Optional[str]:
    """Poster art for any recognized movie/TV/anime — no account required, ever.

    Uses Apple's iTunes catalog (high quality, no signup) with strict title
    matching so we never show the wrong poster. Never raises: a poster lookup
    failing must never break the recognition result. Returns None when there's
    no confident match — the result card simply shows no poster.
    """
    if content.content_type not in ("movie", "tv_show", "anime"):
        return None
    try:
        return await itunes_poster(content)
    except Exception:
        return None


def _loads_tolerant(text: str) -> dict:
    """json.loads, but repair the common truncation cases first.

    Gemini (and LLMs generally) occasionally emit JSON that's cut a few chars
    short: a dangling string or a missing closing brace/bracket. We balance
    those so a near-complete reply still yields a usable result.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        t = text.rstrip().rstrip(",")
        if t.count('"') % 2 == 1:      # unterminated string -> close it
            t += '"'
        t += "]" * (t.count("[") - t.count("]"))   # balance brackets
        t += "}" * (t.count("{") - t.count("}"))   # balance braces
        return json.loads(t)


def _parse_json_blob(text: str) -> ScreenContent:
    """Tolerantly parse a model's JSON reply into ScreenContent."""
    text = text.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fences if the model added them.
        text = text.split("```", 2)[1] if "```" in text[3:] else text
        text = text.lstrip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip("`").strip()
    start = text.find("{")
    if start != -1:
        text = text[start:]
    try:
        return ScreenContent(**_loads_tolerant(text))
    except (json.JSONDecodeError, ValidationError, TypeError):
        raise ProviderError("🔍 Couldn't read the AI's answer — try again.")


# --- Gemini (free) -----------------------------------------------------------
async def identify_gemini(b64: str) -> ScreenContent:
    if not GEMINI_API_KEY:
        raise ProviderError("⚠️ No Clú: Gemini key missing — add GEMINI_API_KEY to server .env")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent")
    body = {
        "contents": [{
            "parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
                {"text": IDENTIFY_PROMPT},
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_SCHEMA,
            "temperature": 0,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            r = await http.post(url, params={"key": GEMINI_API_KEY}, json=body)
    except httpx.HTTPError:
        raise ProviderError("⚠️ No Clú: couldn't reach Google — check the Mac's internet")

    if r.status_code in (400, 401, 403):
        raise ProviderError("⚠️ No Clú: Gemini key is missing or invalid — check server .env")
    if r.status_code == 429:
        raise ProviderError("⚠️ No Clú: hit Gemini's free daily limit — try again later")
    if r.status_code != 200:
        raise ProviderError(f"⚠️ No Clú: Gemini error ({r.status_code}) — try again")

    try:
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, ValueError):
        raise ProviderError("🔍 Gemini didn't return a result — try again.")
    return _parse_json_blob(text)


# --- Anthropic (paid) --------------------------------------------------------
async def identify_anthropic(b64: str) -> ScreenContent:
    import anthropic  # imported lazily so Gemini-only users needn't configure it
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ProviderError("⚠️ No Clú: Anthropic key missing — add ANTHROPIC_API_KEY to server .env")
    client = anthropic.AsyncAnthropic()
    try:
        response = await client.messages.parse(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": IDENTIFY_PROMPT},
                ],
            }],
            output_format=ScreenContent,
        )
    except anthropic.AuthenticationError:
        raise ProviderError("⚠️ No Clú: Anthropic key is missing or invalid — check server .env")
    except anthropic.RateLimitError:
        raise ProviderError("⚠️ No Clú: rate limited — wait a moment and try again")
    except anthropic.APIError as e:
        raise ProviderError(f"⚠️ No Clú: Anthropic error ({getattr(e, 'status_code', '?')}) — try again")

    content = response.parsed_output
    if content is None:
        raise ProviderError("🔍 Couldn't analyze that screen — try again.")
    return content


async def identify_content(b64: str) -> ScreenContent:
    if PROVIDER == "anthropic":
        return await identify_anthropic(b64)
    if PROVIDER == "gemini":
        return await identify_gemini(b64)
    raise ProviderError(f"⚠️ No Clú: unknown PROVIDER '{PROVIDER}' — use 'gemini' or 'anthropic'")


async def resolve_country(request: Request, country: Optional[str]) -> str:
    if country:
        c = country.strip().upper()
        if len(c) == 2 and c.isalpha():  # only a real 2-letter code; else fall through
            return c
    # When deployed on the public internet, geolocate the caller's IP.
    ip = request.client.host if request.client else ""
    try:
        if ip and not ipaddress.ip_address(ip).is_private:
            async with httpx.AsyncClient(timeout=1.5) as http:
                r = await http.get(f"https://ipapi.co/{ip}/country/")
                if r.status_code == 200 and len(r.text.strip()) == 2:
                    return r.text.strip().upper()
    except Exception:
        pass
    return DEFAULT_COUNTRY


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

            # Prefer a result whose title actually matches (TMDB relevance can
            # rank a same-keyword title first, e.g. "Avatar" vs "Avatar: TLA").
            # Better to show nothing than the wrong title's streaming info.
            want = _norm(content.title)
            match = next(
                (res for res in results
                 if _norm(res.get("title") or res.get("name") or "") == want),
                results[0] if _norm(results[0].get("title") or results[0].get("name") or "") == want else None,
            )
            if match is None:
                return None

            tmdb_id = match["id"]
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


# Search/deep links per streaming service, so each named provider is tappable.
# {q} is filled with the URL-encoded title; {cc} with the 2-letter country.
PROVIDER_LINKS = {
    "Netflix": "https://www.netflix.com/search?q={q}",
    "Amazon Prime Video": "https://www.primevideo.com/search?phrase={q}",
    "Prime Video": "https://www.primevideo.com/search?phrase={q}",
    "Amazon Video": "https://www.primevideo.com/search?phrase={q}",
    "Disney Plus Hotstar": "https://www.hotstar.com/{cc}/search?q={q}",
    "JioHotstar": "https://www.hotstar.com/{cc}/search?q={q}",
    "Jio Hotstar": "https://www.hotstar.com/{cc}/search?q={q}",
    "Hotstar": "https://www.hotstar.com/{cc}/search?q={q}",
    "JioCinema": "https://www.jiocinema.com/search/{q}",
    "Zee5": "https://www.zee5.com/search?q={q}",
    "ZEE5": "https://www.zee5.com/search?q={q}",
    "Sony Liv": "https://www.sonyliv.com/search/{q}",
    "SonyLIV": "https://www.sonyliv.com/search/{q}",
    "Apple TV": "https://tv.apple.com/{cc}/search?term={q}",
    "Apple TV Plus": "https://tv.apple.com/{cc}/search?term={q}",
    "YouTube": "https://www.youtube.com/results?search_query={q}",
    "Crunchyroll": "https://www.crunchyroll.com/search?q={q}",
    "Netflix Kids": "https://www.netflix.com/search?q={q}",
}


def provider_link(name: str, title: str, country: str) -> str:
    """A tappable link for a named provider — its own search page if we know it,
    else a JustWatch fallback so the link is never dead."""
    q = quote_plus(title)
    tmpl = PROVIDER_LINKS.get(name)
    if tmpl:
        return tmpl.format(q=q, cc=country.lower())
    return f"https://www.justwatch.com/{country.lower()}/search?q={q}"


def justwatch_url(content: ScreenContent, country: str) -> Optional[str]:
    """JustWatch's own where-to-watch page for this title in the user's country.

    Free, no key, legitimate (a public web link). It lists every platform the
    title streams on in that region with links — exactly the 'where to watch'
    answer, and it works even without a TMDB key configured.
    """
    if content.content_type not in ("movie", "tv_show", "anime"):
        return None
    return f"https://www.justwatch.com/{country.lower()}/search?q={quote_plus(content.title)}"


def build_summary(content: ScreenContent, watch: Optional[dict], country: str) -> str:
    icons = {"movie": "🎬", "tv_show": "📺", "anime": "🎌", "youtube_video": "▶️",
             "sports": "🏆", "music_video": "🎵", "game": "🎮", "other": "🔍"}
    title = content.title
    if content.year:
        title += f" ({content.year})"
    if content.season and content.episode:
        title += f" — S{content.season}E{content.episode}"

    line = f"{icons[content.content_type]} {title}"
    if content.confidence == "low":
        line += " (best guess)"

    if watch:
        if watch["stream"]:
            line += f"\nStreaming in {country} on: " + ", ".join(watch["stream"][:4])
        elif watch["rent"] or watch["buy"]:
            line += f"\nRent/buy in {country} on: " + ", ".join((watch["rent"] or watch["buy"])[:4])
        else:
            line += f"\nNot available to stream in {country}."
    return line


DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>No Clú — demo</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: linear-gradient(160deg, #1a1030, #0b0b16); color: #eee;
         min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 24px; }
  .wrap { width: 100%; max-width: 420px; }
  h1 { font-size: 34px; margin: 0 0 4px; letter-spacing: -0.5px; }
  .tag { color: #9a9ab0; margin: 0 0 22px; font-size: 15px; }
  .drop { border: 2px dashed #4a4a70; border-radius: 18px; padding: 34px 18px; text-align: center;
          cursor: pointer; transition: .15s; background: rgba(255,255,255,.03); }
  .drop:hover, .drop.over { border-color: #b98cff; background: rgba(185,140,255,.08); }
  .drop input { display: none; }
  .drop .big { font-size: 44px; }
  .drop .hint { color: #9a9ab0; font-size: 14px; margin-top: 8px; }
  .row { display: flex; gap: 10px; align-items: center; margin: 16px 0; }
  .row label { font-size: 14px; color: #b8b8cc; }
  select, button { font-size: 15px; border-radius: 12px; border: none; padding: 12px 14px; }
  select { background: #23233a; color: #eee; flex: 1; }
  button { background: #b98cff; color: #1a1030; font-weight: 700; cursor: pointer; width: 100%; }
  button:disabled { opacity: .5; cursor: default; }
  img.preview { width: 100%; border-radius: 14px; margin-top: 14px; display: none; }
  .notif { margin-top: 20px; background: rgba(255,255,255,.07); border-radius: 18px; overflow: hidden;
           display: none; border: 1px solid rgba(255,255,255,.08); }
  .notif .poster { width: 100%; display: none; }
  .notif .body { padding: 18px 20px; }
  .notif .t { font-size: 20px; font-weight: 700; line-height: 1.25; }
  .notif .conf { font-size: 12px; color: #b98cff; margin-top: 4px; text-transform: uppercase; letter-spacing: .5px; }
  .notif .detail { color: #b8b8cc; font-size: 14px; margin-top: 10px; }
  .notif .watch { margin-top: 12px; }
  .chip { display: inline-block; background: #23233a; border-radius: 999px; padding: 5px 11px;
          font-size: 13px; margin: 4px 6px 0 0; }
  .notif .time { color: #6a6a80; font-size: 12px; margin-top: 12px; }
  .spin { text-align: center; color: #b8b8cc; margin-top: 20px; display: none; }
</style>
</head>
<body>
  <div class="wrap">
    <h1>No Clú 🔍</h1>
    <p class="tag">Drop in a screenshot — see what it is and where to watch it.</p>

    <label class="drop" id="drop">
      <div class="big">📸</div>
      <div>Click to choose a screenshot<br>or drag one here</div>
      <div class="hint">a movie, show, anime, sports, YouTube…</div>
      <input type="file" id="file" accept="image/*">
    </label>
    <img class="preview" id="preview">

    <div class="row">
      <label for="country">Country</label>
      <select id="country">
        <option value="IN">India (IN)</option>
        <option value="US">United States (US)</option>
        <option value="GB">United Kingdom (GB)</option>
        <option value="CA">Canada (CA)</option>
        <option value="AU">Australia (AU)</option>
      </select>
    </div>
    <button id="go" disabled>Identify</button>

    <div class="spin" id="spin">🧠 Looking at your screen…</div>
    <div class="notif" id="notif">
      <img class="poster" id="posterImg" alt="">
      <div class="body">
        <div class="t" id="title"></div>
        <div class="conf" id="conf"></div>
        <div class="detail" id="detail"></div>
        <div class="watch" id="watch"></div>
        <div class="time" id="time"></div>
      </div>
    </div>
  </div>

<script>
  const fileInput = document.getElementById('file');
  const drop = document.getElementById('drop');
  const preview = document.getElementById('preview');
  const go = document.getElementById('go');
  const spin = document.getElementById('spin');
  const notif = document.getElementById('notif');
  let chosen = null;

  function setFile(f) {
    chosen = f;
    go.disabled = !f;
    if (f) { preview.src = URL.createObjectURL(f); preview.style.display = 'block'; }
  }
  fileInput.addEventListener('change', e => setFile(e.target.files[0]));
  ['dragover','dragenter'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => { if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });

  go.addEventListener('click', async () => {
    if (!chosen) return;
    notif.style.display = 'none'; spin.style.display = 'block'; go.disabled = true;
    const country = document.getElementById('country').value;
    const fd = new FormData();
    fd.append('image', chosen);
    try {
      const r = await fetch('/identify?country=' + country, { method: 'POST', body: fd });
      const d = await r.json();
      spin.style.display = 'none';
      const posterImg = document.getElementById('posterImg');
      if (d.poster) { posterImg.src = d.poster; posterImg.style.display = 'block'; }
      else { posterImg.removeAttribute('src'); posterImg.style.display = 'none'; }
      document.getElementById('title').textContent = d.summary ? d.summary.split('\\n')[0] : 'No result';
      document.getElementById('conf').textContent = d.confidence ? (d.confidence + ' confidence') : '';
      document.getElementById('detail').textContent = d.detail || '';
      const w = document.getElementById('watch'); w.innerHTML = '';
      if (d.watch && d.watch.stream && d.watch.stream.length) {
        w.innerHTML = '<div style="color:#b8b8cc;font-size:13px">Streaming in ' + d.country + ':</div>' +
          d.watch.stream.map(s => '<span class="chip">' + s + '</span>').join('');
      } else if (d.watch && (d.watch.rent||[]).concat(d.watch.buy||[]).length) {
        w.innerHTML = '<div style="color:#b8b8cc;font-size:13px">Rent/buy in ' + d.country + ':</div>' +
          (d.watch.rent||[]).concat(d.watch.buy||[]).map(s => '<span class="chip">' + s + '</span>').join('');
      } else if (d.watch) {
        w.innerHTML = '<div style="color:#b8b8cc;font-size:13px">Not available to stream in ' + d.country + '.</div>';
      } else if (d.identified) {
        w.innerHTML = '<div style="color:#6a6a80;font-size:13px">(add a free TMDB key to see where to stream)</div>';
      }
      document.getElementById('time').textContent = d.elapsed_seconds ? ('answered in ' + d.elapsed_seconds + 's') : '';
      notif.style.display = 'block';
    } catch (err) {
      spin.style.display = 'none';
      document.getElementById('title').textContent = '⚠️ Could not reach the server';
      document.getElementById('conf').textContent = ''; document.getElementById('detail').textContent = String(err);
      document.getElementById('watch').innerHTML = ''; document.getElementById('time').textContent = '';
      notif.style.display = 'block';
    }
    go.disabled = false;
  });
</script>
</body>
</html>"""


@app.get("/demo", response_class=HTMLResponse)
async def demo():
    return DEMO_HTML


# --- Mobile home-screen app (Add to Home Screen -> real app icon) -------------
APP_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, maximum-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="No Clu">
<meta name="theme-color" content="#000000">
<link rel="apple-touch-icon" href="/app-icon.png">
<link rel="icon" href="/app-icon.png">
<title>No Clú</title>
<style>
  :root{
    --bg:#000; --card:#0D0B08; --ink:#F4F1EA; --muted:#9a8f7d; --faint:#6a6152;
    --amber:#E0A55A; --amber-bright:#F2C784; --err:#e77;
    --mono: ui-monospace,"SF Mono",Menlo,Consolas,monospace;
    --sans: -apple-system,BlinkMacSystemFont,"SF Pro Text",system-ui,sans-serif;
  }
  *{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
  html,body{height:100%}
  body{background:var(--bg);color:var(--ink);font-family:var(--sans);
       min-height:100dvh;display:flex;flex-direction:column;
       padding:env(safe-area-inset-top) 0 env(safe-area-inset-bottom);-webkit-font-smoothing:antialiased}
  @keyframes spinCW{from{transform:translate(-50%,-50%) rotate(45deg)}to{transform:translate(-50%,-50%) rotate(405deg)}}
  @keyframes spinCCW{from{transform:translate(-50%,-50%) rotate(45deg)}to{transform:translate(-50%,-50%) rotate(-315deg)}}
  @keyframes pulseRing{0%{transform:translate(-50%,-50%) rotate(45deg) scale(.5);opacity:.8}100%{transform:translate(-50%,-50%) rotate(45deg) scale(2.6);opacity:0}}
  @keyframes coreGlow{0%,100%{box-shadow:0 0 22px 4px rgba(224,165,90,.45)}50%{box-shadow:0 0 34px 8px rgba(224,165,90,.7)}}
  @keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
  .brand{font-family:var(--mono);font-weight:700;letter-spacing:3px;color:var(--amber);text-shadow:0 0 14px rgba(224,165,90,.55)}

  /* ---- sign-in gate ---- */
  #gate{flex:1;display:none;flex-direction:column;align-items:center;justify-content:center;padding:30px 24px}
  #gate.show{display:flex}
  #gate .brand{font-size:34px;margin-bottom:6px}
  #gate .tag{color:var(--muted);font-size:14px;margin-bottom:30px;text-align:center}
  .authcard{width:100%;max-width:360px;background:var(--card);border:1px solid rgba(224,165,90,.2);border-radius:18px;padding:22px}
  .seg{display:flex;background:rgba(255,255,255,.04);border-radius:12px;padding:4px;margin-bottom:18px}
  .seg button{flex:1;font-family:var(--mono);font-size:12px;letter-spacing:1px;text-transform:uppercase;
              padding:9px;border:none;border-radius:9px;background:transparent;color:var(--muted);cursor:pointer}
  .seg button.on{background:var(--amber);color:#1a0f00;font-weight:700}
  .field{margin-bottom:12px}
  .field input{width:100%;background:rgba(255,255,255,.05);border:1px solid rgba(224,165,90,.2);border-radius:11px;
               padding:14px;color:var(--ink);font-size:16px}
  .field input:focus{outline:none;border-color:var(--amber)}
  .err{color:var(--err);font-size:13px;min-height:16px;margin:2px 0 10px}
  .primary{width:100%;height:50px;border:none;border-radius:12px;background:var(--amber);color:#1a0f00;
           font-family:var(--mono);font-weight:700;font-size:14px;letter-spacing:1px;text-transform:uppercase;cursor:pointer}
  .primary:disabled{opacity:.6}
  .soon{margin-top:16px;text-align:center;font-size:12px;color:var(--faint)}

  /* ---- main app ---- */
  #app{flex:1;display:none;flex-direction:column}
  #app.show{display:flex}
  .top{display:flex;align-items:center;justify-content:space-between;padding:16px 22px 4px}
  .top .brand{font-size:24px}
  .who{display:flex;align-items:center;gap:10px}
  .who .name{font-family:var(--mono);font-size:11px;color:var(--muted);max-width:130px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .who .out{font-family:var(--mono);font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--amber);
            border:1px solid rgba(224,165,90,.3);border-radius:20px;padding:6px 10px;background:none;cursor:pointer}
  main{flex:1;display:flex;flex-direction:column;align-items:center;padding:10px 22px 30px;overflow-y:auto}
  .status{font-family:var(--mono);font-size:11px;letter-spacing:2.5px;text-transform:uppercase;color:var(--muted);
          margin:14px 0 26px;transition:color .3s;min-height:14px}
  .status.on{color:var(--amber-bright)}
  .stage{position:relative;width:224px;height:224px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
  .ring{position:absolute;top:50%;left:50%;border:1.4px dashed var(--amber);border-radius:26px;opacity:.4}
  .ring.a{width:204px;height:204px;animation:spinCW 14s linear infinite}
  .ring.b{width:168px;height:168px;border-width:1px;opacity:.28;animation:spinCCW 18s linear infinite}
  .stage.scan .ring.a{animation-duration:3.4s}.stage.scan .ring.b{animation-duration:4.4s}
  .pulse{position:absolute;top:50%;left:50%;width:150px;height:150px;border-radius:30px;border:1.5px solid var(--amber);opacity:0}
  .pulse.fire{animation:pulseRing 1s ease-out}
  .lens{position:relative;width:150px;height:150px;transform:rotate(45deg);border-radius:30px;border:1px solid rgba(224,165,90,.3);
        background:rgba(244,241,234,.03);display:flex;align-items:center;justify-content:center;cursor:pointer}
  .pupil{width:54px;height:54px;border-radius:16px;transform:rotate(45deg);
         background:radial-gradient(circle at 36% 32%,var(--amber-bright),var(--amber) 70%);animation:coreGlow 2.4s ease-in-out infinite}
  .stage.scan .pupil{animation:none;background:radial-gradient(circle at 36% 32%,#fff,var(--amber-bright) 70%)}
  .hint{margin-top:24px;color:var(--faint);font-size:13.5px;text-align:center;max-width:280px;line-height:1.5}
  #file{display:none}
  .card{width:100%;margin-top:26px;background:var(--card);border:1px solid rgba(224,165,90,.22);border-radius:18px;overflow:hidden;
        animation:fadeUp .5s ease-out;display:none}
  .card.show{display:block}
  .card img.poster{width:100%;display:none;border-bottom:1px solid rgba(224,165,90,.15)}
  .card .body{padding:18px}
  .card .ttl{font-size:21px;font-weight:700;line-height:1.2}
  .card .meta{font-family:var(--mono);font-size:11px;letter-spacing:1px;color:var(--muted);margin-top:6px;text-transform:uppercase}
  .card .det{color:#cbb89a;font-size:14px;margin-top:12px;line-height:1.5}
  .watchlabel{font-family:var(--mono);font-size:10.5px;letter-spacing:1px;color:var(--faint);text-transform:uppercase;margin-top:16px}
  .chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:9px}
  .chips a{font-family:var(--mono);font-size:11px;letter-spacing:.5px;text-decoration:none;
           padding:8px 12px;border-radius:20px;border:1px solid rgba(224,165,90,.35);color:var(--amber-bright);
           background:rgba(224,165,90,.07)}
  .cta{display:flex;align-items:center;justify-content:center;gap:8px;margin-top:14px;text-decoration:none;
       height:50px;border-radius:12px;background:var(--amber);color:#1a0f00;font-family:var(--mono);font-weight:700;
       font-size:13px;letter-spacing:1.5px;text-transform:uppercase}
  .connect{width:100%;margin-top:34px;background:var(--card);border:1px solid rgba(224,165,90,.2);border-radius:16px;padding:18px}
  .connect h2{font-family:var(--mono);font-size:12px;letter-spacing:1px;color:var(--amber-bright);text-transform:uppercase;margin-bottom:8px}
  .connect p{font-size:13px;color:var(--muted);line-height:1.5;margin-bottom:12px}
  .connect p b{color:var(--ink)}
  .urlbox{background:rgba(0,0,0,.4);border:1px solid rgba(224,165,90,.18);border-radius:10px;padding:11px 12px;
          font-family:var(--mono);font-size:11px;color:var(--amber-bright);word-break:break-all;line-height:1.5}
  .copybtn{margin-top:10px;width:100%;height:44px;border:none;border-radius:11px;background:var(--amber);color:#1a0f00;
           font-family:var(--mono);font-weight:700;font-size:12px;letter-spacing:1px;text-transform:uppercase;cursor:pointer}
  .setupbtn{display:flex;align-items:center;justify-content:center;gap:9px;width:100%;margin-top:34px;
            height:52px;border-radius:14px;text-decoration:none;background:rgba(224,165,90,.1);
            border:1px solid rgba(224,165,90,.4);color:var(--amber-bright);
            font-family:var(--mono);font-weight:700;font-size:12px;letter-spacing:1.5px;text-transform:uppercase}
  .recent{width:100%;margin-top:30px}
  .recent h2{font-family:var(--mono);font-size:10.5px;letter-spacing:2px;color:var(--faint);text-transform:uppercase;margin-bottom:10px}
  .ritem{display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid rgba(150,140,120,.12)}
  .ritem .rth{width:34px;height:48px;border-radius:5px;object-fit:cover;background:rgba(224,165,90,.1);flex-shrink:0}
  .ritem .rmain{flex:1;min-width:0}
  .ritem .rt{font-size:14px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .ritem .rm{font-family:var(--mono);font-size:10px;color:var(--muted)}
</style>
</head>
<body>
  <!-- Sign-in gate -->
  <div id="gate">
    <div class="brand">No Clú</div>
    <div class="tag">Sign in so your scans follow you to every device.</div>
    <div class="authcard">
      <div class="seg">
        <button id="segLogin" class="on" onclick="setMode('login')">Sign in</button>
        <button id="segReg" onclick="setMode('register')">Create account</button>
      </div>
      <div class="field"><input id="identifier" type="text" autocapitalize="none" autocorrect="off"
           placeholder="Email or phone number" inputmode="email"></div>
      <div class="field"><input id="password" type="password" placeholder="Password (8+ characters)"></div>
      <div class="err" id="authErr"></div>
      <button class="primary" id="authBtn" onclick="submitAuth()">Sign in</button>
      <div class="soon">Sign in with Google — coming once the app is hosted.</div>
    </div>
  </div>

  <!-- Main app -->
  <div id="app">
    <div class="top">
      <div class="brand">No Clú</div>
      <div class="who"><span class="name" id="whoName"></span><button class="out" onclick="logout()">Sign out</button></div>
    </div>
    <main>
      <div class="status" id="status">Tap the lens</div>
      <div class="stage" id="stage">
        <div class="ring a"></div><div class="ring b"></div>
        <div class="pulse" id="pulse"></div>
        <div class="lens" id="lens"><div class="pupil"></div></div>
      </div>
      <div class="hint">Tap the lens, then take a photo or pick a screenshot of what you're watching.</div>
      <input type="file" id="file" accept="image/*">
      <div class="card" id="card">
        <img class="poster" id="poster" alt="">
        <div class="body">
          <div class="ttl" id="ttl"></div>
          <div class="meta" id="meta"></div>
          <div class="det" id="det"></div>
          <div class="watchlabel" id="watchLabel" style="display:none"></div>
          <div class="chips" id="chips" style="display:none"></div>
          <a class="cta" id="cta" target="_blank" rel="noopener" style="display:none">▶ <span id="ctaLabel"></span></a>
        </div>
      </div>
      <div class="recent" id="recentWrap" style="display:none">
        <h2>Your scan history</h2>
        <div id="recent"></div>
      </div>

      <a class="setupbtn" href="/shortcut">⚡ Set up one-tap scanning</a>
    </main>
  </div>
<script>
  function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
  var mode='login';
  function setMode(m){
    mode=m;
    document.getElementById('segLogin').classList.toggle('on', m==='login');
    document.getElementById('segReg').classList.toggle('on', m==='register');
    document.getElementById('authBtn').textContent = m==='login'?'Sign in':'Create account';
    document.getElementById('authErr').textContent='';
  }
  async function submitAuth(){
    var id=document.getElementById('identifier').value.trim();
    var pw=document.getElementById('password').value;
    var err=document.getElementById('authErr'), btn=document.getElementById('authBtn');
    err.textContent=''; btn.disabled=true;
    var fd=new FormData(); fd.append('identifier', id); fd.append('password', pw);
    try{
      var r=await fetch('/auth/'+(mode==='login'?'login':'register'),{method:'POST',body:fd});
      var d=await r.json();
      if(d.ok){ await checkAuth(); }
      else { err.textContent=d.error||'Something went wrong.'; }
    }catch(e){ err.textContent='Could not reach the server.'; }
    btn.disabled=false;
  }
  async function logout(){
    try{ await fetch('/auth/logout',{method:'POST'}); }catch(e){}
    document.getElementById('password').value='';
    showGate();
  }
  function showGate(){
    document.getElementById('gate').classList.add('show');
    document.getElementById('app').classList.remove('show');
  }
  function showApp(name){
    document.getElementById('whoName').textContent=name||'';
    document.getElementById('gate').classList.remove('show');
    document.getElementById('app').classList.add('show');
    loadHistory();
  }
  async function checkAuth(){
    try{
      var r=await fetch('/auth/me'); var d=await r.json();
      if(d.signed_in){ showApp(d.name); } else { showGate(); }
    }catch(e){ showGate(); }
  }

  async function loadHistory(){
    try{
      var r=await fetch('/history'); if(r.status!==200) return;
      var d=await r.json(); var list=d.scans||[];
      if(!list.length){ document.getElementById('recentWrap').style.display='none'; return; }
      document.getElementById('recentWrap').style.display='block';
      document.getElementById('recent').innerHTML=list.map(function(s){
        var thumb=s.poster?('<img class="rth" src="'+esc(s.poster)+'">'):'<div class="rth"></div>';
        var m=[(s.type||'').replace('_',' ')]; if(s.year) m.push(s.year);
        return '<div class="ritem">'+thumb+'<div class="rmain"><div class="rt">'+esc(s.title)+
               '</div><div class="rm">'+esc(m.filter(Boolean).join(' · '))+'</div></div></div>';
      }).join('');
    }catch(e){}
  }

  // ---- scanning ----
  var lens=document.getElementById('lens'), file=document.getElementById('file'),
      stage=document.getElementById('stage'), statusEl=document.getElementById('status'),
      pulse=document.getElementById('pulse'), card=document.getElementById('card');
  var busy=false;
  lens.addEventListener('click', function(){ if(!busy) file.click(); });
  file.addEventListener('change', function(e){ if(e.target.files[0]) scan(e.target.files[0]); });
  function firePulse(){ pulse.classList.remove('fire'); void pulse.offsetWidth; pulse.classList.add('fire'); }

  async function scan(f){
    busy=true; card.classList.remove('show');
    stage.classList.add('scan'); statusEl.classList.add('on'); statusEl.textContent='Analyzing…';
    firePulse();
    var fd=new FormData(); fd.append('image', f);
    try{
      var r=await fetch('/identify?country=IN',{method:'POST',body:fd});
      if(r.status===401){ showGate(); stage.classList.remove('scan'); busy=false; return; }
      var d=await r.json(); render(d); loadHistory();
    }catch(err){
      statusEl.textContent='Could not reach server';
      document.getElementById('ttl').textContent='⚠️ Connection failed';
      document.getElementById('meta').textContent=''; document.getElementById('det').textContent=String(err);
      document.getElementById('cta').style.display='none';
      document.getElementById('poster').style.display='none'; card.classList.add('show');
    }
    stage.classList.remove('scan'); busy=false;
  }
  function render(d){
    if(!d.identified){
      statusEl.classList.remove('on'); statusEl.textContent='Tap the lens';
      document.getElementById('ttl').textContent=d.summary||'No result';
      document.getElementById('meta').textContent=''; document.getElementById('det').textContent='';
      document.getElementById('cta').style.display='none';
      document.getElementById('poster').style.display='none'; card.classList.add('show'); return;
    }
    statusEl.textContent='Match locked';
    var poster=document.getElementById('poster');
    if(d.poster){ poster.src=d.poster; poster.style.display='block'; } else { poster.style.display='none'; }
    document.getElementById('ttl').textContent=d.title||'Unknown';
    var meta=[];
    if(d.type) meta.push(String(d.type).replace('_',' '));
    if(d.year) meta.push(d.year);
    if(d.season&&d.episode) meta.push('S'+d.season+' · E'+d.episode);
    document.getElementById('meta').textContent=meta.join('  ·  ');
    document.getElementById('det').textContent=d.detail||'';

    // Where to watch, in the user's region, with a tappable link per platform.
    var label=document.getElementById('watchLabel'), chips=document.getElementById('chips'), cta=document.getElementById('cta');
    var cc=d.country||'IN';
    var provs=d.providers||[];
    if(provs.length){
      label.textContent='Streaming in '+cc; label.style.display='block';
      chips.innerHTML=provs.map(function(p){return '<a href="'+esc(p.url)+'" target="_blank" rel="noopener">'+esc(p.name)+'</a>';}).join('');
      chips.style.display='flex';
    } else { label.style.display='none'; chips.style.display='none'; chips.innerHTML=''; }

    if(d.justwatch){
      cta.href=d.justwatch;
      document.getElementById('ctaLabel').textContent=(provs.length?'See all options in '+cc:'Where to watch in '+cc);
      cta.style.display='flex';
    } else { cta.style.display='none'; }
    card.classList.add('show');
  }

  setMode('login');
  checkAuth();
</script>
</body>
</html>"""


SHORTCUT_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, maximum-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="No Clu">
<link rel="apple-touch-icon" href="/app-icon.png">
<link rel="icon" href="/app-icon.png">
<title>Set up one-tap scanning · No Clú</title>
<style>
  :root{
    --bg:#000; --card:#0D0B08; --ink:#F4F1EA; --muted:#9a8f7d; --faint:#6a6152;
    --amber:#E0A55A; --amber-bright:#F2C784;
    --mono: ui-monospace,"SF Mono",Menlo,Consolas,monospace;
    --sans: -apple-system,BlinkMacSystemFont,"SF Pro Text",system-ui,sans-serif;
  }
  *{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
  body{background:var(--bg);color:var(--ink);font-family:var(--sans);min-height:100dvh;
       padding:calc(env(safe-area-inset-top) + 18px) 22px calc(env(safe-area-inset-bottom) + 40px);
       -webkit-font-smoothing:antialiased;max-width:520px;margin:0 auto}
  a.back{font-family:var(--mono);font-size:12px;letter-spacing:1px;color:var(--muted);text-decoration:none}
  .brand{font-family:var(--mono);font-weight:700;letter-spacing:3px;color:var(--amber);
         text-shadow:0 0 14px rgba(224,165,90,.5);font-size:15px;margin:18px 0 4px}
  h1{font-size:25px;line-height:1.25;margin-top:8px}
  .sub{color:var(--muted);font-size:14px;line-height:1.55;margin-top:10px}
  .step{background:var(--card);border:1px solid rgba(224,165,90,.2);border-radius:16px;padding:18px;margin-top:18px}
  .step .n{font-family:var(--mono);font-size:11px;letter-spacing:1px;color:var(--amber-bright);text-transform:uppercase}
  .step h2{font-size:17px;margin-top:6px}
  .step p{color:var(--muted);font-size:13.5px;line-height:1.55;margin-top:8px}
  .step p b{color:var(--ink)}
  .urlbox{background:rgba(0,0,0,.45);border:1px solid rgba(224,165,90,.18);border-radius:10px;padding:11px 12px;
          font-family:var(--mono);font-size:11px;color:var(--amber-bright);word-break:break-all;line-height:1.5;margin-top:12px}
  .btn{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;height:50px;margin-top:12px;
       border:none;border-radius:12px;background:var(--amber);color:#1a0f00;cursor:pointer;text-decoration:none;
       font-family:var(--mono);font-weight:700;font-size:12.5px;letter-spacing:1px;text-transform:uppercase}
  .btn.ghost{background:rgba(224,165,90,.1);border:1px solid rgba(224,165,90,.4);color:var(--amber-bright)}
  ol.manual{margin:12px 0 0 18px;color:var(--muted);font-size:13.5px;line-height:1.7}
  ol.manual b{color:var(--ink)}
  code{font-family:var(--mono);font-size:12px;background:rgba(224,165,90,.12);color:var(--amber-bright);
       padding:2px 6px;border-radius:5px;word-break:break-all}
  #need,#setup{display:none}
  #need.show,#setup.show{display:block}
  #need{text-align:center;padding-top:60px}
</style>
</head>
<body>
  <a class="back" href="/app">← Back to app</a>

  <div id="need">
    <div class="brand">No Clú</div>
    <h1>Please sign in first</h1>
    <p class="sub">Open the app, sign in, then come back here to set up one-tap scanning.</p>
    <a class="btn" href="/app" style="max-width:260px;margin:22px auto 0">Go to sign in</a>
  </div>

  <div id="setup">
    <div class="brand">No Clú</div>
    <h1>One-tap scanning</h1>
    <p class="sub">Add a Shortcut to your iPhone so a single tap identifies whatever you're watching — and saves it to your history here. Two quick steps.</p>

    <div class="step">
      <div class="n">Step 1</div>
      <h2>Copy your personal link</h2>
      <p>This link is unique to <b id="who">you</b> — it's what ties scans to your account.</p>
      <div class="urlbox"><span id="link">…</span></div>
      <button class="btn" id="copyBtn" onclick="copyLink()">Copy my link</button>
    </div>

    <div class="step" id="addStep">
      <div class="n">Step 2</div>
      <h2 id="addTitle">Add the ready-made Shortcut</h2>
      <div id="oneTap">
        <p>Tap the button below. iPhone will open Shortcuts and ask you to <b>paste your personal link</b> — press and hold, then <b>Paste</b> (it's already copied from Step 1). Tap <b>Add Shortcut</b>. Done!</p>
        <a class="btn" id="addBtn" href="__ICLOUD_URL__">＋ Add the No Clú Shortcut</a>
        <p style="margin-top:14px">Then bind it to your <b>Action Button</b> or <b>Back Tap</b> (Settings → Accessibility → Touch → Back Tap) so one tap scans instantly.</p>
      </div>
      <div id="manual" style="display:none">
        <p>Build a tiny Shortcut once (takes a minute):</p>
        <ol class="manual">
          <li>Open the <b>Shortcuts</b> app → <b>+</b> to create a new one.</li>
          <li>Add action <b>Take Screenshot</b>.</li>
          <li>Add action <b>Get Contents of URL</b>. Set URL to your copied link, tap <b>Show More</b>: Method <b>POST</b>, Request Body <b>File</b>, and choose the <b>Screenshot</b> variable.</li>
          <li>Add action <b>Show Result</b> (or Quick Look).</li>
          <li>Name it <b>No Clú</b>, then bind it to your <b>Action Button</b> or <b>Back Tap</b>.</li>
        </ol>
      </div>
    </div>
  </div>

<script>
  var ICLOUD = "__ICLOUD_URL__";
  var myLink = "";
  async function init(){
    try{
      var r = await fetch('/auth/me'); var d = await r.json();
      if(!d.signed_in){ document.getElementById('need').classList.add('show'); return; }
      myLink = d.shortcut_url || "";
      document.getElementById('who').textContent = d.name || "you";
      document.getElementById('link').textContent = myLink || "(sign in to get your link)";
      // If no ready-made Shortcut is configured, show manual build steps instead.
      if(!ICLOUD){
        document.getElementById('oneTap').style.display='none';
        document.getElementById('manual').style.display='block';
        document.getElementById('addTitle').textContent='Build the Shortcut';
      }
      document.getElementById('setup').classList.add('show');
    }catch(e){ document.getElementById('need').classList.add('show'); }
  }
  function copyLink(){
    var btn=document.getElementById('copyBtn');
    if(!myLink) return;
    function done(){ btn.textContent='Copied ✓ — now tap Step 2'; setTimeout(function(){ btn.textContent='Copy my link'; },2600); }
    if(navigator.clipboard && navigator.clipboard.writeText){
      navigator.clipboard.writeText(myLink).then(done, fallback);
    } else { fallback(); }
    function fallback(){
      var t=document.createElement('textarea'); t.value=myLink; document.body.appendChild(t);
      t.select(); try{ document.execCommand('copy'); }catch(e){} document.body.removeChild(t); done();
    }
  }
  init();
</script>
</body>
</html>"""


# --- Accounts (email/phone + password; Google added after hosting) -----------
def _identifier_kind(identifier: str):
    """Classify a login identifier as ('email'|'phone', normalized) or (None, None)."""
    identifier = (identifier or "").strip()
    if auth.valid_email(identifier):
        return "email", identifier.lower()
    if auth.valid_phone(identifier):
        return "phone", identifier
    return None, None


@app.post("/auth/register")
async def auth_register(request: Request, response: Response, identifier: str = Form(...), password: str = Form(...)):
    kind, value = _identifier_kind(identifier)
    if not kind:
        return JSONResponse({"ok": False, "error": "Enter a valid email or phone number."}, status_code=400)
    pw_problem = auth.password_problem(password)
    if pw_problem:
        return JSONResponse({"ok": False, "error": pw_problem}, status_code=400)

    session = db.SessionLocal()
    try:
        existing = (db.get_user_by_email(session, value) if kind == "email"
                    else db.get_user_by_phone(session, value))
        if existing:
            return JSONResponse({"ok": False, "error": "That account already exists — try signing in."}, status_code=409)
        kwargs = {"password_hash": auth.hash_password(password), kind: value}
        user = db.create_user(session, **kwargs)
        _set_session_cookie(request, response, user.id)
        return {"ok": True}
    finally:
        session.close()


@app.post("/auth/login")
async def auth_login(request: Request, response: Response, identifier: str = Form(...), password: str = Form(...)):
    kind, value = _identifier_kind(identifier)
    if not kind:
        return JSONResponse({"ok": False, "error": "Enter a valid email or phone number."}, status_code=400)
    session = db.SessionLocal()
    try:
        user = (db.get_user_by_email(session, value) if kind == "email"
                else db.get_user_by_phone(session, value))
        if not user or not auth.verify_password(password, user.password_hash or ""):
            return JSONResponse({"ok": False, "error": "Wrong email/phone or password."}, status_code=401)
        _set_session_cookie(request, response, user.id)
        return {"ok": True}
    finally:
        session.close()


@app.post("/auth/logout")
async def auth_logout(response: Response):
    response.delete_cookie(auth.SESSION_COOKIE)
    return {"ok": True}


@app.get("/auth/me")
async def auth_me(request: Request):
    uid = current_user_id(request)
    if uid is None:
        return {"signed_in": False}
    session = db.SessionLocal()
    try:
        user = db.get_user_by_id(session, uid)
        if not user:
            return {"signed_in": False}
        token = db.ensure_scan_token(session, user)
        # Personalized Shortcut endpoint — base_url auto-tracks local vs hosted.
        shortcut_url = f"{str(request.base_url).rstrip('/')}/scan?token={token}&country={DEFAULT_COUNTRY}"
        return {"signed_in": True,
                "name": user.display_name or user.email or user.phone,
                "shortcut_url": shortcut_url}
    finally:
        session.close()


@app.get("/history")
async def history(request: Request):
    uid = current_user_id(request)
    if uid is None:
        return JSONResponse({"error": "not signed in"}, status_code=401)
    session = db.SessionLocal()
    try:
        scans = db.recent_scans(session, uid)
        return {"scans": [
            {"title": s.title, "type": s.content_type, "year": s.year,
             "poster": s.poster, "detail": s.detail,
             "at": s.scanned_at.isoformat() if s.scanned_at else None}
            for s in scans
        ]}
    finally:
        session.close()


@app.get("/app", response_class=HTMLResponse)
async def app_page():
    return APP_HTML


@app.get("/shortcut", response_class=HTMLResponse)
async def shortcut_page():
    """Setup sub-page: gives the signed-in user their personal scan link and,
    when ICLOUD_SHORTCUT_URL is configured, a one-tap install of the template."""
    return SHORTCUT_HTML.replace("__ICLOUD_URL__", ICLOUD_SHORTCUT_URL)


@app.get("/app-icon.png")
async def app_icon():
    """Amber-diamond home-screen icon, generated once so no asset file is needed."""
    size = 180
    icon = Image.new("RGB", (size, size), (0, 0, 0))
    d = ImageDraw.Draw(icon)
    cx = cy = size // 2
    d.polygon([(cx, 34), (size - 34, cy), (cx, size - 34), (34, cy)],
              outline=(224, 165, 90), width=4)
    d.polygon([(cx, 74), (size - 74, cy), (cx, size - 74), (74, cy)],
              fill=(224, 165, 90))
    buf = io.BytesIO()
    icon.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/")
async def health():
    key_present = bool(GEMINI_API_KEY) if PROVIDER == "gemini" else bool(os.getenv("ANTHROPIC_API_KEY"))
    return {"app": "No Clú", "provider": PROVIDER, "key_configured": key_present,
            "tmdb": bool(TMDB_API_KEY), "default_country": DEFAULT_COUNTRY}


@app.post("/identify")
async def identify(request: Request, image: UploadFile = File(...), country: Optional[str] = None):
    started = time.monotonic()
    if daily_cap.exceeded:
        return {"identified": False,
                "summary": "🌙 No Clú's free daily limit is used up — try again after midnight!",
                "elapsed_seconds": round(time.monotonic() - started, 2)}

    resolved_country = await resolve_country(request, country)
    try:
        b64 = standard_b64encode(shrink(await image.read())).decode()
        content = await identify_content(b64)
    except ProviderError as e:
        return {"identified": False, "summary": str(e),
                "elapsed_seconds": round(time.monotonic() - started, 2)}
    except Exception:
        return {"identified": False, "summary": "🔍 Couldn't read that screen — try again.",
                "elapsed_seconds": round(time.monotonic() - started, 2)}
    daily_cap.record()  # only count successful recognitions against the free quota

    watch = await tmdb_where_to_watch(content, resolved_country)
    poster = await fetch_poster(content)

    # "Where to watch in <country>" — build tappable per-platform links.
    # `providers` is the exact list (from TMDB, when a key is configured);
    # `justwatch` always works (no key) and lists every platform in-region.
    providers = []
    if watch and watch.get("stream"):
        providers = [{"name": name, "url": provider_link(name, content.title, resolved_country)}
                     for name in watch["stream"]]
    jw = justwatch_url(content, resolved_country)

    # Save to the signed-in user's history so it syncs across their devices.
    # Never let a history-write failure break the recognition result.
    uid = current_user_id(request)
    if uid is not None:
        try:
            session = db.SessionLocal()
            try:
                db.add_scan(session, uid, title=content.title, content_type=content.content_type,
                            year=content.year, poster=poster, detail=content.detail)
            finally:
                session.close()
        except Exception:
            pass

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
        "providers": providers,
        "justwatch": jw,
        "poster": poster,
        "summary": build_summary(content, watch, resolved_country),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


async def _read_uploaded_image(request: Request) -> Optional[bytes]:
    """Get the screenshot bytes however the iOS Shortcut sent them.

    Accepts all three ways a Shortcut's "Get Contents of URL" can post:
    - Request Body = File  -> raw image bytes as the whole body
    - Request Body = Form  -> a file field (named 'image', or any field)
    So the user doesn't have to configure the form field exactly right.
    """
    ctype = request.headers.get("content-type", "")
    if "multipart/form-data" in ctype:
        form = await request.form()
        upload = form.get("image")
        if upload is None:  # fall back to the first file-like field of any name
            for value in form.values():
                if hasattr(value, "read"):
                    upload = value
                    break
        if upload is not None and hasattr(upload, "read"):
            return await upload.read()
        return None
    body = await request.body()
    return body or None


@app.post("/scan", response_class=PlainTextResponse)
async def scan_text(request: Request, country: Optional[str] = None, token: Optional[str] = None):
    """Plain-text sibling of /identify for the iOS Shortcut.

    Returns a ready-to-show string (title + where-to-watch link) so the
    Shortcut needs no JSON parsing — just "Take Screenshot -> Get Contents of
    URL -> Show Notification". If a personal `token` is present, the scan is
    also saved to that account's history (that's how a back-tap scan lands in
    the app, since a Shortcut carries no login session).
    """
    if daily_cap.exceeded:
        return "🌙 No Clú's free daily limit is used up — try again after midnight!"

    raw = await _read_uploaded_image(request)
    if not raw:
        return ("🔍 No screenshot received. In the Shortcut's 'Get Contents of URL', "
                "set Request Body to File and choose the Screenshot.")

    resolved_country = await resolve_country(request, country)
    try:
        b64 = standard_b64encode(shrink(raw)).decode()
        content = await identify_content(b64)
    except ProviderError as e:
        return str(e)
    except Exception:
        return "🔍 Couldn't read that screen — try again."
    daily_cap.record()  # only count successful recognitions against the free quota

    watch = await tmdb_where_to_watch(content, resolved_country)

    # Save to the account that owns this token, so back-tap scans sync to the app.
    # Best-effort: a history-write failure must never break the notification.
    if token:
        try:
            session = db.SessionLocal()
            try:
                user = db.get_user_by_scan_token(session, token)
                if user:
                    poster = await fetch_poster(content)
                    db.add_scan(session, user.id, title=content.title,
                                content_type=content.content_type, year=content.year,
                                poster=poster, detail=content.detail)
            finally:
                session.close()
        except Exception:
            pass

    lines = [build_summary(content, watch, resolved_country)]
    jw = justwatch_url(content, resolved_country)
    if jw:
        lines.append(f"▶ Where to watch in {resolved_country}: {jw}")
    return "\n".join(lines)
