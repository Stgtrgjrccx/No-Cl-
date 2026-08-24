"""No Clú — one-tap "Shazam for your screen".

POST /identify with a screenshot -> a vision AI identifies what's playing,
TMDB tells you where to stream it in your country.

The vision "brain" is pluggable via the PROVIDER setting in .env:
  PROVIDER=gemini     -> Google Gemini free tier (no credit card)  [default]
  PROVIDER=anthropic  -> Anthropic Claude (paid, sharper on obscure content)
Switching later is a one-word change plus the matching API key.
"""

import asyncio
import io
import ipaddress
import json
import os
import secrets
import time
import unicodedata
from base64 import standard_b64encode
from datetime import datetime
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple
from urllib.parse import quote_plus, urlencode

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import (HTMLResponse, JSONResponse, PlainTextResponse,
                               RedirectResponse)
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
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w780"
DEFAULT_COUNTRY = os.getenv("DEFAULT_COUNTRY", "US")
# iCloud share link of the ready-made "No Clú" Shortcut template. Set this in the
# host's env once the template is shared; the /shortcut page then offers one-tap
# install. Left blank => the page falls back to short manual build instructions.
ICLOUD_SHORTCUT_URL = os.getenv("ICLOUD_SHORTCUT_URL", "").strip()
# Social sign-in appears only once its credentials exist, so the sign-in screen
# never shows a button that cannot work. Email + password always works.
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
APPLE_CLIENT_ID = os.getenv("APPLE_CLIENT_ID", "").strip()
GOOGLE_STATE_COOKIE = "noclu_oauth_state"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
# Trailing label on the Shortcut's reply; its URL is what the Shortcut opens.
SCAN_APP_LINK_LABEL = "📱 Open in No Clú"

# --- Provider settings -------------------------------------------------------
# Gemini: gemini-flash-latest is fast, multimodal, and free-tier friendly (some
# accounts have zero free quota on the pinned gemini-2.0-flash, so "latest" is a
# safer default). Get a key (no card) at https://aistudio.google.com/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Tried in order, FASTEST first — see identify_gemini for why. Measured on a
# 1080p frame: flash-lite ~1.4s, flash ~3.5s, same answer on ordinary content.
# Later entries are stronger, so a low-confidence answer escalates along the
# chain; a model whose daily free quota is used up is skipped rather than
# failing the scan, so results degrade instead of stopping.
GEMINI_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-flash-lite-latest,gemini-flash-latest"
).split(",") if m.strip()]
# Anthropic: claude-opus-4-8 is the most accurate; claude-haiku-4-5 the fastest.
ANTHROPIC_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")

# --- Free vision providers beyond Gemini -------------------------------------
# All three speak the OpenAI chat-completions shape, so one adapter serves them
# all. Every one is optional: leave a key unset and that provider is simply
# skipped. They exist for two jobs Gemini alone does badly:
#
#   ESCALATE  — a frontier model for scans the fast model is unsure about.
#               "Which Saif Ali Khan film is this?" is world knowledge, which is
#               exactly where a bigger model beats a faster one. GitHub Models
#               gives GPT-4o vision at ~50 requests/day free, and escalation
#               only fires on uncertain scans, so that budget is ample.
#   BACKSTOP  — somewhere to go when Gemini's daily quota is gone, so scans
#               degrade instead of failing.
#
# Deliberately NOT solved by minting extra Gemini keys: keys inside one Google
# project share a single quota pool, so it would not work, and creating extra
# projects to dodge the limit breaks Google's terms — risking the account the
# whole app depends on.
# Each takes a COMMA-SEPARATED list, tried in order. Model ids on these
# services churn constantly — the first Groq id here was a guess that returned
# 404 and silently disabled the whole provider for hours. A list means a
# renamed or retired model costs one wasted call, not a dead fallback.
GITHUB_MODELS_TOKEN = os.getenv("GITHUB_MODELS_TOKEN", "").strip()
GITHUB_MODELS_MODEL = os.getenv("GITHUB_MODELS_MODEL", "gpt-4o,gpt-4o-mini")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
# Groq's vision line-up churns fast: both Llama 4 vision ids I tried returned
# 404 (scout was deprecated in June, maverick renamed). Per Groq's own vision
# docs the current multimodal model is qwen3.6-27b; the Llama ids stay behind it
# in case an account still has them.
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "qwen/qwen3.6-27b,"
    "meta-llama/llama-4-scout-17b-16e-instruct")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "pixtral-12b-2409,pixtral-large-latest")


def _models(spec: str) -> List[str]:
    return [m.strip() for m in spec.split(",") if m.strip()]


def _extra_providers() -> List[Dict[str, str]]:
    """Configured OpenAI-compatible vision providers, best-answer first.

    Order is by expected accuracy, not speed — these only run when Gemini was
    unsure or unavailable, so being slower than Gemini is the accepted cost.
    """
    # One entry per (service, model): a 404 on a renamed model then falls
    # through to the next id instead of disabling the service.
    services = [
        # A frontier model: the escalation target, and the only extra trusted
        # to overrule Gemini when both are equally sure.
        (GITHUB_MODELS_TOKEN, "github-models", "https://models.inference.ai.azure.com",
         GITHUB_MODELS_MODEL, STRENGTH_FRONTIER),
        # Backstops, not authorities. Large free quotas, weaker judgement —
        # they answer when nothing better could, and never win a tie.
        (GROQ_API_KEY, "groq", "https://api.groq.com/openai/v1",
         GROQ_MODEL, STRENGTH_BACKSTOP),
        (MISTRAL_API_KEY, "mistral", "https://api.mistral.ai/v1",
         MISTRAL_MODEL, STRENGTH_BACKSTOP),
    ]
    configured = []
    for key, name, base, spec, strength in services:
        if not key:
            continue
        for model in _models(spec):
            configured.append({
                "label": f"{name}/{model}", "service": name, "base": base,
                "key": key, "model": model, "strength": strength,
            })
    return configured

# Vision models read a ~1.15MP image at full fidelity; anything bigger just adds
# upload + processing latency, so downscale before sending.
MAX_IMAGE_EDGE = 1920
JPEG_QUALITY = 88
# Row brightness spread (0-255 scale) below which a row is flat
# interface rather than video.
MIN_ROW_VARIANCE = 6.0
MAX_FRAMES = 5
MAX_AUDIO_BYTES = 2 * 1024 * 1024
AUDIO_MIME_TYPES = frozenset({"audio/mp4", "audio/m4a", "audio/x-m4a",
                              "audio/mpeg", "audio/wav", "audio/x-wav"})

IDENTIFY_PROMPT = """\
This is a single frame captured from someone's phone or TV screen. Identify what \
content is being watched. Look at everything: the video frame itself, actors' faces, \
UI elements (Netflix/YouTube/Prime player chrome, titles, captions, channel names, \
progress bars, watermarks, scoreboards).

Rules:
- You may be given SEVERAL frames captured a second apart from the same video, \
and sometimes a short audio clip. Treat them as one piece of content, not several.

- LOOK BEFORE YOU READ. Study the picture first and form a view from it alone: \
who is on screen (lead and supporting cast), the setting and location, the \
period and costuming, the cinematography and colour grade, the language implied \
by lip movement and signage. Only after that, read any text. Use the text to \
CONFIRM or CORRECT what you saw — never let text alone decide while you ignore \
the picture.

- Text is not all equal. Text belonging to the PLAYER or the content — a title \
card, Netflix/Prime/YouTube chrome, an episode label, official credits — is \
strong evidence. Text belonging to whoever POSTED the clip is not: compilation \
headers ("Best Bollywood Movies"), part numbers ("PART:173"), account handles \
("@movixo._"), hashtags and viewer comments describe the post, not the film. \
Never take those as the title.

- RECOGNISING AN ACTOR IS NOT IDENTIFYING THE FILM. Actors appear in many \
films. If you recognise a face but cannot confirm which specific title THIS \
scene comes from, you MUST set confidence to "low" and name the actor in detail \
instead of choosing a title from their filmography. Naming the wrong film is far \
worse than admitting you are unsure.

- Use "high" only when the title is written on screen, or the specific scene is \
unmistakable to you. Knowing WHO is in it is not knowing WHAT it is — that is \
"low". Use "medium" when several strong visual signals agree but no text confirms.

- If the picture and the audio disagree, say so in detail and lower confidence \
rather than inventing a compromise answer.
- Give the official title, not a description of the scene.
- For TV shows and anime, include season/episode if the UI shows it or you can tell \
from the scene; otherwise leave them null.
- For YouTube, the title is the video title and detail should name the channel if visible.
- For live sports, title is the matchup/event (e.g. "India vs Australia — 3rd ODI").
- If you genuinely cannot identify it, set content_type to "other", title to your best \
guess of what kind of content it is, and confidence to "low".
- detail: one short sentence of context (what scene / why you're confident / what gave it away).
- evidence: the ONE specific thing your answer rests on, stated plainly — \
"title card reads Jawan", "Netflix player shows S2:E5", or "Saif Ali Khan, \
specific film not identified".
- evidence_type: which KIND of evidence that was. Choose honestly:
    "title_text"  — the title itself is written on screen
    "player_ui"   — a player/app shows the title or episode
    "scene"       — no text, but you recognise THIS SPECIFIC SCENE or shot
    "person_only" — you recognise a face or voice but not which title it is from
    "none"        — neither
  Choosing "person_only" is not a failure; it is the correct, useful answer \
whenever you know WHO but not WHAT. Recognising an actor and then naming any \
film from their filmography is the single worst thing you can do here.

Respond with ONLY a JSON object (no markdown, no code fences) with exactly these keys:
  content_type: one of "movie","tv_show","anime","youtube_video","sports","music_video","game","other"
  title: string
  year: integer or null
  season: integer or null
  episode: integer or null
  confidence: one of "high","medium","low"
  detail: string
  evidence: string
  evidence_type: one of "title_text","player_ui","scene","person_only","none"
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
    # What the answer actually rests on. Naming the evidence out loud is what
    # makes the confidence auditable instead of decorative — a model that has
    # to write "Saif Ali Khan, specific film not identified" is much less
    # willing to also claim "high".
    evidence: str = ""
    evidence_type: Literal[
        "title_text", "player_ui", "scene", "person_only", "none"] = "none"
    # Which model actually produced this. Filled in by _best_of, never by the
    # model itself — without it there is no way to tell from the outside
    # whether escalation reached the frontier provider or quietly fell through
    # to the backstop, which is exactly the confusion that cost hours here.
    source: str = ""


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
        "evidence": {"type": "string"},
        "evidence_type": {"type": "string", "enum": [
            "title_text", "player_ui", "scene", "person_only", "none"]},
    },
    "required": ["content_type", "title", "confidence", "detail",
                 "evidence", "evidence_type"],
    # evidence is ordered BEFORE confidence on purpose: the model fills fields in
    # this order, so it must commit to what it actually saw before it grades how
    # sure it is. Grading first and justifying afterwards is how "high" got
    # attached to a face it merely recognised.
    "propertyOrdering": ["content_type", "title", "year", "season", "episode",
                         "evidence", "evidence_type", "confidence", "detail"],
}


class ProviderError(Exception):
    """Raised with a user-facing message when the vision call fails."""


class ModelUnavailable(ProviderError):
    """This model can't serve us right now — try the next one in the chain.

    Covers a spent daily allowance (429) and a model Google has renamed or
    withdrawn (404). Both are recoverable by falling through; neither should
    end a scan while another model is left to try.
    """


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


def _video_band(img: "Image.Image") -> Optional[Tuple[int, int]]:
    """The (top, bottom) rows holding the actual video in a phone screenshot.

    A Reel screenshot is mostly app furniture: status bar, like/comment/share
    rail, avatar, caption, comment box. Measured on real failing screenshots,
    the film itself is only 35-38% of the frame — so scaling the whole picture
    down to the size cap spent two thirds of the budget on Instagram's UI and
    left the faces we need to recognise at a third of their possible detail.

    Found by row colour-variance: interface rows are flat, film rows are not.
    Returns None when the result cannot be trusted, and the caller falls back.
    """
    small = img.convert("L")
    small.thumbnail((96, 192))           # detection needs shape, not detail
    w, h = small.size
    if h < 40:
        return None

    # Spread of brightness ACROSS each row. Interface rows are near-uniform;
    # film rows are not.
    #
    # Measured alternative, rejected: mean absolute horizontal change (an edge
    # metric) computed entirely in PIL's C code. It is much faster but wrong
    # here — a single sharp UI line scored 68 while real film texture scored
    # 8-12, so a relative threshold discarded the video and kept the chrome.
    # The Python loop below costs ~8ms against the ~73ms this frame spends
    # being decoded, so it was never the bottleneck worth removing.
    px = small.load()
    rows = []
    for y in range(h):
        vals = [px[x, y] for x in range(w)]
        mean = sum(vals) / w
        rows.append((sum((v - mean) ** 2 for v in vals) / w) ** 0.5)

    # An absolute floor as well as a relative one. A purely relative threshold
    # is meaningless on a flat frame: float noise makes every row's variance
    # ~7e-15, the cutoff ~2e-15, and every row then reads as "busy" — handing
    # back an 87% band for a picture containing no video whatsoever.
    peak = max(rows)
    if peak < MIN_ROW_VARIANCE:
        return None                       # nothing here looks like video
    cutoff = max(peak * 0.35, MIN_ROW_VARIANCE)
    busy = [r > cutoff for r in rows]
    for y in range(int(h * 0.06)):       # status bar / notification zone
        busy[y] = False
    for y in range(int(h * 0.93), h):    # comment box
        busy[y] = False

    best = run = None
    for y in range(h + 1):
        if y < h and busy[y]:
            run = y if run is None else run
        elif run is not None:
            if best is None or (y - run) > (best[1] - best[0]):
                best = (run, y)
            run = None
    if best is None:
        return None
    # A dark clip has little row variance and yields a uselessly thin band —
    # measured at 3.9% of frame height on one real screenshot. Sending a bad
    # crop is worse than sending none, so reject it and let the caller default.
    if (best[1] - best[0]) < h * 0.20:
        return None
    scale = img.size[1] / h
    return int(best[0] * scale), int(best[1] * scale)


def _crop_to_content(img: "Image.Image") -> "Image.Image":
    """Trim a tall phone screenshot down to the video inside it.

    Only applied to portrait frames taller than 1.6:1 — that shape is a phone
    screenshot. A landscape TV grab is left whole, because there the title can
    legitimately sit outside the picture area.
    """
    w, h = img.size
    if h < w * 1.6:
        return img
    band = _video_band(img)
    if band is None:
        band = (int(h * 0.22), int(h * 0.78))   # safe default, never skipped
    top, bottom = band
    return img.crop((0, top, w, bottom))


def shrink(raw: bytes) -> bytes:
    """Prepare one frame for the vision model.

    Crops away phone chrome, then scales to the size cap. Deliberately does NOT
    re-encode an image that is already a small JPEG: the previous version always
    saved at quality 88, which on a re-compressed Reel screenshot made every
    single test file LARGER while stacking a second generation of artifacts on
    top of Instagram's. Spending bandwidth to damage the picture is the worst of
    both.
    """
    img = Image.open(io.BytesIO(raw))
    original = img.size
    img = _crop_to_content(img)
    cropped = img.size != original

    fits = max(img.size) <= MAX_IMAGE_EDGE
    if not cropped and fits and (img.format or "").upper() == "JPEG":
        return raw  # already small enough, and re-encoding could only hurt it

    img.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))   # never upscales
    if img.mode != "RGB":
        img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


def _encode_frames(blobs: List[bytes]) -> List[str]:
    """Base64 JPEGs, ready for the model. Unreadable frames are skipped.

    One corrupt frame in a burst must not sink the whole scan, so failures
    here are dropped rather than raised.
    """
    frames = []
    for blob in blobs[:MAX_FRAMES]:
        if not blob:
            continue
        try:
            frames.append(standard_b64encode(shrink(blob)).decode())
        except Exception:
            continue
    return frames


def _usable_audio(data: bytes, mime: str) -> Optional[Dict[str, str]]:
    """Audio the model can use, or None. Never raises.

    Audio is a bonus signal: when it's missing, oversized or the wrong type we
    carry on with the frames alone rather than failing the scan.
    """
    if not data or len(data) > MAX_AUDIO_BYTES:
        return None
    base = (mime or "").split(";")[0].strip().lower()
    if base not in AUDIO_MIME_TYPES:
        return None
    return {"mime": base, "data": standard_b64encode(data).decode()}


def _gemini_parts(frames: List[str], audio: Optional[Dict[str, str]]) -> List[dict]:
    """Assemble one request: the frames in capture order, then any audio, then
    the instructions."""
    parts: List[dict] = [{"inlineData": {"mimeType": "image/jpeg", "data": f}}
                         for f in frames]
    if audio:
        parts.append({"inlineData": {"mimeType": audio["mime"], "data": audio["data"]}})
    parts.append({"text": IDENTIFY_PROMPT})
    return parts


def _itunes_upsize(url: str, box: int = 1200) -> str:
    """Swap Apple's default 100x100 thumbnail for a much larger box.

    Apple's iTunes Search API always returns "100x100bb" in artworkUrl100
    regardless of the source image's real resolution; requesting a bigger
    box gets back the best resolution Apple actually has, up to `box`.
    """
    return url.replace("100x100bb", f"{box}x{box}bb")


def _norm(s: str) -> str:
    """Fold a title down to just its comparable characters.

    Keeps letters and digits from ANY script. The earlier [^a-z0-9] version
    silently flattened every Devanagari, Japanese, Korean and Cyrillic title
    to the empty string, so a native-script title could never match anything —
    it only ever looked correct because the catalogs were being asked for
    romanized titles. NFKC first so composed and decomposed forms agree.

    Combining marks are kept as well as base letters: Indic vowel signs are
    marks, not alphanumerics, so dropping them would fold "सीता" and "सिता"
    onto one another. ASCII behaviour is unchanged — punctuation, spacing and
    case still fall away.
    """
    return "".join(
        ch for ch in unicodedata.normalize("NFKC", s).casefold()
        if ch.isalnum() or unicodedata.category(ch) in ("Mn", "Mc")
    )


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


def _tmdb_names(res: dict) -> List[str]:
    """Every name TMDB holds for one result.

    Indian releases are routinely indexed under a romanized `title` AND a
    native-script `original_title` — sometimes with the two swapped. Checking
    only `title`, as the where-to-watch lookup originally did, is a large part
    of why Hindi and South Indian films came back with nothing attached.
    """
    return [res.get("title") or "", res.get("name") or "",
            res.get("original_title") or "", res.get("original_name") or ""]


def _tmdb_matches(res: dict, title: str) -> bool:
    """True when any of TMDB's names for a result is exactly the title we want.

    Deliberately still an exact (normalized) comparison. Widening this to a
    prefix or substring test would reintroduce the "Avatar" vs "Avatar: The
    Last Airbender" mismatch — a missing poster is a shrug, a confidently
    wrong one is a visible bug.
    """
    want = _norm(title)
    return bool(want) and any(_norm(name) == want for name in _tmdb_names(res))


async def _tmdb_find(http: httpx.AsyncClient, media: str,
                     content: ScreenContent) -> Optional[dict]:
    """The single TMDB entry we're confident a scan refers to, or None.

    Shared by the poster and where-to-watch lookups so the two can never
    disagree about which film they're describing.
    """
    params: dict = {"api_key": TMDB_API_KEY, "query": content.title}
    year_key = "primary_release_year" if media == "movie" else "first_air_date_year"
    if content.year:
        params[year_key] = content.year

    url = f"https://api.themoviedb.org/3/search/{media}"
    r = await http.get(url, params=params)
    results = r.json().get("results", [])
    if not results and year_key in params:
        # Retry without the year — the AI's year guess can be off by one.
        params.pop(year_key)
        r = await http.get(url, params=params)
        results = r.json().get("results", [])

    return next((res for res in results if _tmdb_matches(res, content.title)), None)


async def tmdb_poster(content: ScreenContent) -> Optional[str]:
    """Cover art from TMDB — the only source here that carries Indian cinema.

    Apple's catalog has effectively no Hindi or South Indian film, so every
    such scan showed a blank tile no matter how well it was recognized. TMDB
    indexes them, which is the whole reason this sits alongside iTunes.

    w780 rather than `original`: TMDB originals are routinely 2000x3000 and
    several hundred KB, and the largest this is ever drawn is a phone-width
    cover. Not worth the mobile data.
    """
    if not TMDB_API_KEY or content.content_type not in ("movie", "tv_show", "anime"):
        return None
    media = "movie" if content.content_type == "movie" else "tv"
    try:
        async with httpx.AsyncClient(timeout=4.0) as http:
            match = await _tmdb_find(http, media, content)
    except Exception:
        return None
    path = (match or {}).get("poster_path")
    return TMDB_IMAGE_BASE + path if path else None


async def fetch_poster(content: ScreenContent) -> Optional[str]:
    """Poster art for any recognized movie/TV/anime.

    Two catalogs, strict title matching on both, so we never show the wrong
    poster. Apple is preferred when it has the title (higher resolution, and
    it needs no key at all); TMDB covers everything Apple doesn't, which in
    practice means most of Indian cinema.

    The two are queried CONCURRENTLY rather than one-after-the-other: the
    fallback exists for titles Apple has never carried, and paying its latency
    serially on every single scan would slow the common case down for nothing.

    Never raises — a poster lookup failing must not break recognition. Returns
    None when neither source is confident, and the card simply shows no poster.
    """
    if content.content_type not in ("movie", "tv_show", "anime"):
        return None
    # return_exceptions keeps one source blowing up from taking the other down.
    found = await asyncio.gather(itunes_poster(content), tmdb_poster(content),
                                 return_exceptions=True)
    for poster in found:  # gather preserves order, so Apple wins ties
        if isinstance(poster, str) and poster:
            return poster
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
async def _gemini_once(model: str, parts: List[dict]) -> ScreenContent:
    """One attempt against one model. Raises ModelUnavailable on 429 or 404."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_SCHEMA,
            "temperature": 0,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=45.0) as http:
            r = await http.post(url, params={"key": GEMINI_API_KEY}, json=body)
    except httpx.HTTPError:
        raise ProviderError("⚠️ No Clú: couldn't reach Google — check your internet")

    # Check for recoverable errors (fall through to the next model):
    # - 429: daily quota exhausted
    # - 404: model renamed or withdrawn by Google
    # - 5xx (500-599): transient server-side error on Google's end
    # These are recoverable; another model in the chain might succeed.
    if r.status_code in (429, 404) or 500 <= r.status_code < 600:
        raise ModelUnavailable(f"{model} unavailable (HTTP {r.status_code})")
    # Configuration errors must propagate immediately to avoid masking misconfigurations:
    if r.status_code in (400, 401, 403):
        raise ProviderError("⚠️ No Clú: Gemini key is missing or invalid — check the server settings")
    if r.status_code != 200:
        raise ProviderError(f"⚠️ No Clú: Gemini error ({r.status_code}) — try again")

    try:
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, ValueError):
        raise ProviderError("🔍 Gemini didn't return a result — try again.")
    return _cap_confidence(_parse_json_blob(text))


CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

# The most a model is ALLOWED to claim, given what it says it actually had.
# Asking politely does not work: with the rule written in the prompt, all three
# real screenshots still came back medium-or-high on evidence that read "The
# actress Pooja Hegde is clearly visible in this frame" — recognising a face
# and then naming a film from her filmography, which is the exact failure the
# rule forbids. A ceiling applied in code is not optional, so it holds for
# every provider including the ones that ignore instructions.
EVIDENCE_CEILING = {
    "title_text": "high",    # the title is written on screen — nothing beats that
    "player_ui": "high",     # the player names it
    "scene": "medium",       # a specific scene recognised, but nothing confirms it
    "person_only": "low",    # knowing WHO is not knowing WHAT
    "none": "low",
}


def _cap_confidence(content: ScreenContent) -> ScreenContent:
    """Lower a claim the stated evidence cannot support. Never raises it."""
    ceiling = EVIDENCE_CEILING.get(content.evidence_type, "low")
    if CONFIDENCE_RANK.get(content.confidence, 0) <= CONFIDENCE_RANK[ceiling]:
        return content
    return content.model_copy(update={"confidence": ceiling})

# How far a model's judgement is trusted when two answers are equally confident.
# Position in the chain is about SPEED, not authority, so strength is declared
# separately: the chain ends with a high-volume backstop, which must never
# outrank the frontier model that ran before it.
STRENGTH_BACKSTOP = 0     # large free quota, weaker judgement
STRENGTH_FAST = 1         # the everyday model
STRENGTH_STRONG = 2       # the better model in the same family
STRENGTH_FRONTIER = 3     # escalation target

# A model that just reported "out of quota" will almost certainly say the same
# on the next scan, so re-asking wastes a round trip on EVERY subsequent scan.
# Five minutes is a deliberate middle: long enough that a daily-exhausted model
# stops being asked once per scan, short enough to recover quickly from a
# per-minute limit (Groq's free tier is 30/min), and self-healing without any
# knowledge of which kind of limit was hit.
MODEL_COOLDOWN_SECONDS = 300
_model_cooldown: Dict[str, float] = {}

# Why each provider last failed — status codes only, never response bodies.
# A provider that fails silently is worse than one that is absent: escalation
# looked configured and healthy for hours while never once being consulted,
# because a failure and a cooldown are indistinguishable from the outside.
_provider_last_error: Dict[str, str] = {}


def _on_cooldown(label: str) -> bool:
    until = _model_cooldown.get(label)
    if until is None:
        return False
    if time.monotonic() >= until:
        del _model_cooldown[label]
        return False
    return True


async def _best_of(attempts) -> ScreenContent:
    """Walk (label, attempt) pairs in order, stopping at the first sure answer.

    Shared by the Gemini-only chain and the full cross-provider chain so the two
    can never drift apart.

    Only "high" stops the walk. "High" means the title was read off the screen
    or the scene was unmistakable; anything less is inference, and inference is
    exactly what a stronger model should be asked to check. This was measured:
    across the real failing screenshots and a 12-run repeat, every wrong answer
    that survived the anti-guessing rules was a "medium" — including "Sacred
    Games" for Jawaani Jaaneman, which stopped the chain before it ever reached
    GPT-4o.

    Ties go to the STRONGER model, which is not the same as the later one — a
    mistake that cost a working answer. The chain runs fast-then-fallback, so
    the last entry is Groq, a high-volume backstop, not an authority. Letting
    position decide ties handed Crew from GPT-4o's correct "Crew" to Groq's
    confident "The Buckingham Murders" — a different Kareena Kapoor film, i.e.
    exactly the actor-over-extension this whole design exists to stop. Strength
    is therefore declared per attempt and compared explicitly.

    Two different failures land here and both degrade rather than break: a model
    being out of quota (skipped, and remembered so it is not re-asked), and a
    model being unsure (kept, but keep looking). Only a chain where nothing
    answered at all raises.
    """
    best: Optional[ScreenContent] = None
    best_score = None
    for label, strength, attempt in attempts:
        if _on_cooldown(label):
            continue
        try:
            result = await attempt()
        except ModelUnavailable as exc:
            _model_cooldown[label] = time.monotonic() + MODEL_COOLDOWN_SECONDS
            _provider_last_error[label] = str(exc)
            continue   # out of quota, withdrawn, or a bad key on an optional extra
        # Confidence first, model strength only as the tie-break: a weaker model
        # being surer of itself does not make it right.
        result = result.model_copy(update={"source": label})
        score = (CONFIDENCE_RANK.get(result.confidence, 0), strength)
        if best_score is None or score > best_score:
            best, best_score = result, score
        if best.confidence == "high":
            return best    # read off the screen — nothing to gain from asking again
    if best is not None:
        return best        # nobody was certain; the best of them still beats nothing
    raise ProviderError("⚠️ No Clú: hit the free daily limit — try again after midnight!")


async def identify_gemini(frames: List[str],
                          audio: Optional[Dict[str, str]] = None) -> ScreenContent:
    """Identify via the Gemini model chain: fastest first, escalating when needed.

    The chain is ordered FAST-first, not best-first. Measured on a 1080p frame:
    gemini-flash-lite answers in ~1.4s, gemini-flash in ~3.5s, and on ordinary
    content they agree. Leading with the slow model made every single scan pay
    for the hard ones.

    The models still report their own confidence, so we don't have to guess:
    a "low" answer is handed to the next (stronger) model in the chain, and its
    answer wins if it is more sure. That keeps the common scan at ~1.4s while
    the genuinely ambiguous ones still get the better model's attention.

    A model whose daily free quota is gone is skipped rather than failing the
    scan, exactly as before.
    """
    if not GEMINI_API_KEY:
        raise ProviderError("⚠️ No Clú: Gemini key missing — add GEMINI_API_KEY")
    parts = _gemini_parts(frames, audio)
    return await _best_of([
        (m, STRENGTH_FAST if i == 0 else STRENGTH_STRONG, partial(_gemini_once, m, parts))
        for i, m in enumerate(GEMINI_MODELS)])


# --- Anthropic (paid) --------------------------------------------------------
def _why(response: "httpx.Response") -> str:
    """A short, safe slice of a provider's error message.

    Truncated hard, and anything token-shaped is dropped — this surfaces on the
    public health endpoint, so it must carry a diagnosis and nothing else.
    """
    try:
        body = response.json()
        message = body.get("error", {})
        message = message.get("message") if isinstance(message, dict) else str(message)
    except Exception:
        message = response.text
    message = " ".join(str(message or "").split())[:160]
    return " ".join(w for w in message.split()
                    if not any(w.startswith(p) for p in ("ghp_", "github_pat_", "gsk_", "Bearer")))


async def _openai_compatible_once(provider: Dict[str, str],
                                  frames: List[str]) -> ScreenContent:
    """One attempt against any OpenAI-compatible vision endpoint.

    Images only — none of these free tiers accept the audio clip Gemini can
    take, so a scan that reaches here is judged on the picture alone.

    ONE frame, not the five Gemini gets. Images dominate the token bill on these
    tiers and the budgets are small: GitHub Models allows 8K input tokens per
    request, and Groq's free tier caps at 8,000 tokens PER MINUTE across every
    request — a second frame roughly doubles that for a still Reel where the
    extra moment adds almost nothing.

    Raises ModelUnavailable on 429/404/5xx so the caller falls through, matching
    _gemini_once. Auth failures also fall through rather than killing the scan —
    a misconfigured backstop must never take down a working primary.
    """
    body = {
        "model": provider["model"],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{
            "role": "user",
            "content": (
                [{"type": "text", "text": IDENTIFY_PROMPT}] +
                [{"type": "image_url",
                  "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
                 for f in frames[:1]]
            ),
        }],
    }
    url = provider["base"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {provider['key']}",
               "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            r = await http.post(url, json=body, headers=headers)
            # Groq's qwen3.6-27b rejects our request with "Failed to validate
            # JSON. Please adjust your prompt." — its JSON mode refusing its own
            # output. Asking for JSON mode is an optimisation, not a
            # requirement: the prompt already demands a bare JSON object and
            # _parse_json_blob copes with fences, prose and truncation. So drop
            # the constraint and ask once more rather than lose the provider.
            if r.status_code == 400 and "json" in _why(r).lower():
                body.pop("response_format", None)
                r = await http.post(url, json=body, headers=headers)
    except httpx.HTTPError:
        raise ModelUnavailable(f"{provider['label']} unreachable")

    if r.status_code != 200:
        # Everything is recoverable here, including 401/403: these providers are
        # optional extras, so a bad key must degrade to the next one silently
        # rather than surface as a failed scan.
        #
        # Carry a short slice of the provider's own message. A bare status code
        # cost real time tonight: "400" alone cannot distinguish an unsupported
        # response_format from a malformed image part, and "401" cannot
        # distinguish a wrong token from a missing permission.
        raise ModelUnavailable(
            f"{provider['label']} unavailable (HTTP {r.status_code}) {_why(r)}")
    try:
        text = r.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError):
        raise ModelUnavailable(f"{provider['label']} returned no usable content")
    return _cap_confidence(_parse_json_blob(text))


async def identify_anthropic(frames: List[str],
                             audio: Optional[Dict[str, str]] = None) -> ScreenContent:
    import anthropic  # imported lazily so Gemini-only users needn't configure it
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ProviderError("⚠️ No Clú: Anthropic key missing — add ANTHROPIC_API_KEY to server .env")
    if not frames:
        raise ProviderError("🔍 Couldn't read that screen — try again.")
    b64 = frames[0]  # Anthropic path stays single-frame, no audio
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


async def identify_content(frames: List[str],
                           audio: Optional[Dict[str, str]] = None) -> ScreenContent:
    """Identify the screen, escalating across models and then across providers.

    One chain, walked fastest-first: the Gemini models, then any extra free
    provider that is configured. A confident answer stops the walk immediately,
    so the ordinary scan still costs exactly one fast call. Only an unsure
    answer keeps going, and only an unsure answer ever reaches a paid-quality
    model like GPT-4o — which is what keeps a ~50/day free budget sufficient.

    Two distinct failures both land here and both degrade rather than break:
    a model being out of quota, and a model being unsure. In either case the
    best answer seen so far is returned; only a chain where nothing answered at
    all raises.
    """
    if PROVIDER == "anthropic":
        return await identify_anthropic(frames, audio)
    if PROVIDER != "gemini":
        raise ProviderError(f"⚠️ No Clú: unknown PROVIDER '{PROVIDER}' — use 'gemini' or 'anthropic'")

    extras = _extra_providers()
    if not GEMINI_API_KEY and not extras:
        raise ProviderError("⚠️ No Clú: Gemini key missing — add GEMINI_API_KEY")

    attempts = []
    if GEMINI_API_KEY:
        parts = _gemini_parts(frames, audio)
        attempts += [
            (m, STRENGTH_FAST if i == 0 else STRENGTH_STRONG, partial(_gemini_once, m, parts))
            for i, m in enumerate(GEMINI_MODELS)]
    attempts += [(p["label"], p["strength"], partial(_openai_compatible_once, p, frames))
                 for p in extras]
    return await _best_of(attempts)


# An IP's country does not change between scans, so the geo lookup is worth
# remembering. This was a blocking 1.5s-timeout round trip on the front of
# EVERY scan, usually just to rediscover the country we already default to.
_COUNTRY_CACHE: Dict[str, str] = {}
_COUNTRY_CACHE_MAX = 512
_GEO_TIMEOUT = 1.0


async def resolve_country(request: Request, country: Optional[str]) -> str:
    if country:
        c = country.strip().upper()
        if len(c) == 2 and c.isalpha():  # only a real 2-letter code; else fall through
            return c
    # When deployed on the public internet, geolocate the caller's IP.
    ip = request.client.host if request.client else ""
    if not ip:
        return DEFAULT_COUNTRY
    cached = _COUNTRY_CACHE.get(ip)
    if cached:
        return cached
    try:
        if ipaddress.ip_address(ip).is_private:
            return DEFAULT_COUNTRY
        async with httpx.AsyncClient(timeout=_GEO_TIMEOUT) as http:
            r = await http.get(f"https://ipapi.co/{ip}/country/")
        if r.status_code == 200 and len(r.text.strip()) == 2:
            resolved = r.text.strip().upper()
            if len(_COUNTRY_CACHE) >= _COUNTRY_CACHE_MAX:
                _COUNTRY_CACHE.clear()  # tiny cache; a full reset is fine and keeps it bounded
            _COUNTRY_CACHE[ip] = resolved
            return resolved
    except Exception:
        pass
    # Remember the failure too, so a slow or blocked geo service is paid for
    # once per address instead of on every scan from it.
    _COUNTRY_CACHE[ip] = DEFAULT_COUNTRY
    return DEFAULT_COUNTRY


async def tmdb_where_to_watch(content: ScreenContent, country: str) -> Optional[dict]:
    """Look up streaming availability for movies/TV in the given country."""
    if not TMDB_API_KEY or content.content_type not in ("movie", "tv_show", "anime"):
        return None

    media = "movie" if content.content_type == "movie" else "tv"

    try:
        async with httpx.AsyncClient(timeout=4.0) as http:
            # Same confident-match search the poster uses, so the streaming
            # info and the cover art can never describe different films.
            match = await _tmdb_find(http, media, content)
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
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/ScrollTrigger.min.js"></script>
<script src="https://cdn.jsdelivr.net/gh/studio-freight/lenis@1.0.29/bundled/lenis.min.js"></script>
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
<!-- UI: Premium Dark Mode & GSAP Onboarding Overhaul -->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, maximum-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="No Clu">
<meta name="theme-color" content="#0A0614">
<link rel="apple-touch-icon" href="/app-icon.png">
<link rel="icon" href="/app-icon.png">
<title>No Clú</title>
<style>
  /* ==== NEON NIGHT ====================================================
     A single committed visual world: signage after dark. Deep violet
     ground, an ambient wash of pink/cyan/violet behind everything, and
     controls drawn as lit glass tubes rather than filled slabs. It does
     not follow the viewer's light/dark preference on purpose — a neon
     sign has one state, and inverting it would destroy the idea. */
  :root{
    --bg:#0A0614; --ink:#F6E9FF; --muted:#B79DD0; --faint:#7B6B94;
    --card:rgba(28,13,50,.62); --err:#FF6E8A;
    --pink:#FF2D95; --pink-soft:#FF9AC9; --cyan:#00E5FF; --cyan-soft:#9BE7FF; --violet:#7B2DFF;
    --edge:rgba(255,45,149,.30); --edge-cyan:rgba(0,229,255,.26);
    --mono: ui-monospace,"SF Mono",Menlo,Consolas,monospace;
    --sans: -apple-system,BlinkMacSystemFont,"SF Pro Text",system-ui,sans-serif;
  }
  *{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
  html,body{height:100%}
  body{background:var(--bg);color:var(--ink);font-family:var(--sans);
       min-height:100dvh;display:flex;flex-direction:column;position:relative;
       padding:env(safe-area-inset-top) 0 env(safe-area-inset-bottom);-webkit-font-smoothing:antialiased}
  /* the sign glow — fixed so it never scrolls away, behind all content */
  body:before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
    background:
      radial-gradient(46% 26% at 16% 8%, rgba(255,45,149,.55), transparent 66%),
      radial-gradient(42% 24% at 88% 34%, rgba(0,229,255,.34), transparent 64%),
      radial-gradient(60% 32% at 44% 102%, rgba(123,45,255,.46), transparent 66%)}
  body>*{position:relative;z-index:1}
  @keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
  /* the brand is the sign itself: a white-hot core inside a pink bloom */
  .brand{font-family:var(--mono);font-weight:700;letter-spacing:3px;color:#FFF;
         text-shadow:0 0 1px #fff,0 0 8px var(--pink),0 0 22px var(--pink),0 0 46px rgba(255,45,149,.6)}

  /* ---- sign-in gate ---- */
  #gate{flex:1;display:none;flex-direction:column;align-items:center;justify-content:center;padding:30px 24px}
  #gate.show{display:flex}
  #gate .brand{font-size:34px;margin-bottom:22px}
  .authcard{width:100%;max-width:360px;background:var(--card);border:1px solid var(--edge);border-radius:20px;padding:22px;
            backdrop-filter:blur(22px) saturate(150%);-webkit-backdrop-filter:blur(22px) saturate(150%);
            box-shadow:0 0 0 1px rgba(255,255,255,.04) inset,0 18px 60px rgba(0,0,0,.55),0 0 40px rgba(255,45,149,.14)}
  .seg{display:flex;background:rgba(255,255,255,.05);border-radius:13px;padding:4px;margin-bottom:18px}
  .seg button{flex:1;font-family:var(--mono);font-size:12px;letter-spacing:1px;text-transform:uppercase;
              padding:9px;border:none;border-radius:10px;background:transparent;color:var(--muted);cursor:pointer}
  .seg button.on{background:linear-gradient(135deg,var(--pink),var(--violet));color:#fff;font-weight:700;
                 box-shadow:0 0 18px rgba(255,45,149,.55)}
  .field{margin-bottom:12px}
  .field input{width:100%;background:rgba(10,6,20,.55);border:1px solid var(--edge-cyan);border-radius:12px;
               padding:14px;color:var(--ink);font-size:16px}
  .field input::placeholder{color:#9A88B6}
  .field input:focus{outline:none;border-color:var(--cyan);box-shadow:0 0 0 3px rgba(0,229,255,.16),0 0 18px rgba(0,229,255,.3)}
  .err{color:var(--err);font-size:13px;min-height:16px;margin:2px 0 10px}
  /* neon tube: lit outline, not a painted slab */
  .primary{width:100%;height:50px;border:1.5px solid var(--pink);border-radius:13px;
           background:rgba(255,45,149,.12);color:#FFE9F5;
           font-family:var(--mono);font-weight:700;font-size:14px;letter-spacing:1.5px;text-transform:uppercase;cursor:pointer;
           text-shadow:0 0 10px rgba(255,45,149,.9);
           box-shadow:0 0 20px rgba(255,45,149,.45),inset 0 0 20px rgba(255,45,149,.18)}
  .primary:active{transform:scale(.985)}
  .primary:disabled{opacity:.55;box-shadow:none}

  /* ---- main app ---- */
  #app{flex:1;display:none;flex-direction:column}
  #app.show{display:flex}
  .top{display:flex;align-items:center;justify-content:space-between;padding:16px 22px 4px}
  .top .brand{font-size:24px}
  .who{display:flex;align-items:center;gap:10px}
  .who .name{font-family:var(--mono);font-size:11px;color:var(--muted);max-width:130px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .who .out{font-family:var(--mono);font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--cyan-soft);
            border:1px solid var(--edge-cyan);border-radius:20px;padding:6px 10px;background:rgba(0,229,255,.06);cursor:pointer}
  main{flex:1;display:flex;flex-direction:column;align-items:center;padding:10px 22px 30px;overflow-y:auto}
  /* ---- the prism lens: a 5.4s prism-led sequence -----------------------
     Fold edge-on, wind down, collapse to a white point, then refract out as a
     spectrum. `.on` is set on .stage (an ANCESTOR of .pi), so every state rule
     must read `.stage.on .pi ...` — putting both classes on one selector
     silently disables the whole animation. -------------------------------- */
  /* The blades travel ~186% of their own size and deliberately overflow the
     stage — clipping them mid-flight looks broken, and `overflow:hidden` is
     defeated by preserve-3d anyway. The stage just reserves the vertical
     space so they never land on the status text or the history list. */
  .stage{position:relative;width:min(300px,86vw);aspect-ratio:1;display:flex;align-items:center;
         justify-content:center;flex-shrink:0}
  .pi{position:relative;display:grid;place-items:center;width:150px;height:150px;
      perspective:760px;cursor:pointer;-webkit-tap-highlight-color:transparent;
      opacity:1;transition:opacity .3s ease}
  .pi>*{position:absolute;pointer-events:none}
  .halo{inset:-26%;border-radius:50%;opacity:.5;
        background:radial-gradient(circle,rgba(255,45,149,.26),rgba(123,45,255,.14) 40%,transparent 64%)}
  .prism-rim{inset:0;border-radius:26%;opacity:.28;transform:rotate(45deg);
    background:conic-gradient(from 210deg,rgba(155,107,255,.6),rgba(255,45,149,.55),
      rgba(123,227,192,.6),rgba(0,229,255,.55),rgba(155,107,255,.6));
    -webkit-mask:radial-gradient(circle,transparent 62%,#000 64%);
            mask:radial-gradient(circle,transparent 62%,#000 64%)}
  .prog{inset:-15%;border-radius:50%;opacity:0;
        background:conic-gradient(rgba(0,229,255,.6) 0deg,transparent 0deg)}
  .shards{inset:0;transform:rotate(45deg);transform-style:preserve-3d}
  /* Over the neon wash, a 210% saturate boost turned the whole prism into one
     flat magenta slab and killed the facets. Held near 120% it reads as glass
     picking up the sign light, which is the point. */
  .sh{position:absolute;left:50%;top:50%;width:52%;height:52%;transform-origin:0 0;overflow:hidden;
    background:linear-gradient(140deg,rgba(255,255,255,.58),rgba(255,255,255,.14) 46%,rgba(255,255,255,.36));
    backdrop-filter:blur(13px) saturate(155%);
    -webkit-backdrop-filter:blur(13px) saturate(155%);
    border:.5px solid rgba(255,255,255,.55);border-radius:3px 22% 3px 3px;
    box-shadow:inset 0 1.5px 0 rgba(255,255,255,.8),0 8px 20px rgba(0,0,0,.5)}
  .sh:before{content:"";position:absolute;inset:0;background:var(--c,#fff);opacity:0;mix-blend-mode:screen}
  .sh:after{content:"";position:absolute;inset:0;opacity:.6;
    box-shadow:inset 1.5px 0 0 rgba(155,107,255,.6),inset -1.5px 0 0 rgba(123,227,192,.6)}
  .sh:nth-child(1){transform:translate(-100%,-100%)}
  .sh:nth-child(2){transform:translate(0,-100%) rotate(90deg)}
  .sh:nth-child(3){transform:translate(0,0) rotate(180deg)}
  .sh:nth-child(4){transform:translate(-100%,0) rotate(270deg)}
  .core{width:26%;height:26%;border-radius:50%;transform:scale(.6);
    background:radial-gradient(circle at 36% 30%,#FFF0FA,var(--pink) 66%);
    box-shadow:0 0 24px 5px rgba(255,45,149,.6)}
  .beam{left:50%;top:50%;width:3px;height:52%;transform-origin:50% 0;opacity:0;border-radius:2px;
        transform:rotate(var(--a)) scaleY(.1)}
  .flash{inset:-8%;border-radius:50%;opacity:0;background:radial-gradient(circle,#fff,transparent 58%)}

  .stage.on .shards{animation:pl_cl 5.4s cubic-bezier(.22,.9,.28,1) forwards}
  @keyframes pl_cl{0%{transform:rotate(45deg) rotateX(0) scale(1)}
    14%{transform:rotate(45deg) rotateX(72deg)}
    30%{transform:rotate(280deg) rotateX(72deg) scale(.4)}
    38%{transform:rotate(340deg) rotateX(72deg) scale(.1)}
    48%{transform:rotate(370deg) rotateX(26deg) scale(1.06)}
    100%{transform:rotate(400deg) rotateX(0) scale(1)}}
  .stage.on .sh{animation:pl_sh 5.4s cubic-bezier(.22,.9,.28,1) forwards}
  .stage.on .sh:nth-child(2){animation-name:pl_sh2}
  .stage.on .sh:nth-child(3){animation-name:pl_sh3}
  .stage.on .sh:nth-child(4){animation-name:pl_sh4}
  @keyframes pl_sh{0%,38%{transform:translate(-100%,-100%)}
    100%{transform:translate(-186%,-158%) rotateY(38deg)}}
  @keyframes pl_sh2{0%,38%{transform:translate(0,-100%) rotate(90deg)}
    100%{transform:translate(106%,-186%) rotate(90deg) rotateY(38deg)}}
  @keyframes pl_sh3{0%,38%{transform:translate(0,0) rotate(180deg)}
    100%{transform:translate(158%,106%) rotate(180deg) rotateY(38deg)}}
  @keyframes pl_sh4{0%,38%{transform:translate(-100%,0) rotate(270deg)}
    100%{transform:translate(-186%,158%) rotate(270deg) rotateY(38deg)}}
  .stage.on .sh:before{animation:pl_tint 5.4s cubic-bezier(.22,.9,.28,1) forwards}
  @keyframes pl_tint{0%,36%{opacity:0}48%{opacity:.85}100%{opacity:.5}}
  .stage.on .core{animation:pl_core 5.4s cubic-bezier(.22,.9,.28,1) forwards}
  @keyframes pl_core{0%{transform:scale(.6)}
    34%{transform:scale(.1);background:radial-gradient(circle,#fff,#fff)}
    42%{transform:scale(1.9);box-shadow:0 0 80px 26px rgba(255,255,255,.9)}
    56%{transform:scale(.85);background:radial-gradient(circle at 36% 30%,#fff,#FF7A8A 60%)}
    100%{transform:scale(1.1);background:radial-gradient(circle at 36% 30%,#fff,var(--pink) 62%);
      box-shadow:0 0 56px 14px rgba(255,45,149,.65)}}
  /* the flash is softened from the mockup — a full-white frame is harsh at night */
  .stage.on .flash{animation:pl_flash 5.4s linear forwards}
  @keyframes pl_flash{0%,34%{opacity:0;transform:scale(.3)}
    40%{opacity:.6;transform:scale(1)}50%{opacity:0;transform:scale(1.7)}100%{opacity:0}}
  .stage.on .beam{animation:pl_beam 5.4s cubic-bezier(.22,.9,.28,1) forwards;
                  animation-delay:calc(2.1s + var(--d,0s))}
  @keyframes pl_beam{0%{opacity:0;transform:rotate(var(--a)) scaleY(.06)}
    8%{opacity:1;transform:rotate(var(--a)) scaleY(1.05)}
    50%{opacity:.9;transform:rotate(calc(var(--a) + 30deg)) scaleY(1.3)}
    100%{opacity:0;transform:rotate(calc(var(--a) + 48deg)) scaleY(1.4)}}
  .stage.on .halo{animation:pl_halo 5.4s cubic-bezier(.22,.9,.28,1) forwards}
  @keyframes pl_halo{0%{opacity:.5;transform:scale(1)}16%{opacity:.18;transform:scale(.84)}
    50%{opacity:.4}62%{opacity:1;transform:scale(1.16)}100%{opacity:.9;transform:scale(1)}}
  .stage.on .prism-rim{animation:pl_rim 5.4s cubic-bezier(.4,0,.2,1) forwards}
  @keyframes pl_rim{0%{opacity:.28;transform:rotate(45deg)}14%{opacity:.12;transform:rotate(24deg)}
    52%{opacity:.7;transform:rotate(560deg)}100%{opacity:.85;transform:rotate(700deg)}}
  .stage.on .prog{animation:pl_prog 5.4s linear forwards}
  @keyframes pl_prog{0%{opacity:0}5%{opacity:.7}
    93%{opacity:.5;background:conic-gradient(rgba(0,229,255,.6) 360deg,transparent 360deg)}
    100%{opacity:0;background:conic-gradient(rgba(0,229,255,.6) 360deg,transparent 360deg)}}
  /* closing: the open form contracts away, then the closed diamond fades back */
  .stage.closing .pi{animation:pl_collapse .8s cubic-bezier(.22,.9,.28,1) forwards}
  @keyframes pl_collapse{0%{opacity:1;transform:scale(1)}100%{opacity:0;transform:scale(.84)}}
  /* cyan tube, so it never competes with the pink primary action */
  .setupbtn{display:flex;align-items:center;justify-content:center;gap:9px;width:100%;margin-top:34px;
            height:52px;border-radius:26px;text-decoration:none;background:rgba(0,229,255,.08);
            border:1.5px solid var(--cyan);color:#EAFBFF;text-shadow:0 0 10px rgba(0,229,255,.8);
            box-shadow:0 0 20px rgba(0,229,255,.35),inset 0 0 18px rgba(0,229,255,.14);
            font-family:var(--mono);font-weight:700;font-size:12px;letter-spacing:1.5px;text-transform:uppercase}
  /* social sign-in — rendered only when the provider is actually configured */
  .or{display:flex;align-items:center;gap:10px;margin:18px 0 14px;color:var(--faint);
      font-family:var(--mono);font-size:10px;letter-spacing:1.5px;text-transform:uppercase}
  .or:before,.or:after{content:"";flex:1;height:1px;background:rgba(183,157,208,.22)}
  .social{display:flex;align-items:center;justify-content:center;gap:10px;width:100%;height:50px;
          border-radius:13px;text-decoration:none;font-size:15px;font-weight:600;margin-bottom:10px}
  .social.google{background:#F6E9FF;color:#1A1024}
  .social.apple{background:rgba(255,255,255,.04);color:var(--ink);border:1px solid rgba(246,233,255,.42)}
  .social:active{transform:scale(.98)}
  .modal{position:fixed;inset:0;background:rgba(6,3,14,.82);display:none;
         align-items:center;justify-content:center;padding:28px;z-index:50;
         backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
  .modal.show{display:flex}
  .modal .box{background:var(--card);border:1px solid var(--edge);border-radius:20px;
              backdrop-filter:blur(22px) saturate(150%);-webkit-backdrop-filter:blur(22px) saturate(150%);
              box-shadow:0 20px 60px rgba(0,0,0,.6),0 0 46px rgba(255,45,149,.18);
              padding:24px;max-width:340px;text-align:center;animation:fadeUp .35s ease-out}
  .modal h3{font-size:18px;margin-bottom:10px}
  .modal p{color:var(--muted);font-size:14px;line-height:1.55}
  .modal button{margin-top:20px;width:100%;height:46px;border:1.5px solid var(--pink);border-radius:13px;
                background:rgba(255,45,149,.12);color:#FFE9F5;font-family:var(--mono);font-weight:700;
                text-shadow:0 0 10px rgba(255,45,149,.9);box-shadow:0 0 18px rgba(255,45,149,.4);
                font-size:12px;letter-spacing:1.5px;text-transform:uppercase;cursor:pointer}
  .modal button.ghost{margin-top:10px;background:transparent;color:var(--muted);text-shadow:none;
                box-shadow:none;border:1px solid rgba(183,157,208,.3)}
  .seeall{display:block;margin-top:14px;text-align:center;text-decoration:none;
          font-family:var(--mono);font-size:11px;letter-spacing:1.5px;
          color:var(--cyan-soft);text-transform:uppercase;text-shadow:0 0 12px rgba(0,229,255,.6)}
  .recent{width:100%;margin-top:30px}
  .recent h2{font-family:var(--mono);font-size:10.5px;letter-spacing:2px;color:var(--faint);text-transform:uppercase;margin-bottom:10px}
  .ritem{display:flex;align-items:center;gap:12px;padding:10px 0;text-decoration:none;color:inherit;
         border-bottom:1px solid rgba(0,229,255,.16)}
  .ritem .go{color:var(--cyan);font-size:17px;text-shadow:0 0 10px rgba(0,229,255,.8)}
  .ritem .rth{width:34px;height:48px;border-radius:5px;object-fit:cover;background:rgba(123,45,255,.25);flex-shrink:0}
  .ritem .rmain{flex:1;min-width:0}
  .ritem .rt{font-size:14px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .ritem .rm{font-family:var(--mono);font-size:10px;color:var(--cyan-soft)}
  /* The lens is a 5.4s light show; anyone who asked the OS for less motion
     gets the resting prism and no sequence at all. */
  @media (prefers-reduced-motion:reduce){
    .stage.on *,.stage.closing *{animation:none !important}
  }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/ScrollTrigger.min.js"></script>
<script src="https://cdn.jsdelivr.net/gh/studio-freight/lenis@1.0.29/bundled/lenis.min.js"></script>
</head>
<body>
  <!-- Sign-in gate -->
  <div id="gate">
    <div class="brand">No Clú</div>
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

      <div id="socialWrap" style="display:none">
        <div class="or"><span>or</span></div>
        <a class="social google" id="btnGoogle" href="/auth/google/start" style="display:none">
          <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 01-1.8 2.72v2.26h2.9c1.7-1.57 2.7-3.88 2.7-6.62z"/><path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.9-2.26c-.8.54-1.83.86-3.06.86-2.35 0-4.34-1.59-5.05-3.72H.96v2.33A9 9 0 009 18z"/><path fill="#FBBC05" d="M3.95 10.7A5.4 5.4 0 013.68 9c0-.59.1-1.17.27-1.7V4.97H.96A9 9 0 000 9c0 1.45.35 2.83.96 4.03l3-2.32z"/><path fill="#EA4335" d="M9 3.58c1.32 0 2.51.46 3.44 1.35l2.58-2.58C13.46.9 11.43 0 9 0A9 9 0 00.96 4.97l3 2.33C4.66 5.17 6.65 3.58 9 3.58z"/></svg>
          <span>Continue with Google</span>
        </a>
        <a class="social apple" id="btnApple" href="/auth/apple/start" style="display:none">
          <svg width="16" height="18" viewBox="0 0 16 18" fill="currentColor" aria-hidden="true"><path d="M13.1 9.3c0-2.15 1.75-3.18 1.83-3.23-1-1.46-2.55-1.66-3.1-1.68-1.32-.13-2.58.78-3.25.78-.67 0-1.7-.76-2.8-.74-1.44.02-2.77.84-3.51 2.13-1.5 2.6-.38 6.45 1.08 8.56.71 1.03 1.56 2.19 2.68 2.15 1.08-.04 1.48-.7 2.78-.7 1.3 0 1.66.7 2.8.68 1.15-.02 1.88-1.05 2.58-2.09.82-1.2 1.15-2.36 1.17-2.42-.03-.01-2.25-.86-2.26-3.44zM10.98 2.9c.59-.72.98-1.71.87-2.7-.85.03-1.87.56-2.48 1.28-.55.63-1.02 1.65-.9 2.62.94.07 1.9-.48 2.51-1.2z"/></svg>
          <span>Continue with Apple</span>
        </a>
      </div>
    </div>
  </div>

  <!-- Main app -->
  <div id="app">
    <div class="top">
      <div class="brand">No Clú</div>
      <div class="who"><span class="name" id="whoName"></span><button class="out" onclick="askLogout()">Sign out</button></div>
    </div>
    <main>
      <div class="stage" id="stage">
        <div class="pi" id="lens">
          <div class="halo"></div><div class="prog"></div><div class="prism-rim"></div>
          <div class="shards">
            <i class="sh" style="--c:#FF5E6B"></i><i class="sh" style="--c:#FFC04D"></i>
            <i class="sh" style="--c:#7BE3C0"></i><i class="sh" style="--c:#8FD4FF"></i>
          </div>
          <div class="flash"></div><div class="core"></div>
          <div class="beam" style="--a:-45deg;--d:0s;background:linear-gradient(#FF5E6B,transparent)"></div>
          <div class="beam" style="--a:-27deg;--d:.05s;background:linear-gradient(#FF9F45,transparent)"></div>
          <div class="beam" style="--a:-9deg;--d:.1s;background:linear-gradient(#FFE86B,transparent)"></div>
          <div class="beam" style="--a:9deg;--d:.15s;background:linear-gradient(#7BE3C0,transparent)"></div>
          <div class="beam" style="--a:27deg;--d:.2s;background:linear-gradient(#8FD4FF,transparent)"></div>
          <div class="beam" style="--a:45deg;--d:.25s;background:linear-gradient(#9B6BFF,transparent)"></div>
          <div class="beam" style="--a:135deg;--d:.3s;background:linear-gradient(#FF5E6B,transparent)"></div>
          <div class="beam" style="--a:225deg;--d:.35s;background:linear-gradient(#8FD4FF,transparent)"></div>
        </div>
      </div>
      <div class="recent" id="recentWrap" style="display:none">
        <h2>Recent scans</h2>
        <div id="recent"></div>
        <a class="seeall" href="/history">See all →</a>
      </div>

      <a class="setupbtn" href="/shortcut">⚡ Set up one-tap scanning</a>
    </main>
    <div class="modal" id="syncModal">
      <div class="box">
        <h3>Your scans are saved to your account</h3>
        <p>Everything you scan is stored here and available on any device you sign in from.</p>
        <button onclick="closeSyncPopup()">Got it</button>
      </div>
    </div>

    <div class="modal" id="logoutModal">
      <div class="box">
        <h3>Sign out of No Clú?</h3>
        <p>Your scans stay safe in your account — sign back in any time to see them.</p>
        <button onclick="confirmLogout()">Sign out</button>
        <button class="ghost" onclick="cancelLogout()">Cancel</button>
      </div>
    </div>
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
      if(d.ok){ await checkAuth(true); }
      else { err.textContent=d.error||'Something went wrong.'; }
    }catch(e){ err.textContent='Could not reach the server.'; }
    btn.disabled=false;
  }
  // Signing out is easy to hit by accident, so it takes a deliberate confirm.
  function askLogout(){ document.getElementById('logoutModal').classList.add('show'); }
  function cancelLogout(){ document.getElementById('logoutModal').classList.remove('show'); }
  async function confirmLogout(){
    document.getElementById('logoutModal').classList.remove('show');
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
  function showSyncPopup(){ document.getElementById('syncModal').classList.add('show'); }
  function closeSyncPopup(){ document.getElementById('syncModal').classList.remove('show'); }
  async function checkAuth(justSignedIn){
    try{
      var r=await fetch('/auth/me'); var d=await r.json();
      if(d.signed_in){
        showApp(d.name);
        // Only after an explicit sign-in — not every time the app reopens.
        if(justSignedIn) showSyncPopup();
      } else { showGate(); }
    }catch(e){ showGate(); }
  }

  async function loadHistory(){
    try{
      var r=await fetch('/api/history?limit=3'); if(r.status!==200) return;
      var d=await r.json(); var list=d.scans||[];
      if(!list.length){ document.getElementById('recentWrap').style.display='none'; return; }
      document.getElementById('recentWrap').style.display='block';
      document.getElementById('recent').innerHTML=list.map(function(s){
        var thumb=s.poster?('<img class="rth" src="'+esc(s.poster)+'" onerror="this.outerHTML=\\'<div class=rth></div>\\'">'):'<div class="rth"></div>';
        var m=[(s.type||'').replace('_',' ')];
        if(s.year) m.push(s.year);
        if(s.season && s.episode) m.push('S'+s.season+' · E'+s.episode);
        return '<a class="ritem" href="/title/'+s.id+'">'+thumb+
               '<div class="rmain"><div class="rt">'+esc(s.title)+
               '</div><div class="rm">'+esc(m.filter(Boolean).join(' · '))+
               '</div></div><div class="go">›</div></a>';
      }).join('');
    }catch(e){}
  }

  // ---- the lens ----
  // Decorative for now: iOS won't let a web page read your screen, so real
  // scanning happens from the Shortcut (Siri or Back Tap). Tapping toggles the
  // lens between idle and active and plays the pulse — it makes no claim to be
  // scanning, so it can't mislead.
  var lens=document.getElementById('lens'), stage=document.getElementById('stage');
  var lensBusy=false;
  // Fire-and-reset: one tap plays the full 5.4s sequence, then it closes itself
  // and returns to rest. Locked throughout so a second tap can't strand it.
  var LENS_OPEN_MS=5400, LENS_CLOSE_MS=820;
  function playLens(){
    if(lensBusy) return;
    lensBusy=true;
    stage.classList.add('on');
    setTimeout(function(){
      stage.classList.add('closing');
      setTimeout(function(){
        stage.classList.remove('on','closing');
        lensBusy=false;
      }, LENS_CLOSE_MS);
    }, LENS_OPEN_MS);
  }
  lens.addEventListener('click', playLens);

  // Show a social button only if its credentials exist server-side.
  (function(){
    var g = "__GOOGLE_ENABLED__" === "1", a = "__APPLE_ENABLED__" === "1";
    if(g) document.getElementById('btnGoogle').style.display='flex';
    if(a) document.getElementById('btnApple').style.display='flex';
    if(g || a) document.getElementById('socialWrap').style.display='block';
  })();

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
<meta name="theme-color" content="#0A0614">
<link rel="apple-touch-icon" href="/app-icon.png">
<link rel="icon" href="/app-icon.png">
<title>Set up one-tap scanning · No Clú</title>
<style>
  /* Neon night — see APP_HTML for the full rationale. */
  :root{
    --bg:#0A0614; --ink:#F6E9FF; --muted:#B79DD0; --faint:#7B6B94;
    --card:rgba(28,13,50,.62);
    --pink:#FF2D95; --pink-soft:#FF9AC9; --cyan:#00E5FF; --cyan-soft:#9BE7FF; --violet:#7B2DFF;
    --edge:rgba(255,45,149,.30); --edge-cyan:rgba(0,229,255,.26);
    --mono: ui-monospace,"SF Mono",Menlo,Consolas,monospace;
    --sans: -apple-system,BlinkMacSystemFont,"SF Pro Text",system-ui,sans-serif;
  }
  *{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
  body{background:var(--bg);color:var(--ink);font-family:var(--sans);min-height:100dvh;position:relative;
       padding:calc(env(safe-area-inset-top) + 18px) 22px calc(env(safe-area-inset-bottom) + 40px);
       -webkit-font-smoothing:antialiased;max-width:520px;margin:0 auto}
  body:before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
    background:
      radial-gradient(46% 26% at 16% 8%, rgba(255,45,149,.5), transparent 66%),
      radial-gradient(42% 24% at 88% 34%, rgba(0,229,255,.3), transparent 64%),
      radial-gradient(60% 32% at 44% 102%, rgba(123,45,255,.42), transparent 66%)}
  body>*{position:relative;z-index:1}
  a.back{font-family:var(--mono);font-size:12px;letter-spacing:1px;color:var(--cyan-soft);text-decoration:none}
  .brand{font-family:var(--mono);font-weight:700;letter-spacing:3px;color:#FFF;
         text-shadow:0 0 1px #fff,0 0 8px var(--pink),0 0 22px var(--pink);font-size:15px;margin:18px 0 4px}
  h1{font-size:25px;line-height:1.25;margin-top:8px}
  .sub{color:var(--muted);font-size:14px;line-height:1.55;margin-top:10px}
  .step{background:var(--card);border:1px solid var(--edge);border-radius:18px;padding:18px;margin-top:18px;
        backdrop-filter:blur(20px) saturate(150%);-webkit-backdrop-filter:blur(20px) saturate(150%);
        box-shadow:0 14px 44px rgba(0,0,0,.5),0 0 34px rgba(255,45,149,.12)}
  /* the step number is the one place numbering is honest — this is a real sequence */
  .step .n{font-family:var(--mono);font-size:11px;letter-spacing:1px;color:var(--pink-soft);text-transform:uppercase;
           text-shadow:0 0 12px rgba(255,45,149,.7)}
  .step h2{font-size:17px;margin-top:6px}
  .step p{color:var(--muted);font-size:13.5px;line-height:1.55;margin-top:8px}
  .step p b{color:var(--ink)}
  .urlbox{background:rgba(10,6,20,.6);border:1px solid var(--edge-cyan);border-radius:11px;padding:11px 12px;
          font-family:var(--mono);font-size:11px;color:var(--cyan-soft);word-break:break-all;line-height:1.5;margin-top:12px}
  .btn{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;height:50px;margin-top:12px;
       border:1.5px solid var(--pink);border-radius:13px;background:rgba(255,45,149,.12);color:#FFE9F5;
       cursor:pointer;text-decoration:none;text-shadow:0 0 10px rgba(255,45,149,.9);
       box-shadow:0 0 20px rgba(255,45,149,.45),inset 0 0 18px rgba(255,45,149,.16);
       font-family:var(--mono);font-weight:700;font-size:12.5px;letter-spacing:1.5px;text-transform:uppercase}
  .btn.ghost{background:rgba(0,229,255,.08);border-color:var(--cyan);color:#EAFBFF;
       text-shadow:0 0 10px rgba(0,229,255,.8);box-shadow:0 0 18px rgba(0,229,255,.32)}
  ol.manual{margin:12px 0 0 18px;color:var(--muted);font-size:13.5px;line-height:1.7}
  ol.manual b{color:var(--ink)}
  .trigger{padding:13px 0;border-bottom:1px solid rgba(0,229,255,.16)}
  .trigger:last-child{border-bottom:none}
  .tname{font-size:14.5px;font-weight:600;color:var(--ink);display:flex;align-items:center;gap:8px;flex-wrap:wrap}
  .tbadge{font-family:var(--mono);font-size:9.5px;letter-spacing:1px;text-transform:uppercase;
          padding:3px 7px;border-radius:20px;border:1px solid rgba(183,157,208,.35);color:var(--faint)}
  .tbadge.best{border-color:var(--cyan);color:#EAFBFF;background:rgba(0,229,255,.1);
          box-shadow:0 0 14px rgba(0,229,255,.35)}
  .tsteps{color:var(--muted);font-size:13px;line-height:1.55;margin-top:6px}
  .tsteps b{color:var(--ink)}
  .note{display:block;color:var(--faint);font-size:12px;margin-top:6px;line-height:1.5}
  code{font-family:var(--mono);font-size:12px;background:rgba(0,229,255,.12);color:var(--cyan-soft);
       padding:2px 6px;border-radius:5px;word-break:break-all}
  #need,#setup{display:none}
  #need.show,#setup.show{display:block}
  #need{text-align:center;padding-top:60px}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/ScrollTrigger.min.js"></script>
<script src="https://cdn.jsdelivr.net/gh/studio-freight/lenis@1.0.29/bundled/lenis.min.js"></script>
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
      </div>
      <div id="manual" style="display:none">
        <p>Build it once (about a minute). The last two actions are what make the result open in this app.</p>
        <ol class="manual">
          <li>Open the <b>Shortcuts</b> app → <b>+</b> to create a new one.</li>
          <li>Add action <b>Take Screenshot</b>.</li>
          <li>Add action <b>Get Contents of URL</b>. Set URL to your copied link, tap <b>Show More</b>: Method <b>POST</b>, Request Body <b>File</b>, and choose the <b>Screenshot</b> variable.</li>
          <li>Add action <b>Show Alert</b>. Set the message to the <b>Contents of URL</b> variable, and change the OK button text to <b>Open in No Clú</b>.</li>
          <li>Add action <b>Get URLs from Input</b> (input = <b>Contents of URL</b>), then <b>Get Last Item from List</b>, then <b>Open URLs</b>.</li>
          <li>Name it <b>No Clú</b>.</li>
        </ol>
      </div>
    </div>

    <div class="step">
      <div class="n">Step 3</div>
      <h2>Pick how to launch it</h2>
      <p>You only need <b>one</b> of these.</p>

      <div class="trigger">
        <div class="tname">🗣️ Just ask Siri <span class="tbadge best">no setup</span></div>
        <div class="tsteps">Say <b>“Hey Siri, No Clú”</b>. This works the moment the Shortcut is added — nothing else to configure.</div>
      </div>

      <div class="trigger">
        <div class="tname">👆 Back Tap — double-tap the back of your phone</div>
        <div class="tsteps">Open <b>Settings</b> → <b>Accessibility</b> → <b>Touch</b> → <b>Back Tap</b> → <b>Double Tap</b> → choose <b>No Clú</b>.
          <span class="note">Apple only lets this be set in Settings — no app can turn it on for you, which is why it's the longest of the four.</span></div>
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


HISTORY_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, maximum-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#0A0614">
<link rel="apple-touch-icon" href="/app-icon.png">
<link rel="icon" href="/app-icon.png">
<title>Your scan history · No Clú</title>
<style>
  /* Neon night — see APP_HTML for the full rationale. */
  :root{--bg:#0A0614;--ink:#F6E9FF;--muted:#B79DD0;--faint:#7B6B94;
        --pink:#FF2D95;--pink-soft:#FF9AC9;--cyan:#00E5FF;--cyan-soft:#9BE7FF;--violet:#7B2DFF;
        --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
        --sans:-apple-system,BlinkMacSystemFont,"SF Pro Text",system-ui,sans-serif}
  *{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
  body{background:var(--bg);color:var(--ink);font-family:var(--sans);min-height:100dvh;position:relative;
       padding:calc(env(safe-area-inset-top) + 18px) 22px calc(env(safe-area-inset-bottom) + 40px);
       -webkit-font-smoothing:antialiased;max-width:560px;margin:0 auto}
  body:before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
    background:
      radial-gradient(46% 26% at 16% 8%, rgba(255,45,149,.5), transparent 66%),
      radial-gradient(42% 24% at 88% 34%, rgba(0,229,255,.3), transparent 64%),
      radial-gradient(60% 32% at 44% 102%, rgba(123,45,255,.42), transparent 66%)}
  body>*{position:relative;z-index:1}
  a.back{font-family:var(--mono);font-size:12px;letter-spacing:1px;color:var(--cyan-soft);text-decoration:none}
  h1{font-size:24px;margin:18px 0 4px;text-shadow:0 0 20px rgba(255,45,149,.45)}
  .sub{color:var(--cyan-soft);font-family:var(--mono);font-size:11px;letter-spacing:1.5px;
       text-transform:uppercase;margin-bottom:18px;font-variant-numeric:tabular-nums}
  .row{display:flex;align-items:center;gap:13px;padding:11px 0;text-decoration:none;color:inherit;
       border-bottom:1px solid rgba(0,229,255,.16)}
  .row .th{width:42px;height:60px;border-radius:6px;object-fit:cover;background:rgba(123,45,255,.25);flex-shrink:0}
  .row .main{flex:1;min-width:0}
  .row .t{font-size:15px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .row .m{font-family:var(--mono);font-size:10.5px;color:var(--muted);margin-top:3px}
  .row .go{color:var(--cyan);font-size:18px;text-shadow:0 0 10px rgba(0,229,255,.8)}
  .empty{color:var(--faint);font-size:14px;text-align:center;padding:50px 0}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/ScrollTrigger.min.js"></script>
<script src="https://cdn.jsdelivr.net/gh/studio-freight/lenis@1.0.29/bundled/lenis.min.js"></script>
</head>
<body>
  <a class="back" href="/app">← Back to app</a>
  <h1>Your scan history</h1>
  <div class="sub" id="count"></div>
  <div id="list"></div>
<script>
  function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,function(c){
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
  function meta(s){
    var bits=[];
    if(s.type) bits.push(String(s.type).replace('_',' '));
    if(s.year) bits.push(s.year);
    if(s.season && s.episode) bits.push('S'+s.season+' · E'+s.episode);
    return bits.join('  ·  ');
  }
  async function load(){
    var r;
    try{ r=await fetch('/api/history?limit=100'); }catch(e){ return; }
    if(r.status===401){ location.href='/app'; return; }
    if(r.status!==200){
      document.getElementById('count').textContent='';
      document.getElementById('list').innerHTML='<div class="empty">Couldn\\'t load your history.</div>';
      return;
    }
    var d=await r.json(), list=d.scans||[];
    document.getElementById('count').textContent = !list.length ? '' :
      (list.length>=100 ? '100+ scans' : list.length+' scan'+(list.length===1?'':'s'));
    if(!list.length){
      document.getElementById('list').innerHTML='<div class="empty">No scans yet.</div>';
      return;
    }
    document.getElementById('list').innerHTML=list.map(function(s){
      var thumb=s.poster?('<img class="th" src="'+esc(s.poster)+'" onerror="this.outerHTML=\\'<div class=th></div>\\'">'):'<div class="th"></div>';
      return '<a class="row" href="/title/'+s.id+'">'+thumb+
             '<div class="main"><div class="t">'+esc(s.title)+'</div>'+
             '<div class="m">'+esc(meta(s))+'</div></div><div class="go">›</div></a>';
    }).join('');
  }
  load();
</script>
</body>
</html>"""


TITLE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, maximum-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#0A0614">
<link rel="apple-touch-icon" href="/app-icon.png">
<link rel="icon" href="/app-icon.png">
<title>No Clú</title>
<style>
  /* Neon night — see APP_HTML for the full rationale. */
  :root{--bg:#0A0614;--ink:#F6E9FF;--muted:#B79DD0;--faint:#7B6B94;
        --pink:#FF2D95;--pink-soft:#FF9AC9;--cyan:#00E5FF;--cyan-soft:#9BE7FF;--violet:#7B2DFF;
        --edge-cyan:rgba(0,229,255,.26);
        --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
        --sans:-apple-system,BlinkMacSystemFont,"SF Pro Text",system-ui,sans-serif}
  *{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
  body{background:var(--bg);color:var(--ink);font-family:var(--sans);min-height:100dvh;position:relative;
       padding:calc(env(safe-area-inset-top) + 18px) 22px calc(env(safe-area-inset-bottom) + 40px);
       -webkit-font-smoothing:antialiased;max-width:520px;margin:0 auto}
  body:before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
    background:
      radial-gradient(46% 26% at 16% 8%, rgba(255,45,149,.5), transparent 66%),
      radial-gradient(42% 24% at 88% 34%, rgba(0,229,255,.3), transparent 64%),
      radial-gradient(60% 32% at 44% 102%, rgba(123,45,255,.42), transparent 66%)}
  body>*{position:relative;z-index:1}
  a.back{font-family:var(--mono);font-size:12px;letter-spacing:1px;color:var(--cyan-soft);text-decoration:none}
  /* the cover art is the brightest thing on the page — give it a lit edge */
  .cover{width:100%;border-radius:18px;margin-top:16px;display:none;
         border:1px solid rgba(255,45,149,.4);box-shadow:0 18px 50px rgba(0,0,0,.6),0 0 38px rgba(255,45,149,.28)}
  h1{font-size:27px;line-height:1.2;margin-top:18px;text-shadow:0 0 24px rgba(255,45,149,.45)}
  .meta{font-family:var(--mono);font-size:11px;letter-spacing:1px;color:var(--cyan-soft);
        margin-top:8px;text-transform:uppercase;text-shadow:0 0 12px rgba(0,229,255,.5)}
  .detail{color:var(--muted);font-size:14.5px;line-height:1.55;margin-top:14px}
  /* An uncertain answer has to LOOK uncertain. Amber, not the pink or cyan the
     rest of the app uses, so it reads as a caution rather than as decoration. */
  .unsure{display:none;margin-top:14px;padding:12px 14px;border-radius:13px;
          background:rgba(255,176,46,.1);border:1px solid rgba(255,176,46,.42)}
  .unsure.show{display:block}
  .unsure b{display:block;font-family:var(--mono);font-size:10.5px;letter-spacing:1.5px;
            text-transform:uppercase;color:#FFC96B;margin-bottom:5px}
  .unsure span{font-size:13.5px;line-height:1.5;color:var(--ink)}
  .label{font-family:var(--mono);font-size:10.5px;letter-spacing:1.5px;color:var(--faint);
         text-transform:uppercase;margin-top:26px}
  .chips{display:flex;flex-wrap:wrap;gap:9px;margin-top:11px}
  .chips a{font-family:var(--mono);font-size:11.5px;letter-spacing:.5px;text-decoration:none;
           padding:10px 14px;border-radius:22px;border:1px solid var(--edge-cyan);
           color:#EAFBFF;background:rgba(0,229,255,.08);
           box-shadow:0 0 14px rgba(0,229,255,.22),inset 0 0 12px rgba(0,229,255,.1)}
  .grp{margin-top:16px}
  .glabel{font-family:var(--mono);font-size:10px;letter-spacing:1px;color:var(--muted);text-transform:uppercase}
  .grp .chips{margin-top:8px}
  .nowhere{color:var(--faint);font-size:13.5px;line-height:1.5;margin-top:18px}
  /* Primary action: one tap straight to the best place to watch. It is the only
     filled control in the whole app — everything else is an outlined tube, so
     this reads as the brightest sign on the street. */
  .primary-action{display:flex;align-items:center;justify-content:center;gap:8px;margin-top:22px;
       text-decoration:none;height:54px;border-radius:27px;color:#fff;
       background:linear-gradient(120deg,var(--pink),var(--violet));
       font-family:var(--mono);font-weight:700;font-size:13px;letter-spacing:1.5px;text-transform:uppercase;
       box-shadow:0 0 34px rgba(255,45,149,.6),0 8px 26px rgba(0,0,0,.5);transition:transform .15s}
  .primary-action:active{transform:scale(.97)}
  .cta{display:flex;align-items:center;justify-content:center;gap:8px;margin-top:16px;
       text-decoration:none;height:52px;border-radius:26px;background:rgba(255,45,149,.12);
       border:1.5px solid var(--pink);color:#FFE9F5;text-shadow:0 0 10px rgba(255,45,149,.9);
       box-shadow:0 0 20px rgba(255,45,149,.42),inset 0 0 18px rgba(255,45,149,.16);
       font-family:var(--mono);font-weight:700;font-size:12.5px;letter-spacing:1.5px;text-transform:uppercase}
  /* demoted when a primary action already exists above it */
  .cta.secondary{background:rgba(0,229,255,.07);border-color:var(--edge-cyan);color:var(--cyan-soft);
       text-shadow:none;box-shadow:none;height:46px;font-size:12px}
  .msg{color:var(--faint);font-size:14px;text-align:center;padding:60px 0}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/ScrollTrigger.min.js"></script>
<script src="https://cdn.jsdelivr.net/gh/studio-freight/lenis@1.0.29/bundled/lenis.min.js"></script>
</head>
<body>
  <a class="back" href="/history">← Back to history</a>
  <div id="msg" class="msg">Loading…</div>
  <div id="body" style="display:none">
    <img class="cover" id="cover" alt="">
    <h1 id="ttl"></h1>
    <div class="meta" id="meta"></div>
    <div class="unsure" id="unsure"><b>Best guess</b><span id="unsureWhy"></span></div>
    <div class="detail" id="detail"></div>
    <a class="primary-action" id="primaryAction" target="_blank" rel="noopener" style="display:none">
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M3 2l9 5-9 5V2z" fill="#FFFFFF"></path></svg>
      <span id="primaryLabel"></span>
    </a>
    <div class="label" id="label" style="display:none"></div>
    <div id="groups"></div>
    <div class="nowhere" id="nowhere" style="display:none"></div>
    <a class="cta" id="cta" target="_blank" rel="noopener" style="display:none"></a>
  </div>
<script>
  function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,function(c){
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
  // "IN" -> "India", so labels don't read "where to watch in IN".
  function countryName(cc){
    if(!cc) return '';
    try{ return new Intl.DisplayNames(['en'],{type:'region'}).of(cc) || cc; }
    catch(e){ return cc; }
  }
  async function load(){
    var id = location.pathname.split('/').pop();
    var r;
    try{ r=await fetch('/api/title/'+encodeURIComponent(id)); }
    catch(e){ document.getElementById('msg').textContent='Could not reach the server.'; return; }
    if(r.status===401){ location.href='/app'; return; }
    if(r.status!==200){ document.getElementById('msg').textContent='Scan not found.'; return; }
    var d=await r.json();
    document.title = d.title ? (d.title+' · No Clú') : 'No Clú';
    if(d.poster){
      var c=document.getElementById('cover');
      c.onerror=function(){ c.style.display='none'; };
      c.src=d.poster; c.style.display='block';
    }
    document.getElementById('ttl').textContent=d.title||'Unknown';
    var bits=[];
    if(d.type) bits.push(String(d.type).replace('_',' '));
    if(d.year) bits.push(d.year);
    if(d.season && d.episode) bits.push('S'+d.season+' · E'+d.episode);
    document.getElementById('meta').textContent=bits.join('  ·  ');
    // Show a guess AS a guess, and say what it rested on. Scans saved before
    // confidence was stored have none, and get no badge rather than a wrong one.
    // Medium counts as unsure: over 12 measured runs the only confidently-wrong
    // answer left was a "medium" one. Only "high" — a title read off the screen
    // or an unmistakable scene — goes unbadged.
    if(d.confidence === 'low' || d.confidence === 'medium'){
      document.getElementById('unsureWhy').textContent = d.evidence ||
        'The picture alone was not enough to be sure of the title.';
      document.getElementById('unsure').classList.add('show');
    }
    document.getElementById('detail').textContent=d.detail||'';
    // Where to watch, grouped so "already included in your subscription" is
    // never mistaken for "costs money to rent".
    // "in IN" reads like a typo — show the country's name when the browser knows it.
    var cc=countryName(d.country||'');
    var provs=d.providers||[], w=d.watch||{};

    // Reset every watch-related slot first, so re-rendering can never leave a
    // stale group list beside a contradictory "nothing listed" message.
    document.getElementById('groups').innerHTML='';
    document.getElementById('label').style.display='none';
    document.getElementById('nowhere').style.display='none';

    // One-tap primary action. The verb reflects how it's offered, so a rental
    // is never dressed up as something already included.
    var prim=d.primary, act=document.getElementById('primaryAction');
    if(prim && prim.url){
      var verb = prim.kind==='rent' ? 'Rent on' : prim.kind==='buy' ? 'Buy on' : 'Open in';
      act.href=prim.url;
      document.getElementById('primaryLabel').textContent=verb+' '+prim.name;
      act.style.display='flex';
    } else { act.style.display='none'; }

    function chips(list){
      return list.map(function(p){
        return '<a href="'+esc(p.url)+'" target="_blank" rel="noopener">'+esc(p.name)+'</a>';
      }).join('');
    }
    var groups=[
      ['stream', 'Included with subscription'],
      ['rent',   'Rent'],
      ['buy',    'Buy']
    ];
    var html='';
    groups.forEach(function(g){
      var list=w[g[0]]||[];
      if(list.length) html += '<div class="grp"><div class="glabel">'+g[1]+
                              '</div><div class="chips">'+chips(list)+'</div></div>';
    });
    if(html){
      document.getElementById('label').textContent='Where to watch in '+cc;
      document.getElementById('label').style.display='block';
      document.getElementById('groups').innerHTML=html;
    } else if(provs.length){
      // TMDB gave names but no category breakdown — show them ungrouped.
      document.getElementById('label').textContent='Where to watch in '+cc;
      document.getElementById('label').style.display='block';
      document.getElementById('groups').innerHTML='<div class="grp"><div class="chips">'+chips(provs)+'</div></div>';
    } else {
      document.getElementById('nowhere').textContent =
        'No streaming services listed for '+cc+' — try the full search below.';
      document.getElementById('nowhere').style.display='block';
    }
    if(d.justwatch){
      var cta=document.getElementById('cta');
      cta.href=d.justwatch;
      cta.classList.toggle('secondary', !!prim);
      cta.textContent = html||provs.length ? 'See all options & prices' : ('▶ Where to watch in '+cc);
      cta.style.display='flex';
    }
    document.getElementById('msg').style.display='none';
    document.getElementById('body').style.display='block';
  }
  load();
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


def _google_redirect_uri(request: Request) -> str:
    """Must match a URI registered in the Google Cloud console, exactly."""
    return f"{str(request.base_url).rstrip('/')}/auth/google/callback"


@app.get("/auth/google/start")
async def auth_google_start(request: Request):
    """Send the user to Google, carrying a signed one-time state."""
    if not GOOGLE_CLIENT_ID:
        # Not configured: behave as if the feature doesn't exist rather than
        # bouncing the user to a broken Google error page.
        return JSONResponse({"error": "google sign-in is not configured"}, status_code=404)

    state = auth.make_oauth_state()
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": _google_redirect_uri(request),
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    }
    redirect = RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}", status_code=302)
    # The same value goes in a cookie; the callback demands both and that they
    # match, so a state minted in someone else's browser is useless here.
    redirect.set_cookie(
        GOOGLE_STATE_COOKIE, state, max_age=auth.OAUTH_STATE_MAX_AGE,
        httponly=True, samesite="lax",
        secure=request.headers.get("x-forwarded-proto", request.url.scheme) == "https",
    )
    return redirect


@app.get("/auth/google/callback")
async def auth_google_callback(request: Request, code: Optional[str] = None,
                               state: Optional[str] = None, error: Optional[str] = None):
    """Exchange Google's code for the user's identity and sign them in."""
    if error or not code:
        return RedirectResponse("/app?signin=cancelled", status_code=302)
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
        return JSONResponse({"error": "google sign-in is not configured"}, status_code=404)

    cookie_state = request.cookies.get(GOOGLE_STATE_COOKIE, "")
    if (not state or not cookie_state or not secrets.compare_digest(state, cookie_state)
            or auth.read_oauth_state(state) is None):
        return JSONResponse({"error": "sign-in expired or invalid — please try again"},
                            status_code=400)

    try:
        async with httpx.AsyncClient(timeout=10.0) as http:
            token_res = await http.post(GOOGLE_TOKEN_URL, data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": _google_redirect_uri(request),
                "grant_type": "authorization_code",
            })
            if token_res.status_code != 200:
                return JSONResponse({"error": "Google sign-in failed — please try again"},
                                    status_code=400)
            access_token = token_res.json().get("access_token", "")
            info_res = await http.get(GOOGLE_USERINFO_URL,
                                      headers={"Authorization": f"Bearer {access_token}"})
            if info_res.status_code != 200:
                return JSONResponse({"error": "Google sign-in failed — please try again"},
                                    status_code=400)
            info = info_res.json()
    except Exception:
        return JSONResponse({"error": "Couldn't reach Google — please try again"},
                            status_code=502)

    google_id = info.get("sub")
    if not google_id:
        return JSONResponse({"error": "Google sign-in failed — please try again"},
                            status_code=400)
    # Only trust a verified address; an unverified one could belong to someone
    # else and would link this login onto their existing account.
    email = info.get("email") if info.get("email_verified") else None

    session = db.SessionLocal()
    try:
        user = db.link_or_create_google_user(
            session, google_id=google_id, email=email, display_name=info.get("name"))
        user_id = user.id
    finally:
        session.close()

    response = RedirectResponse("/app", status_code=302)
    _set_session_cookie(request, response, user_id)
    response.delete_cookie(GOOGLE_STATE_COOKIE)
    return response


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


HISTORY_LIMIT_DEFAULT = 30
HISTORY_LIMIT_MAX = 100


def _clamp_limit(value: Optional[int], default: int = HISTORY_LIMIT_DEFAULT) -> int:
    """A sane row count, whatever the query string contained."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(n, HISTORY_LIMIT_MAX))


def _scan_to_dict(scan) -> dict:
    """One stored scan as JSON. `id` is what makes a row tappable."""
    return {
        "id": scan.id,
        "title": scan.title,
        "type": scan.content_type,
        "year": scan.year,
        "season": scan.season,
        "episode": scan.episode,
        "poster": scan.poster,
        "detail": scan.detail,
        # Older rows predate these columns and read as None, which the UI
        # treats the same as "high" — no badge on scans we can't grade.
        "confidence": getattr(scan, "confidence", None),
        "evidence": getattr(scan, "evidence", None),
        "at": scan.scanned_at.isoformat() if scan.scanned_at else None,
    }


def _group_watch(watch: Optional[dict], title: str, country: str) -> dict:
    """Turn a raw TMDB availability dict into linked, categorised options.

    Shared by /identify and /api/title so the result card and the detail page
    can never disagree about where something can be watched.

    Returns:
      providers — flat, deduped across categories (a service that both streams
                  and sells a title appears once)
      watch     — the same entries kept in their stream/rent/buy buckets
      primary   — the one service worth a single tap. A subscription the user
                  likely already pays for beats a rental, which beats a purchase.
    """
    buckets = {"stream": [], "rent": [], "buy": []}
    providers = []
    seen = set()
    if watch:
        for key in ("stream", "rent", "buy"):
            for name in watch.get(key) or []:
                entry = {"name": name, "url": provider_link(name, title, country)}
                buckets[key].append(entry)
                if name not in seen:
                    seen.add(name)
                    providers.append(entry)
    primary = None
    for key in ("stream", "rent", "buy"):
        if buckets[key]:
            primary = dict(buckets[key][0])
            primary["kind"] = key  # drives the button's verb: open / rent / buy
            break
    return {"providers": providers, "watch": buckets, "primary": primary}


async def _none():
    """An awaitable that resolves to None — lets gather() keep a fixed shape
    when one of its branches is skipped."""
    return None


async def watch_options(content: ScreenContent, country: str) -> dict:
    """Where to watch, as tappable chips plus a JustWatch catch-all.

    Never raises: a detail page must still render its cover and description
    when availability lookup fails.
    """
    try:
        watch = await tmdb_where_to_watch(content, country)
    except Exception:
        watch = None
    try:
        out = _group_watch(watch, content.title, country)
    except Exception:
        out = _group_watch(None, content.title, country)
    try:
        out["justwatch"] = justwatch_url(content, country)
    except Exception:
        out["justwatch"] = None
    return out


@app.get("/api/history")
async def api_history(request: Request, limit: Optional[int] = None):
    uid = current_user_id(request)
    if uid is None:
        return JSONResponse({"error": "not signed in"}, status_code=401)
    session = db.SessionLocal()
    try:
        scans = db.recent_scans(session, uid, limit=_clamp_limit(limit))
        return {"scans": [_scan_to_dict(s) for s in scans]}
    finally:
        session.close()


@app.get("/api/title/{scan_id}")
async def api_title(request: Request, scan_id: int):
    uid = current_user_id(request)
    if uid is None:
        return JSONResponse({"error": "not signed in"}, status_code=401)
    session = db.SessionLocal()
    try:
        scan = db.get_scan_for_user(session, scan_id, uid)
        if scan is None:
            # 404 (not 403) so scan ids can't be probed for existence.
            return JSONResponse({"error": "not found"}, status_code=404)
        data = _scan_to_dict(scan)
    finally:
        session.close()

    country = await resolve_country(request, None)
    content = ScreenContent(
        content_type=data["type"] or "other", title=data["title"],
        year=data["year"], season=data["season"], episode=data["episode"],
        confidence="high", detail=data["detail"] or "",
    )
    data["country"] = country

    # Scans stored before a catalog could find them kept a NULL poster, and
    # nothing would ever have filled it in. Opening one is the natural moment
    # to try again, so old Indian titles pick up art instead of staying blank
    # forever. Runs alongside the watch lookup, not before it.
    need_poster = not data.get("poster")
    watch, late_poster = await asyncio.gather(
        watch_options(content, country),
        fetch_poster(content) if need_poster else _none(),
    )
    data.update(watch)
    if late_poster:
        data["poster"] = late_poster
        # Best-effort: a failed backfill must never cost the user the page.
        try:
            session = db.SessionLocal()
            try:
                db.set_scan_poster(session, scan_id, uid, late_poster)
            finally:
                session.close()
        except Exception:
            pass
    return data


@app.get("/app", response_class=HTMLResponse)
async def app_page():
    return (APP_HTML
            .replace("__GOOGLE_ENABLED__", "1" if GOOGLE_CLIENT_ID else "0")
            .replace("__APPLE_ENABLED__", "1" if APPLE_CLIENT_ID else "0"))


@app.get("/shortcut", response_class=HTMLResponse)
async def shortcut_page():
    """Setup sub-page: gives the signed-in user their personal scan link and,
    when ICLOUD_SHORTCUT_URL is configured, a one-tap install of the template."""
    return SHORTCUT_HTML.replace("__ICLOUD_URL__", ICLOUD_SHORTCUT_URL)


@app.get("/history", response_class=HTMLResponse)
async def history_page():
    return HISTORY_HTML


@app.get("/title/{scan_id}", response_class=HTMLResponse)
async def title_page(scan_id: int):
    return TITLE_HTML


@app.get("/app-icon.png")
async def app_icon():
    """Neon prism home-screen icon, generated so no asset file is needed.

    Matches the Neon night palette the rest of the app moved to: a hot-pink
    diamond on the same deep violet ground, so the icon on the home screen
    and the first screen it opens are the same object.
    """
    size = 180
    ground, neon = (10, 6, 20), (255, 45, 149)
    icon = Image.new("RGB", (size, size), ground)
    d = ImageDraw.Draw(icon)
    cx = cy = size // 2
    d.polygon([(cx, 34), (size - 34, cy), (cx, size - 34), (34, cy)],
              outline=neon, width=4)
    d.polygon([(cx, 74), (size - 74, cy), (cx, size - 74), (74, cy)],
              fill=neon)
    buf = io.BytesIO()
    icon.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/")
async def health():
    key_present = bool(GEMINI_API_KEY) if PROVIDER == "gemini" else bool(os.getenv("ANTHROPIC_API_KEY"))
    # Names only, never values: this is how a key can be confirmed as installed
    # without the key itself ever leaving the dashboard it was pasted into.
    return {"app": "No Clú", "provider": PROVIDER, "key_configured": key_present,
            "tmdb": bool(TMDB_API_KEY), "default_country": DEFAULT_COUNTRY,
            "extra_providers": sorted({p["service"] for p in _extra_providers()}),
            "provider_errors": dict(_provider_last_error)}


@app.post("/identify")
async def identify(request: Request, country: Optional[str] = None):
    started = time.monotonic()
    if daily_cap.exceeded:
        return {"identified": False,
                "summary": "🌙 No Clú's free daily limit is used up — try again after midnight!",
                "elapsed_seconds": round(time.monotonic() - started, 2)}

    # Start the geo lookup but don't wait on it: nothing before the answer needs
    # the country, so on a cache miss its round trip hides entirely behind the
    # AI call instead of being dead time at the front of the scan.
    country_task = asyncio.ensure_future(resolve_country(request, country))
    try:
        media = await _read_uploaded_media(request)
        frames = _encode_frames(media["images"])
        if not frames:
            raise ValueError("no readable frames")
        content = await identify_content(frames, media["audio"])
    except ProviderError as e:
        country_task.cancel()
        return {"identified": False, "summary": str(e),
                "elapsed_seconds": round(time.monotonic() - started, 2)}
    except Exception:
        country_task.cancel()
        return {"identified": False, "summary": "🔍 Couldn't read that screen — try again.",
                "elapsed_seconds": round(time.monotonic() - started, 2)}
    resolved_country = await country_task
    daily_cap.record()  # only count successful recognitions against the free quota

    # Independent lookups, so run them together — this used to be two serial
    # round trips on the critical path of every scan. Both swallow their own
    # errors and return None, so neither can break the result.
    watch, poster = await asyncio.gather(
        tmdb_where_to_watch(content, resolved_country),
        fetch_poster(content),
    )

    # "Where to watch in <country>" — build tappable per-platform links.
    # Same helper the detail page uses, so both surfaces agree. `justwatch`
    # always works (no key) and lists every platform in-region.
    grouped = _group_watch(watch, content.title, resolved_country)
    providers = grouped["providers"]
    jw = justwatch_url(content, resolved_country)

    # Save to the signed-in user's history so it syncs across their devices.
    # Never let a history-write failure break the recognition result.
    uid = current_user_id(request)
    if uid is not None:
        try:
            session = db.SessionLocal()
            try:
                db.add_scan(session, uid, title=content.title, content_type=content.content_type,
                            year=content.year, poster=poster, detail=content.detail,
                            season=content.season, episode=content.episode,
                            confidence=content.confidence, evidence=content.evidence)
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
        "evidence": content.evidence,
        "evidence_type": content.evidence_type,
        "source": content.source,
        "country": resolved_country,
        # NOTE: `watch` here is TMDB's RAW dict (plain provider-name strings),
        # which /demo renders. /api/title's `watch` is the grouped, linked shape
        # ({stream:[{name,url}]}). Different endpoints, different shapes — read
        # `providers` and `primary` below if you want links.
        "watch": watch,
        "providers": providers,
        "primary": grouped["primary"],
        "justwatch": jw,
        "poster": poster,
        "summary": build_summary(content, watch, resolved_country),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


async def _read_uploaded_media(request: Request) -> Dict[str, object]:
    """Collect every frame the Shortcut sent, plus any audio clip.

    Accepts all the shapes a Shortcut's "Get Contents of URL" can post:
    - Request Body = File -> raw bytes as the whole body (one frame, no audio)
    - Request Body = Form -> any number of file fields. A part's declared
      content type decides audio vs. frame when present and specific; the
      field name is only consulted as a fallback when the content type is
      missing or generic (e.g. application/octet-stream), so the Shortcut is
      easy to build without a strict field-naming contract. A part routed
      toward audio that turns out not to be usable audio (wrong type, too
      big, empty) is never dropped — it falls back to being a candidate
      frame instead, so readable input is never lost to misclassification.
    """
    images: List[bytes] = []
    audio: Optional[Dict[str, str]] = None

    ctype = request.headers.get("content-type", "")
    if "multipart/form-data" in ctype:
        form = await request.form()
        for name, value in form.multi_items():
            if not hasattr(value, "read"):
                continue
            data = await value.read()
            raw_type = (getattr(value, "content_type", "") or "").strip()
            base_type = raw_type.split(";")[0].strip().lower()
            if base_type and base_type != "application/octet-stream":
                looks_like_audio = base_type.startswith("audio/")
            else:
                looks_like_audio = "audio" in name.lower()

            used_as_audio = False
            if looks_like_audio and audio is None:
                clip = _usable_audio(data, raw_type)
                if clip is not None:
                    audio = clip
                    used_as_audio = True
            if not used_as_audio:
                images.append(data)
        return {"images": images, "audio": audio}

    body = await request.body()
    return {"images": [body] if body else [], "audio": None}


@app.post("/scan", response_class=PlainTextResponse)
async def scan_text(request: Request, country: Optional[str] = None,
                    token: Optional[str] = None, reply: Optional[str] = None):
    """Plain-text sibling of /identify for the iOS Shortcut.

    Returns a ready-to-show string (title + where-to-watch link) so the
    Shortcut needs no JSON parsing — just "Take Screenshot -> Get Contents of
    URL -> Show Notification". If a personal `token` is present, the scan is
    also saved to that account's history (that's how a back-tap scan lands in
    the app, since a Shortcut carries no login session).
    """
    if daily_cap.exceeded:
        return "🌙 No Clú's free daily limit is used up — try again after midnight!"

    media = await _read_uploaded_media(request)
    if not media["images"]:
        return ("🔍 No screenshot received. In the Shortcut's 'Get Contents of URL', "
                "set Method to POST and Request Body to File.")

    resolved_country = await resolve_country(request, country)
    try:
        frames = _encode_frames(media["images"])
        if not frames:
            raise ValueError("no readable frames")
        content = await identify_content(frames, media["audio"])
    except ProviderError as e:
        return str(e)
    except Exception:
        return "🔍 Couldn't read that screen — try again."
    daily_cap.record()  # only count successful recognitions against the free quota

    # Independent lookups, run together. These used to be sequential, and the
    # poster was fetched INSIDE the save block — so a back-tap scan waited for
    # two round trips one after the other on the slowest path in the product.
    watch, poster = await asyncio.gather(
        tmdb_where_to_watch(content, resolved_country),
        fetch_poster(content),
    )

    # Save to the account that owns this token, so back-tap scans sync to the app.
    # Best-effort: a history-write failure must never break the notification.
    saved_id = None
    if token:
        try:
            session = db.SessionLocal()
            try:
                user = db.get_user_by_scan_token(session, token)
                if user:
                    # confidence and evidence were being dropped here while
                    # /identify stored them, so the "Best guess" badge never
                    # appeared on back-tap scans — the ones most likely to be
                    # uncertain, and the only ones most users ever make.
                    saved = db.add_scan(session, user.id, title=content.title,
                                        content_type=content.content_type, year=content.year,
                                        poster=poster, detail=content.detail,
                                        season=content.season, episode=content.episode,
                                        confidence=content.confidence,
                                        evidence=content.evidence)
                    saved_id = saved.id
            finally:
                session.close()
        except Exception:
            pass

    lines = [build_summary(content, watch, resolved_country)]
    jw = justwatch_url(content, resolved_country)
    if jw:
        lines.append(f"▶ Where to watch in {resolved_country}: {jw}")
    # Last line, and deliberately the LAST url in the reply: the Shortcut grabs
    # the final URL and opens it, landing the user on this exact scan in the app.
    app_link = None
    if saved_id is not None:
        base = str(request.base_url).rstrip("/")
        app_link = f"{base}/title/{saved_id}"
        lines.append(f"{SCAN_APP_LINK_LABEL}: {app_link}")

    # reply=link returns the bare URL and nothing else, so a Shortcut can pipe
    # this straight into "Open URLs". The default reply is a human-readable
    # summary containing several URLs, which forces the Shortcut to add
    # "Get URLs from Input" and "Get Last Item" to dig the right one out —
    # three fragile actions whose chaining broke the moment one was edited.
    # One action that cannot be mis-wired beats three that can.
    if reply == "link":
        return app_link or ""
    return "\n".join(lines)
