# Poster art (no-signup sources) + free-tier cost guard

**Date:** 2026-07-22
**Status:** Approved by user, ready for implementation plan

## Why

Two problems surfaced in the same conversation:

1. The user wanted real poster art in No Clú's results, but couldn't complete
   TMDB's signup flow. Need a poster source that requires **zero account
   creation**, so this can never happen again.
2. The user wants to eventually share No Clú with other people. Doing so
   naively would mean everyone shares the owner's single free Gemini API key
   — one shared daily quota that, once exhausted, breaks the app for
   everyone, including the owner, with no warning.

Both are addressed here. Everything else the user mentioned in the same
conversation (Google/Apple sign-in, hosting for other people, per-user data,
and the longer-term ambition to grow this into a company) is **explicitly
out of scope** — see "Out of scope" below.

**Hard requirement carried through this whole spec:** the app must stay
**free for end users, permanently** — explicitly modeled on Shazam, which
has never charged the people using it or asked them to configure anything.
Nothing in this document proposes charging users or asking them to bring
their own API key. If the free Gemini quota is ever a real constraint at
scale, that's a cost the owner (or a future business model that doesn't
touch the end-user experience) needs to solve — not something passed on to
whoever's tapping the lens.

## Scope

**In scope:**
- Poster art lookup with no account/API key required, for any recognized
  movie, TV show, or anime.
- A daily usage cap on the shared free Gemini key, with a graceful,
  friendly message when it's hit — never a broken error.

**Out of scope (deferred, not designed here):**
- Google/Apple sign-in
- Hosting the app for other people (currently runs only on the owner's Mac)
- Multi-user data (per-user recent scans, accounts, etc.)
- How to fund/absorb API cost if the app ever outgrows the free daily cap
  — ruled out as a **user-facing subscription or "bring your own key"**
  requirement (see hard requirement above); whatever the eventual answer is,
  it isn't designed here
- Any business/legal work needed to operate this commercially (e.g. the
  copyright/ToS implications of recognizing content from paid streaming
  services as a product other people pay for) — flagged to the user as
  something to get real legal advice on before monetizing, not something
  addressed in this spec

These are independent pieces that each need their own brainstorm before
they're built; bundling them into this spec would make it too large to
implement or verify as one unit.

## Design: poster art

**Chain, cheapest/best-quality first:**

1. **iTunes Search API** (`https://itunes.apple.com/search`) — no key, no
   signup, ever. Query by title (+ year when available, mirroring the
   existing TMDB retry-without-year pattern). Request the largest artwork
   size available by upsizing the `artworkUrl100` field (replace the
   `100x100bb` suffix with a much larger box, e.g. `1200x1200bb`) rather
   than using the default thumbnail.
2. **Wikipedia page image fallback** — if iTunes returns no match (common
   for anime, regional, or obscure titles), query the MediaWiki API
   (`action=query&prop=pageimages`) for the title and use the page's lead
   image at a large thumbnail size (`pithumbsize=1000` or similar).
3. If both come back empty, `poster` is simply `null` — the result still
   shows title/detail/summary normally. A missing poster is never an error.

**Fail-soft requirement:** both lookups run with a short timeout inside a
try/except that returns `None` on any failure (matching the existing
`tmdb_where_to_watch` pattern in `main.py`). A poster-lookup failure must
never surface as an error to the user or block the core recognition result.

**Relationship to TMDB:** TMDB-based "where to watch" data is unchanged and
stays optional (gated on `TMDB_API_KEY` being set, as today). Poster art no
longer depends on TMDB at all.

## Design: free-tier cost guard

**Problem being solved:** the owner's Gemini API key has a daily free quota
shared across every request the server makes. If shared with other people
later, this quota is consumed by everyone together with no visibility, and
once exhausted, `/identify` starts failing for the rest of the day with a
raw provider error.

**Approach:** a simple in-process daily counter in `main.py`.

- Track a count of `/identify` calls for "today" (server local date).
- Reset the counter automatically when the date rolls over.
- Before calling the vision provider, check the count against a configurable
  ceiling, `DAILY_SCAN_CAP` (env var, **default 100/day**). This is a
  deliberately conservative starting point, not a measured value — Google
  does not publish a fixed public free-tier number; it's visible per account
  in the user's own AI Studio dashboard
  (https://aistudio.google.com/rate-limit). The user should check that page
  once and raise `DAILY_SCAN_CAP` if their real limit is comfortably higher.
- If the cap is reached, return the same shape as other graceful failures
  already in `main.py` (`identified: false`, a friendly `summary` message —
  e.g. "🌙 No Clú's free daily limit is used up — try again after
  midnight!") instead of calling the provider at all.

**Known limitation, accepted for now:** the counter is in-memory, so it
resets if the server process restarts. Acceptable for a single-process
personal/small-scale server; would need a persistent store (file or small
database) if this becomes a real concern later.

**Future direction (not designed now):** the app stays free for users no
matter what — that's non-negotiable, not a cost lever. If the free cap
becomes a real constraint at scale, the options are things like the owner
absorbing metered API cost directly, or a business model that funds usage
without charging the person tapping the lens (e.g. sponsorship, a
completely separate paid product tier that doesn't touch the free
recognition feature). Which of those, and how it would work, is deliberately
left undesigned until it's actually needed — the one constraint that's
already locked in is that the core "tap and identify" experience never
starts costing the user money or effort.

## Error handling summary

Both features follow the existing `ProviderError` / graceful-degradation
pattern already established in `main.py`: never raise an unhandled
exception to the client, always fall back to a usable (if less complete)
result, and always explain what happened in plain language via `summary`.

## Verification plan

- Poster lookup: test against a well-known movie (should hit iTunes), a
  popular anime (likely falls through to Wikipedia), and a nonsense title
  (should return `poster: null` with no error).
- Cost guard: set `DAILY_SCAN_CAP` to a small number (e.g. 2) locally,
  confirm the third call returns the friendly limit message instead of
  calling Gemini, and confirm the counter resets after simulating a day
  rollover.
