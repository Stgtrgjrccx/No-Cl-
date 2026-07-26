# Public hosting + accounts + cross-device synced history

**Date:** 2026-07-25
**Status:** Approved in principle by user ("all three free — go ahead"), design pending review

## Why

The user wants No Clú usable by other people, with **each person's scan
history accessible from any device, anywhere**. That requires moving off the
owner's Mac (public hosting), giving people accounts (sign-in), and storing
history server-side keyed to each account (a database). The user's driving
goal is the cross-device history sync; sign-in is the mechanism that enables it.

## Hard requirements (carried from earlier decisions)

- **Free for end users, permanently.** No subscriptions, no paywalls, no
  "bring your own key," no per-user cost. Any infrastructure cost is the
  owner's to absorb. (Modeled on Shazam.)
- **No paid SMS.** Phone OTP (text-message codes) is explicitly excluded
  because every SMS costs money and India requires DLT/TRAI registration —
  both conflict with the free-forever rule. Phone sign-in, if offered, uses a
  password, not an OTP.
- **Future-proof for a native iOS app.** All state (recognition API + auth +
  history) lives in the hosted backend, so a future native app is just
  another client of the same API and same logins — no rebuild.

## Scope

**In scope:**
- Host the FastAPI server publicly on a free tier (public `https://` URL).
- A persistent database storing users and their scan history.
- Three free sign-in methods: **Google (OAuth)**, **email + password**,
  **phone number + password**.
- Sync: a signed-in user's scans are saved server-side and shown on any
  device they log in from.
- A sign-in screen shown on first open of the app.

**Out of scope (deferred):**
- SMS OTP verification (paid; excluded by the hard requirement above).
- Apple and Facebook sign-in (Apple needs the $99/yr developer account;
  Facebook needs business verification + app review). Both slot in later
  behind the same account system with no rebuild.
- A native iOS app (Stage D — needs the paid Apple account).
- Real legal/ToS review before commercial scaling.

## Free infrastructure stack

| Piece | Service | Cost | Why |
|---|---|---|---|
| Code hosting | GitHub | Free | Source of truth; Render deploys from it |
| Web server | Render (Web Service) | Free, no card | Runs FastAPI publicly 24/7 |
| Database | Neon (Postgres) | Free, no card | Persistent user + history storage |
| Google sign-in | Google Cloud OAuth | Free | The "Sign in with Google" button |

**Known free-tier limitations (accepted for launch):**
- Render free web services **spin down after ~15 min idle**; the next request
  cold-starts in ~30–60s. Acceptable for early users; revisit with a paid
  dyno or a keep-warm ping later.
- The shared Gemini free quota is protected by the existing `DAILY_SCAN_CAP`.
- Uploaded screenshots transit a public server — add a plain-language privacy
  note; do not store raw images (only the identification result + metadata).

## Architecture

The existing `server/main.py` FastAPI app stays the single backend. Added
concerns are split into focused modules so `main.py` doesn't balloon:

- `server/db.py` — database connection + schema (users, scans). Uses a
  `DATABASE_URL` env var; falls back to local SQLite when unset so the app
  still runs on the owner's Mac for development.
- `server/auth.py` — sign-in logic: Google OAuth flow, email/phone + password
  (hashing via `passlib`/bcrypt), session issuance (signed cookie / token).
- `server/main.py` — wires routes: existing `/identify` now records a scan
  for the signed-in user; new `/auth/*` routes; `/history`; the sign-in UI is
  served before the app UI for signed-out users.

**Data model (minimal):**
- `users`: `id`, `google_id?`, `email?`, `phone?`, `password_hash?`,
  `created_at`. A user has at least one identifier; login matches any.
- `scans`: `id`, `user_id`, `title`, `type`, `year`, `poster`, `detail`,
  `scanned_at`. Replaces the current per-device localStorage history.

**Auth flows:**
- **Google:** standard OAuth 2.0 authorization-code flow; redirect URI is the
  hosted `https://` callback (this is *why* hosting must come first — Google
  rejects LAN/`http` redirect URIs).
- **Email/phone + password:** register (hash password), log in (verify hash),
  issue session. Email supports a reset link later (free email tier);
  phone-only has no free reset path — documented as a known limitation, with
  a prompt to also add an email for recovery.

## Build order (each step independently testable)

1. **Prepare the repo for deployment** (Procfile/start command, `DATABASE_URL`
   handling, `PORT` binding) — no external accounts needed; verify locally.
2. **Host on Render + Neon** — get the public URL working with a real
   database (owner creates the free accounts; guided).
3. **Accounts + password sign-in** — DB schema, register/login, session,
   sign-in UI. Fully testable before Google is wired.
4. **Google sign-in** — OAuth on top of the account system (owner creates the
   free Google Cloud OAuth credentials; guided).
5. **Synced history** — `/identify` writes a scan row; the app's "Recent
   scans" reads from `/history` instead of localStorage.

Steps 1 and 3 are buildable now without any of the owner's external accounts;
steps 2 and 4 gate on account creation the owner must do (guided, like the API
keys).

## Verification plan

- Register with email+password, add a scan, log out, log in on a "different
  device" (separate browser/incognito) → the scan history appears. Same for a
  Google account and a phone+password account.
- Signed-out users see the sign-in screen, not the scanner.
- `DAILY_SCAN_CAP` still returns the friendly limit message when exceeded.
- Poster/where-to-watch behavior unchanged from the prior spec.
