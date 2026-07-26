# No Clú 📱🔍

**Shazam, but for what's on your screen.** One tap on your iPhone's Action Button (or a double back-tap) → the current screen is captured, analyzed by a vision AI, matched against TMDB — and a notification tells you *what you're watching* and *where you can stream it in your country*. No manual screenshotting, no uploading, no opening an app. Runs **free** on Google Gemini out of the box (paid Claude is a one-word switch away).

```
Tap → capture → identify → "🎬 Interstellar (2014) — Streaming in IN on: Netflix, Prime Video"
                                              (~2–4 seconds end to end)
```

---

## 1. Get your API key (free, no credit card)

No Clú ships set to **Google Gemini's free tier** — no payment, no card. You only need **one** key to start:

| Key | Where | Used for |
|---|---|---|
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → sign in with your Google account → **Create API key** | The vision "brain" that looks at the frame and names the content. **Free. No card.** |
| `TMDB_API_KEY` *(optional)* | [themoviedb.org](https://www.themoviedb.org/signup) → free account → **Settings → API** → copy **API Key (v3 auth)** | Where-to-watch data per country. Skip it and you still get the title, just not the streaming providers. |

### Want to use paid Claude instead (sharper on obscure titles)?

You can switch anytime — it's one word. In `.env`, set `PROVIDER=anthropic` and fill in `ANTHROPIC_API_KEY` (from [console.anthropic.com](https://console.anthropic.com); needs ~$5 credit, ~1¢ per use). Nothing else changes. You can switch back to free Gemini just as easily.

## 2. Run the server (on this Mac)

```bash
# from your "No Clu" folder:
cd server
source .venv/bin/activate      # the environment is already set up for you
uvicorn main:app --host 0.0.0.0 --port 8000
```

(Your `.env` is already created with `PROVIDER=gemini` — you just need to paste your key into it, per step 1.)

Check it's alive: open `http://localhost:8000` — you should see `{"app": "No Clú", "provider": "gemini", "key_configured": true, ...}`. If `key_configured` is `false`, your key didn't get saved into `.env` — recheck step 1.

Find your Mac's local IP (your iPhone will call this while on the same Wi-Fi):

```bash
ipconfig getifaddr en0
```

Say it prints `192.168.1.42` — your endpoint is `http://192.168.1.42:8000/identify`.

## 3. Build the iPhone Shortcut (2 minutes)

Open the **Shortcuts** app → **+** new shortcut, then add these four actions in order:

1. **Take Screenshot**
   *(captures whatever is on screen the instant the shortcut runs)*

2. **Get Contents of URL**
   - URL: `http://192.168.1.42:8000/identify?country=IN` *(your Mac's IP; set `country` to your 2-letter code)*
   - Tap the arrow to expand → **Method: POST**
   - **Request Body: Form**
   - Add new field → type **File** → Key: `image` → Value: select the **Screenshot** variable

3. **Get Dictionary Value**
   - Get **Value** for key `summary` in **Contents of URL**

4. **Show Notification** *(or **Show Result**, or **Speak Text** for full Shazam vibes)*
   - Body: the **Dictionary Value** variable

Name it **No Clú** and pick your one-tap trigger:

- **Action Button** (iPhone 15 Pro and later): Settings → Action Button → Shortcut → No Clú
- **Back Tap** (any iPhone): Settings → Accessibility → Touch → Back Tap → Double Tap → No Clú
- **AssistiveTouch bubble**: Settings → Accessibility → Touch → AssistiveTouch → Single-Tap → No Clú

> ⚠️ Don't launch it from a home-screen icon — that would capture your home screen, not the video. The Action Button / Back Tap triggers fire *over* whatever app you're watching, which is exactly what makes this feel like Shazam.

The first run will ask permission to contact your server — tap **Always Allow**.

## 4. Use it

Watching anything — Netflix, YouTube, Prime, a live match, a reel — press the Action Button. ~2–4 seconds later a notification tells you the title and where it streams in your country.

---

## Using it outside your home Wi-Fi

The local setup only works while your phone can reach your Mac. Two options to make it work anywhere:

- **Tailscale (easiest, free, private):** install [Tailscale](https://tailscale.com) on the Mac and iPhone, then use the Mac's Tailscale IP in the Shortcut URL. Works from anywhere, nothing exposed to the internet.
- **Deploy the server** to [Render](https://render.com), [Railway](https://railway.app), or Fly.io (the `server/` folder deploys as-is with start command `uvicorn main:app --host 0.0.0.0 --port $PORT`), then point the Shortcut at the public URL. When deployed publicly, the server auto-detects the caller's country from their IP, so you can even drop the `?country=` parameter.

## Switching the AI brain

No Clú has two brains built in; pick one with the `PROVIDER` line in `.env`:

| `PROVIDER=` | Cost | Notes |
|---|---|---|
| `gemini` *(default)* | **Free**, no card | Google's free tier. Daily usage limits, plenty for personal use. |
| `anthropic` | Paid (~1¢/use) | Claude — sharper on obscure content. Needs `ANTHROPIC_API_KEY` + ~$5 credit. |

Switching is just: change the word, add the matching key, restart the server. Both keys can live in `.env` at once — only the `PROVIDER` line decides which is active.

## Speed tuning

- The server downscales frames before analysis and runs a single vision call — the round trip is dominated by the model call.
- On free Gemini, `gemini-2.0-flash` (the default) is already the fast one. On paid Claude, set `CLAUDE_MODEL=claude-haiku-4-5` for the quickest responses.

## API

`POST /identify?country=XX` — multipart form with an `image` file. Returns:

```json
{
  "identified": true,
  "type": "movie",
  "title": "Interstellar",
  "year": 2014,
  "confidence": "high",
  "detail": "The docking scene; TARS visible in frame.",
  "country": "IN",
  "watch": {"stream": ["Netflix"], "rent": ["Apple TV"], "buy": [], "link": "https://..."},
  "summary": "🎬 Interstellar (2014)\nStreaming in IN on: Netflix",
  "elapsed_seconds": 2.4
}
```

Handles movies, TV shows, anime (with season/episode when visible), YouTube videos, live sports, music videos, and games.
