# LAWoud

AI-powered legal assistance assistant for **Indian law** — a conversational
chatbot that answers questions from a curated legal knowledge base, asks
clarifying questions when a situation is ambiguous, and can hand off to a
real advocate / legal-aid lookup when professional help is warranted.

Hackathon build: runs entirely on `localhost`. No auth, no database, no
deployment tooling.

## What it does

- **Conversational intake** — for a real incident ("I got arrested"), the
  assistant asks one plain-language clarifying question per turn (up to 3)
  before answering; general questions ("what does Article 21 say?") are
  answered immediately.
- **Retrieval-gated answers** — the model answers *only* from retrieved
  context: a local Markdown corpus first, whitelisted official web sources
  (Tavily) as fallback. It never answers from unsupported model knowledge.
- **Hybrid local search** — a weighted keyword scorer blended with local
  sentence-transformers embeddings (`all-MiniLM-L6-v2`, cached to disk per
  corpus version). Falls back to keyword-only if disabled or unavailable.
- **Route-aware answers** — questions are routed `legal` / `moral` / `mixed`,
  so "my wife filed for divorce but I want to stay with her" leads with
  mediation options before the legal position.
- **Legal-assistance hand-off** — when professional help is warranted, the
  assistant asks for a district/state inline and returns relevance-ranked
  advocates mined from publicly indexed judicial records, a curated local
  directory, or national legal-aid contacts — never fabricated.
- **SSE streaming** — the answer streams token-by-token with status, source
  citations, and structured assistance cards.

## Repo layout

```
Backend/    FastAPI app — chat pipeline, corpus retrieval, AI providers,
            advocate/legal-aid lookup. See Backend/README.md for the full
            API contract and architecture notes.
Frontend/   Zero-dependency HTML/CSS/JS SPA — talks to the backend over SSE.
```

## Quickstart

```bash
cd Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY / GROQ_API_KEY / TAVILY_API_KEY
python run.py          # http://127.0.0.1:8000 — docs at /docs
```

Then serve the frontend and open it:

```bash
cd Frontend
python -m http.server 5173   # http://localhost:5173
```

API keys: [Gemini](https://aistudio.google.com/apikey) (primary),
[Groq](https://console.groq.com/keys) (automatic fallback),
[Tavily](https://app.tavily.com) (whitelisted web-search fallback and live
advocate search). The server starts without them, but `/api/chat` needs at
least one AI provider.

**Verify setup:** `python scripts/check_setup.py`
**Try it in a terminal:** `python scripts/terminal_chat.py`

## Knowledge corpus

Four Markdown files searched as one corpus, all hot-reloading on change:

| File | Contents |
|---|---|
| `Backend/General provisions all.md` | Constitution of India, parsed per-Article (~470 sections; repealed Articles excluded from retrieval) |
| `Backend/data/legal_knowledge.md` | General procedures — FIR, RTI, legal aid eligibility, cybercrime |
| `Backend/data/common_offenses.md` | Everyday offences with IPC + BNS (2023) references — cheque bounce (S.138 NI Act), dowry, theft, assault, criminal intimidation, defamation, cheating |
| `Backend/data/tenancy_and_consumer.md` | Rent deposits, agreements, eviction, repairs + the full consumer-complaint procedure (e-Daakhil, jurisdiction tiers, reliefs) |

## Disclaimer

LAWoud provides general legal **information**, not legal advice. It never
ranks a "best" lawyer and never guarantees an outcome.
