# Strauss Car Advisor

A Hebrew conversational advisor that helps an employee understand how switching their company car —
or waiving it — changes their monthly **net pay**. The employee describes their situation in free
text; the system checks eligibility, collects the missing details, and shows a net-pay comparison
based on Israeli Tax Authority data and the salary rules in the rules file.

> **Core principle:** every number and every decision comes from the deterministic tools (the
> calculation engine), never from the model. The model only converses and picks tools. In addition,
> every employee fact must appear in the employee's **own** messages ("grounding").

---

## Key features

- **Hebrew conversation (RTL)** with car recognition in Hebrew and English, eligibility checks, and
  collection of the details needed to calculate.
- **Net-pay comparison** between the current situation and an alternative (another car or a waiver),
  including the grossed-up electricity reimbursement for an electric car.
- **Local conversation history** — stored only in the browser (localStorage), surviving reloads and
  backend restarts, with no database.
- **Optional document upload** — a payslip or contract; the system extracts values, the employee
  confirms them, and only then are they sent as a normal chat message (so grounding is never bypassed).
- **Privacy:** the backend writes nothing to disk. An uploaded document lives only for the duration
  of the request and is never stored or logged.

---

## Architecture

```
strauss-car-advisor/
├─ backend/            FastAPI + the calculation engine and agent (Python)
│  ├─ app/
│  │  ├─ api/          HTTP layer (main.py, schemas.py, sessions.py, mapping.py)
│  │  ├─ agent/        agent loop, grounding, snapshot, document reader, Gemini provider
│  │  └─ ...           salary engine, catalogue, car search, rules
│  ├─ data/            Tax Authority file (CSV) and the challenge workbook (xlsx)
│  ├─ config/          rules.json and config files
│  ├─ tests/           pytest (no network, no Gemini quota — fake providers)
│  └─ requirements.txt
└─ frontend/           React + TypeScript + Vite (client)
   └─ src/
      ├─ api/          client.ts, types.ts (mirrors schemas.py — change them together)
      ├─ components/   conversation panel, decision panel, history drawer
      ├─ hooks/        useAdvisor.ts (conversation and state management)
      └─ lib/          styling, labels, formatting, history store
```

- **Turn flow:** `POST /api/chat` → `run_turn` (the agent) → tools → engine → result mapped to the
  public contract.
- **No internals leak:** responses carry no tool/provider/debug data, and logs carry no values,
  filenames, or tokens.

---

## Prerequisites

- **Python 3.12+**
- **Node.js 18+** (for Vite)
- A **Gemini API** key (Google AI Studio)

---

## Setup and run

### Backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate        # Windows PowerShell
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Create `backend/.env` (see `backend/.env.example`) and run:

```bash
uvicorn app.api.main:app --reload
```

The server starts on `http://127.0.0.1:8000`. Health check: `GET /api/health`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server starts on `http://localhost:5173` (an allowed CORS origin) and calls the backend API.

---

## Environment variables (`backend/.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | ✅ | — | Your Gemini API key. Keep it only in `.env` — never in `.env.example`, never in a commit. |
| `GEMINI_MODEL` | — | `gemini-flash-latest` | A model name your key is allowed to call. See the note below. |
| `GEMINI_THINKING_LEVEL` | — | `low` | `minimal` / `low` / `medium` / `high` / `default`. Some models do not support a thinking level — use `default` for those. |
| `GEMINI_TIMEOUT_SECONDS` | — | `60` | Per-request timeout. |
| `HISTORY_SIGNING_KEY` | — | ephemeral | Signs (HMAC) the history resume tokens. Without a stable value, history will not survive a restart. |

**Choosing a model:** if chat fails immediately, the model is likely unavailable to your key, or it
does not support the requested thinking level. List the models available to your key:

```bash
python -c "from google import genai,os;[print(m.name) for m in genai.Client(api_key=os.environ['GEMINI_API_KEY']).models.list() if 'generateContent' in (m.supported_actions or [])]"
```

A combination verified to work with a thinking level: `GEMINI_MODEL=gemini-flash-lite-latest` with
`GEMINI_THINKING_LEVEL=low`.

---

## Tests

**Backend** (no network, no Gemini quota — tests use fake providers):

```bash
cd backend
.\.venv\Scripts\python.exe -m pytest -q
```

**Frontend** (Vitest):

```bash
cd frontend
npm test
npm run typecheck
npm run lint
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check. |
| `POST` | `/api/sessions` | Start a new conversation (returns `session_id`). |
| `DELETE` | `/api/sessions/{id}` | Delete a conversation (best-effort). |
| `POST` | `/api/chat` | One conversation turn: a message in, a reply + screen state out. |
| `POST` | `/api/sessions/restore` | Restore a conversation from a history token (a new server session). |
| `POST` | `/api/sessions/{id}/document` | Upload a document to extract values for confirmation (changes no state, not a turn). |

---

## Privacy and security

- The backend **writes nothing to disk**: conversations live in memory only, and history is stored
  only in the user's browser.
- An uploaded document lives only for the duration of the request — never stored, never part of a
  resume token or history, never logged.
- Responses and logs are free of values, filenames, document content, keys, and tokens.
- An uploaded document is treated as **untrusted data**: instructions printed inside it are ignored.

> ⚠️ **Never commit `backend/.env`.** If an API key is ever exposed (for example, written into
> `.env.example` by mistake), rotate it immediately in Google AI Studio.

---

## License

A challenge/demo project. The mock data in the rules file is open; in a pilot you replace the data
file — not the code.
<img width="1273" height="582" alt="image" src="https://github.com/user-attachments/assets/4c95fde8-419f-481e-85e8-a94cc7d1d6b7" />

