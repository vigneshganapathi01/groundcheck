# GroundCheck — hallucination detection dashboard

A local web UI around [LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect).
Paste source context + an LLM answer, and it highlights the parts of the answer
the context doesn't support.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r app/requirements.txt
.venv/bin/python app/server.py
```

Open http://127.0.0.1:8765. The first check downloads the model (~600 MB for base).

## How detection works (and what was changed from LettuceDetect's defaults)

- **Per-sentence scoring (default).** Each sentence of the AI response is checked against the
  context on its own. In a long answer, correct sentences dilute the signal on a wrong one.
  On the travel-policy test, "within 30 calendar days" (policy: 15) scored 0.33 inside the
  full answer and 0.64 on its own. On this project's 11-case test suite, per-sentence scoring
  caught 10 of 13 planted errors against 8 of 13 for the whole-answer pass, with 0 false alarms
  in both. Switch modes in **Models → How to score the AI response**.
- **Unreadable input is refused.** If a passage or the response is binary garbage (e.g. a
  .docx read as plain text), `/api/detect` returns an error instead of meaningless results,
  and the page marks the passage in red.
- **Strictness 0.10–0.95.** The Analysis panel shows how many spans each level (0.30/0.50/0.70/0.80)
  would flag; click a row to apply it.
- **Known limit:** contradictions made of words that *are* in the context can still slip
  through (e.g. Saturday's hours given as Sunday's).

## Uploading documents

- **Source context → Upload** (or drag files onto the section) accepts PDF, DOCX, TXT and MD.
  The server extracts the text (`POST /api/extract`), splits it into passages of about 1,200
  characters labelled by page or part, and groups each file into one collapsible card.
- **AI response → upload icon** loads the response to check from a file.
- When both the context and the response are present, the check runs automatically.
- DOCX tables are rewritten as sentences ("The price is $79."), because the detector
  wrongly flags values it reads from `Price | $79`-style rows.
- Long documents are split into chunks that fit the model. A word counts as supported if
  *any* chunk supports it. (LettuceDetect's default, `max()` across chunks, flags true facts
  that appear in only one chunk.)
- Limits: 20 MB per file. Scanned PDFs (images with no selectable text) need OCR first, and
  old `.doc` files must be saved as `.docx`.

## What's where

- `server.py` — FastAPI backend. `POST /api/detect` runs the model and returns per-token
  probabilities, character offsets and spans; `GET /api/models` lists models.
- `static/index.html` — the dashboard (no build step). The strictness slider re-derives
  spans in the browser from the token probabilities, so it updates without re-running the model.

## API example

```bash
curl -s localhost:8765/api/detect -H 'Content-Type: application/json' -d '{
  "context": ["The population of France is 67 million."],
  "question": "What is the population of France?",
  "answer": "The population of France is 69 million.",
  "model": "base", "threshold": 0.5
}'
```
