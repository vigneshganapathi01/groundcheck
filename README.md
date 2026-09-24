# GroundCheck

**Span-level hallucination detection for AI responses, with a web dashboard.**

**▶ Live demo: https://groundcheck-iota.vercel.app**

The first check after the site has been idle takes a few extra seconds while the model loads.

Paste (or upload) the source documents and the AI's response. GroundCheck highlights the parts
of the response the sources don't support, scores every token, and shows how the result changes
with strictness.

![GroundCheck dashboard](app/screenshot.png)

Built on [LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) (MIT) by KR Labs, a ModernBERT
token classifier fine-tuned on RAGTruth. No API keys needed. Run it locally, or use the live demo.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r app/requirements.txt
.venv/bin/python app/server.py
```

Open http://127.0.0.1:8765. The first check downloads the detection model (~600 MB).

## Features

- Highlighted unsupported spans, a per-token risk chart and heatmap, and a JSON report export
- Upload PDF, DOCX, TXT or MD files as source context (or as the response to check)
- Per-sentence scoring, which catches contradictions a whole-answer pass misses
- A strictness trade-off table (0.30 / 0.50 / 0.70 / 0.80)
- Saved checks with search, pinning and examples; light and dark themes

See [app/README.md](app/README.md) for how detection works, the API, and known limits.

## Repository layout

| Path | What it is |
|---|---|
| `app/server.py` | FastAPI backend: detection, file extraction |
| `app/engine.py` | Scoring engine; runs the model with PyTorch (local) or ONNX Runtime (Vercel) |
| `app/static/index.html` | The dashboard (single file, no build step) |
| `tests/regression_suite.py` | Detection-quality regression tests (runs against any GroundCheck URL) |
| `verify_lettucedetect.py` | Standalone sanity checks for the underlying model |
| `requirements.txt`, `vercel.json` | Vercel deployment (ONNX backend, no PyTorch) |
| `Dockerfile` | Container image for Docker hosts (e.g. Hugging Face Spaces) |

## Deployment

The live demo runs on **Vercel's free tier**. PyTorch (~580 MB) exceeds Vercel's 250 MB
function limit, so the deployment uses an [ONNX export of the model](https://huggingface.co/maddyvicky001/lettucedetect-base-modernbert-onnx)
with ONNX Runtime instead:

- Weights are stored as fp16 and computed in fp32: 300 MB, and within 0.0013 of the PyTorch
  model's probabilities. (8-bit quantization was tested and rejected: errors up to 0.95.)
- `app/engine.py` re-implements LettuceDetect's prompt, tokenisation and chunking. Its output is
  identical to LettuceDetect's with the torch backend.
- The regression suite gives the same result locally and on the live site: 11 of 13 planted
  errors caught, 0 false alarms.

To deploy your own copy: `npx vercel deploy --prod`. Vercel detects the FastAPI app in
`app/server.py`, which switches to the ONNX backend when it runs on Vercel.
Limits of the hosted version: base model only, and very long documents can hit the time limit.
