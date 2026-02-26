# EPUB Translator (LM Studio Only)

Self-hosted FastAPI web app that translates EPUB books with a local LLM via LM Studio while preserving EPUB structure, HTML/XHTML tags, and CSS references.

## Architecture Diagram

![Architecture Diagram](images/architecture.png)

## What This Project Does

- Upload an `.epub` from the browser
- Extract EPUB contents and resolve chapter order via OPF spine
- Parse chapter files and translate only text nodes
- Keep markup structure intact (tags/classes/styles/links)
- Rebuild and return a valid translated EPUB

## Core Translation Logic

### 1) EPUB Parsing and Chapter Order

The backend extracts the EPUB archive and reads:

- `META-INF/container.xml`
- `content.opf`

It uses OPF `manifest` + `spine` to process chapter files in reading order.

### 2) Safe Text-Only Mutation

For each chapter:

- If `.xhtml` (or XML prolog present): parse with strict XML parser (`lxml`) for Apple Books compatibility
- Else parse with HTML parser (`BeautifulSoup`)
- Collect translatable text nodes (`text` and `tail`) while skipping unsafe tags:
  - `script`, `style`, `noscript`, `svg`, `math`, `code`, `pre`

Only the node text is replaced. Structure is preserved.

### 3) Chunking

Text nodes are grouped into batches based on two limits:

- `TRANSLATION_CHUNK_MAX_CHARS`
- `TRANSLATION_CHUNK_MAX_ITEMS`

This balances quality, speed, and model stability.

### 4) LM Studio Translation Protocol (No JSON dependency)

The app sends plain text segments joined by a fixed separator token.
The model must return translated segments in the same order with the same separator.

This avoids fragile JSON parsing issues common in smaller local models.

### 5) Fallback and Recovery

If model output shape mismatches expected segments:

- Lenient alignment is applied
- Missing segments fall back to source text
- If needed, the app retries single-item translation for problematic entries

Detailed fallback logs are printed to container console when enabled.

### 6) Repackaging

The EPUB is re-zipped with correct EPUB requirements:

- `mimetype` first and uncompressed
- all other files deflated

## Features

- Mobile-friendly web UI
- Start/end chapter range selection
- Test mode (first N chapters of selected range)
- Pause/resume long jobs
- Persistent checkpoints in `work/<job_id>/job.json`
- Benchmark stats in UI (duration, chapters, chars/sec)

## Requirements

- Docker
- LM Studio running with local server enabled (OpenAI-compatible API)

## LM Studio Setup

### Required LM Studio server settings

- Start LM Studio local server
- Ensure API endpoint is reachable from Docker
- Load the model you specify in env

Typical Docker-on-mac base URL:

- `http://host.docker.internal:1234/v1`

## Environment Variables

Copy `.env.example` to `.env` and edit values:

- `LMSTUDIO_BASE_URL`:
  LM Studio API base URL (must include `/v1`)
- `LMSTUDIO_MODEL`:
  exact loaded model id/name in LM Studio
- `TRANSLATION_TEMPERATURE`:
  usually `0.0` or `0.1` for stable translation
- `TRANSLATION_CHUNK_MAX_CHARS`:
  max summed source characters per batch
- `TRANSLATION_CHUNK_MAX_ITEMS`:
  max text segments per batch
- `TRANSLATION_DEBUG_FALLBACK`:
  print fallback payload/response logs
- `TRANSLATION_TEST_MODE`:
  default test mode state
- `TRANSLATION_TEST_CHAPTER_LIMIT`:
  default chapter limit when test mode is on

## Recommended Starting Configuration

For local stability/speed with smaller models:

```env
TRANSLATION_TEMPERATURE=0.1
TRANSLATION_CHUNK_MAX_CHARS=12000
TRANSLATION_CHUNK_MAX_ITEMS=200
TRANSLATION_DEBUG_FALLBACK=true
```

If fallback logs are frequent, reduce chunk sizes:

```env
TRANSLATION_CHUNK_MAX_CHARS=7000
TRANSLATION_CHUNK_MAX_ITEMS=100
```

## Model Recommendation

### Practical baseline for Apple Silicon

- `Qwen3 4B` is a good speed/quality compromise for long books
- On a Mac M1 with 32GB RAM, an observed run was:
  - full book translation in ~5000 seconds
  - roughly < 1 minute per chapter average

This is a strong baseline for iterative runs.

If you want better quality consistency, test larger instruct models, but expect slower throughput.

## Run

```bash
docker build -t epub-tradutor .
docker run --rm -p 8000:8000 --env-file .env epub-tradutor
```

Open:

- `http://localhost:8000`

## Known Limitations

- Segment-separator protocol:
  If the model drops or rewrites separator tokens, alignment can degrade and fallback logic is used.
- Fallback substitutions:
  When output shape mismatches expected segments, missing parts may reuse source text to keep the pipeline moving.
- Translation quality variance:
  Smaller local models can paraphrase aggressively, merge/split sentences, or drift style across long books.
- No semantic memory layer:
  There is no persistent glossary/character memory by design in current version, so naming consistency relies on model behavior.
- Resume granularity:
  Resume checkpoints are chapter-cursor based; progress inside a partially translated chunk is not replayed exactly.
- Runtime storage:
  Job state and extracted EPUB working files are stored under `work/`; cleanup policy is manual.
- Language-specific punctuation quirks:
  Some models mishandle nested quotes/apostrophes in plain-text mode and trigger single-item fallback.

## Troubleshooting

### Repeated fallback logs

- Lower chunk settings
- Lower temperature
- Ensure model has enough context and is fully loaded

### EPUB opens with reader errors

- Confirm you are running latest build
- Verify strict XHTML path is active (XML parser)
- Re-run in test mode and validate output first

### Resume after restart

- Jobs in `running/queued` state are marked `paused` on startup
- Use Resume in UI to continue from checkpoint

## Project Layout

- `app.py` — FastAPI backend + translation pipeline
- `frontend/index.html` — single-page UI
- `Dockerfile` — container runtime
- `requirements.txt` — Python dependencies
- `.env.example` — environment template

## Credits

This project was developed with support from OpenAI Codex, including implementation, refactoring, and debugging workflows during iterative local-model integration.
