import asyncio
import json
import os
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from bs4 import BeautifulSoup
from bs4.element import NavigableString
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from lxml import etree
from pydantic import BaseModel

load_dotenv()

APP_ROOT = Path(__file__).resolve().parent
FRONTEND_FILE = APP_ROOT / "frontend" / "index.html"
WORK_ROOT = APP_ROOT / "work"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

SKIP_PARENT_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "math",
    "code",
    "pre",
}


def _parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "on"}


def _parse_int(value: str | int | None, default: int) -> int:
    if isinstance(value, int):
        return value
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except Exception:
        return default


CHUNK_MAX_CHARS = max(500, _parse_int(os.getenv("TRANSLATION_CHUNK_MAX_CHARS", "12000"), 12000))
CHUNK_MAX_ITEMS = max(1, _parse_int(os.getenv("TRANSLATION_CHUNK_MAX_ITEMS", "200"), 200))
DEBUG_FALLBACK_LOGS = _parse_bool(os.getenv("TRANSLATION_DEBUG_FALLBACK", "true"), default=True)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _sanitize_xml_text(text: str) -> str:
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return cleaned.replace("\ufffe", "").replace("\uffff", "")


def _split_edge_whitespace(text: str) -> tuple[str, str, str]:
    leading_match = re.match(r"^\s*", text)
    trailing_match = re.search(r"\s*$", text)
    leading = leading_match.group(0) if leading_match else ""
    trailing = trailing_match.group(0) if trailing_match else ""
    core = text[len(leading) : len(text) - len(trailing)] if text else ""
    return leading, core, trailing


def _looks_translatable(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if re.fullmatch(r"[\W_\d]+", stripped):
        return False
    return True


def _resolve_spine_html_files(extract_dir: Path) -> list[Path]:
    # Resolve reading order from OPF spine; fallback to lexical ordering when metadata is incomplete.
    container_xml = extract_dir / "META-INF" / "container.xml"
    if not container_xml.exists():
        return sorted(
            [p for p in extract_dir.rglob("*") if p.suffix.lower() in {".html", ".htm", ".xhtml"}]
        )

    container_xml_text = container_xml.read_text(encoding="utf-8")
    try:
        container_soup = BeautifulSoup(container_xml_text, "xml")
    except Exception:
        container_soup = BeautifulSoup(container_xml_text, "html.parser")

    rootfile = container_soup.find("rootfile")
    if not rootfile or not rootfile.get("full-path"):
        return sorted(
            [p for p in extract_dir.rglob("*") if p.suffix.lower() in {".html", ".htm", ".xhtml"}]
        )

    opf_path = extract_dir / rootfile.get("full-path")
    if not opf_path.exists():
        return sorted(
            [p for p in extract_dir.rglob("*") if p.suffix.lower() in {".html", ".htm", ".xhtml"}]
        )

    opf_text = opf_path.read_text(encoding="utf-8")
    try:
        opf_soup = BeautifulSoup(opf_text, "xml")
    except Exception:
        opf_soup = BeautifulSoup(opf_text, "html.parser")

    manifest_map: dict[str, Path] = {}
    opf_base = opf_path.parent
    for item in opf_soup.find_all("item"):
        item_id = item.get("id")
        href = item.get("href")
        if not item_id or not href:
            continue
        candidate = (opf_base / href).resolve()
        if candidate.exists() and candidate.suffix.lower() in {".html", ".htm", ".xhtml"}:
            manifest_map[item_id] = candidate

    ordered_files: list[Path] = []
    for itemref in opf_soup.find_all("itemref"):
        item_idref = itemref.get("idref")
        if not item_idref:
            continue
        file_path = manifest_map.get(item_idref)
        if file_path:
            ordered_files.append(file_path)

    if ordered_files:
        return ordered_files

    return sorted(
        [p for p in extract_dir.rglob("*") if p.suffix.lower() in {".html", ".htm", ".xhtml"}]
    )


def _local_name(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        return tag.rsplit("}", 1)[1].lower()
    return tag.lower()


class LLMTranslator:
    SEGMENT_SEPARATOR = "<<SEGMENT_SPLIT_9f3c6d>>"

    def __init__(self, source_language: str, target_language: str) -> None:
        self.source_language = source_language
        self.target_language = target_language
        self.memory: dict[str, str] = {}
        self.provider = "lmstudio"
        self.temperature = float(os.getenv("TRANSLATION_TEMPERATURE", "0.1"))
        self.model = os.getenv("LMSTUDIO_MODEL", "").strip()
        self.lmstudio_base_url = os.getenv("LMSTUDIO_BASE_URL", "").strip().rstrip("/")
        if not self.model:
            raise RuntimeError("Missing LMSTUDIO_MODEL environment variable")
        if not self.lmstudio_base_url:
            raise RuntimeError("Missing LMSTUDIO_BASE_URL environment variable")

    def _build_system_prompt(self) -> str:
        return f"""
You are a professional literary translator.
Translate from {self.source_language} to {self.target_language}.

Rules:
1) Keep meaning, tone, and punctuation fidelity.
2) Preserve proper nouns and names as in source unless translation is obvious and standard.
3) Do not add commentary or explanations.
4) Return plain text only.
""".strip()

    async def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        to_translate: list[str] = []
        to_translate_positions: list[int] = []
        out: list[str] = [""] * len(texts)

        for i, source in enumerate(texts):
            norm = _normalize_whitespace(source)
            if norm in self.memory:
                out[i] = self.memory[norm]
            else:
                to_translate.append(source)
                to_translate_positions.append(i)

        if not to_translate:
            return out

        payload = self._build_batch_payload(to_translate)

        max_retries = 4
        for attempt in range(max_retries):
            raw_text = ""
            try:
                raw_text = await self._call_model(payload)
                parsed = self._parse_batch_output(raw_text, len(to_translate))

                for src, translated in zip(to_translate, parsed, strict=True):
                    self.memory[_normalize_whitespace(src)] = translated
                for idx, translated in zip(to_translate_positions, parsed, strict=True):
                    out[idx] = translated
                return out
            except Exception as exc:
                exc_text = str(exc)
                self._debug_fallback_log(
                    stage="batch-parse-failed",
                    attempt=attempt + 1,
                    error=exc_text,
                    payload=payload,
                    response_locals=locals(),
                )
                if (
                    "Invalid translations array shape" in exc_text
                    or "Response is not valid JSON" in exc_text
                    or "Invalid control character" in exc_text
                    or "Invalid segmented output shape" in exc_text
                ):
                    # Prefer best-effort alignment over recursive split loops.
                    try:
                        recovered = self._parse_segmented_lenient(raw_text, to_translate)
                    except Exception as lenient_exc:
                        self._debug_fallback_log(
                            stage="lenient-parse-failed",
                            attempt=attempt + 1,
                            error=str(lenient_exc),
                            payload=payload,
                            response_locals={"raw_text": raw_text},
                        )
                        recovered = await self._translate_batch_individually(to_translate)

                    for src, translated in zip(to_translate, recovered, strict=True):
                        self.memory[_normalize_whitespace(src)] = translated
                    for idx, translated in zip(to_translate_positions, recovered, strict=True):
                        out[idx] = translated
                    return out
                if attempt == max_retries - 1:
                    raise RuntimeError(f"{self.provider} translation failed: {exc}") from exc
                await asyncio.sleep(2**attempt)

        return out

    def _build_batch_payload(self, items: list[str]) -> str:
        # Plain-text segmented protocol avoids fragile JSON generation in smaller local models.
        segments = []
        for item in items:
            segments.append(item)
        joined = f"\n{self.SEGMENT_SEPARATOR}\n".join(segments)
        return (
            "Translate each segment below from "
            f"{self.source_language} to {self.target_language}. "
            "Return only translated segments in the same order, joined by this exact separator token:\n"
            f"{self.SEGMENT_SEPARATOR}\n\n"
            "SEGMENTS START\n"
            f"{joined}\n"
            "SEGMENTS END"
        )

    def _clean_model_text(self, raw_text: str | None) -> str:
        if not raw_text:
            return ""
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _parse_batch_output(self, raw_text: str | None, expected_count: int) -> list[str]:
        # Strict mode: separator count must match expected segment count.
        cleaned = self._clean_model_text(raw_text)
        if not cleaned:
            raise ValueError("Invalid segmented output shape")
        parts = [p.strip() for p in cleaned.split(self.SEGMENT_SEPARATOR)]
        parts = [p for p in parts if p != ""]

        if expected_count == 1 and len(parts) == 1:
            return parts
        if len(parts) != expected_count:
            raise ValueError("Invalid segmented output shape")
        return parts

    def _parse_segmented_lenient(self, raw_text: str | None, source_items: list[str]) -> list[str]:
        # Lenient mode prevents endless retries: keep best output, pad missing with source.
        cleaned = self._clean_model_text(raw_text)
        if not cleaned:
            raise ValueError("Lenient parse: empty response")

        parts = [p.strip() for p in cleaned.split(self.SEGMENT_SEPARATOR)]
        parts = [p for p in parts if p != ""]
        if not parts:
            raise ValueError("Lenient parse: no usable translated items")

        expected = len(source_items)
        actual = len(parts)
        if actual != expected:
            print(
                f"[WARN] Lenient alignment engaged: expected={expected} actual={actual}. "
                "Missing items will reuse source text; extra items will be truncated."
            )

        if actual > expected:
            return parts[:expected]
        if actual < expected:
            return parts + source_items[actual:expected]
        return parts

    async def _call_model(self, payload: str) -> str:
        return await asyncio.to_thread(self._call_lmstudio, payload)

    async def _translate_batch_individually(self, texts: list[str]) -> list[str]:
        translated: list[str] = []
        for text in texts:
            single_payload = self._build_batch_payload([text])
            max_retries = 3
            last_exc: Exception | None = None
            for attempt in range(max_retries):
                try:
                    raw_text = await self._call_model(single_payload)
                    parsed = self._parse_batch_output(raw_text, 1)
                    translated.append(parsed[0])
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    self._debug_fallback_log(
                        stage="single-item-fallback",
                        attempt=attempt + 1,
                        error=str(exc),
                        payload=single_payload,
                        response_locals=locals(),
                    )
                    await asyncio.sleep(2**attempt)
            if last_exc is not None:
                if (
                    "Invalid control character" in str(last_exc)
                    or "not valid JSON" in str(last_exc)
                    or "Invalid segmented output shape" in str(last_exc)
                ):
                    translated.append(await self._translate_single_plain(text))
                    continue
                raise RuntimeError(f"Single-item recovery failed: {last_exc}") from last_exc
        return translated

    async def _translate_single_plain(self, text: str) -> str:
        prompt = (
            f"Translate from {self.source_language} to {self.target_language}. "
            "Output only translated text, no JSON, no markdown.\n\n"
            f"TEXT:\n{text}"
        )

        endpoint = f"{self.lmstudio_base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a professional translator. Return only translated text."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "stream": False,
        }
        req = urlrequest.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=180) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as exc:
            raise RuntimeError(f"Plain single-item fallback failed: {exc}") from exc

        parsed = json.loads(raw)
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Plain single-item fallback missing choices")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Plain single-item fallback returned empty content")
        return content.strip()

    def _call_lmstudio(self, payload: str) -> str:
        endpoint = f"{self.lmstudio_base_url}/chat/completions"
        base_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": payload},
            ],
            "temperature": self.temperature,
            "stream": False,
        }

        attempted_bodies = [
            {**base_body, "response_format": {"type": "json_object"}},
            base_body,
        ]
        raw = None
        last_error = None

        for body in attempted_bodies:
            req = urlrequest.Request(
                endpoint,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlrequest.urlopen(req, timeout=180) as resp:
                    raw = resp.read().decode("utf-8")
                    break
            except urlerror.HTTPError as exc:
                error_body = ""
                try:
                    error_body = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    error_body = ""
                last_error = f"HTTP {exc.code}: {error_body or str(exc)}"
                if exc.code == 400 and "response_format" in body:
                    continue
                raise RuntimeError(f"LM Studio request failed: {last_error}") from exc
            except urlerror.URLError as exc:
                raise RuntimeError(f"LM Studio request failed: {exc}") from exc

        if raw is None:
            raise RuntimeError(f"LM Studio request failed: {last_error or 'unknown error'}")

        parsed = json.loads(raw)
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LM Studio response missing choices")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LM Studio response missing message content")
        return content

    def _debug_fallback_log(
        self,
        stage: str,
        attempt: int,
        error: str,
        payload: str,
        response_locals: dict[str, Any],
    ) -> None:
        if not DEBUG_FALLBACK_LOGS:
            return
        raw_text = response_locals.get("raw_text", "<raw_text_not_available>")
        print("=== TRANSLATION FALLBACK DEBUG START ===")
        print(f"stage={stage} attempt={attempt} model={self.model}")
        print(f"error={error}")
        print("payload:")
        print(payload)
        print("response_raw:")
        print(raw_text)
        print("=== TRANSLATION FALLBACK DEBUG END ===")


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: str
    output_file: str | None = None
    error: str | None = None
    benchmark: dict[str, Any] | None = None
    options: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    pause_requested: bool = False


jobs: dict[str, JobStatus] = {}
active_tasks: dict[str, asyncio.Task] = {}
app = FastAPI(title="EPUB Translator", version="2.0.0")


def _job_dir(job_id: str) -> Path:
    return WORK_ROOT / job_id


def _job_file(job_id: str) -> Path:
    return _job_dir(job_id) / "job.json"


def _job_to_dict(job: JobStatus) -> dict[str, Any]:
    if hasattr(job, "model_dump"):
        return job.model_dump()
    return job.dict()


def _save_job(job_id: str) -> None:
    # Persist job metadata/checkpoints so long runs can survive process restarts.
    job = jobs.get(job_id)
    if not job:
        return
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    _job_file(job_id).write_text(json.dumps(_job_to_dict(job), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_jobs_from_disk() -> None:
    if not WORK_ROOT.exists():
        return
    for d in sorted([p for p in WORK_ROOT.iterdir() if p.is_dir()]):
        f = d / "job.json"
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            job = JobStatus(**data)
            if job.status in {"running", "queued"}:
                job.status = "paused"
                job.pause_requested = False
                job.progress = "Paused after restart. Resume to continue from checkpoint."
            jobs[job.job_id] = job
            _save_job(job.job_id)
        except Exception:
            continue


@app.on_event("startup")
async def startup_event() -> None:
    _load_jobs_from_disk()


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    if not FRONTEND_FILE.exists():
        raise HTTPException(status_code=500, detail="frontend/index.html not found")
    return FRONTEND_FILE.read_text(encoding="utf-8")


@app.post("/api/translate")
async def start_translation(
    file: UploadFile = File(...),
    source_language: str = Form("Auto-detect"),
    target_language: str = Form(...),
    test_mode: str = Form(""),
    test_chapter_limit: int = Form(0),
    start_chapter: int = Form(1),
    end_chapter: int = Form(0),
) -> dict[str, str]:
    if not file.filename.lower().endswith(".epub"):
        raise HTTPException(status_code=400, detail="Only .epub files are accepted")
    if not target_language.strip():
        raise HTTPException(status_code=400, detail="target_language is required")

    job_id = str(uuid.uuid4())
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)

    input_epub = d / "input.epub"
    with input_epub.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    default_test_mode = _parse_bool(os.getenv("TRANSLATION_TEST_MODE", "false"), default=False)
    default_test_limit = max(1, _parse_int(os.getenv("TRANSLATION_TEST_CHAPTER_LIMIT", "5"), 5))

    resolved_test_mode = _parse_bool(test_mode, default=default_test_mode)
    resolved_test_limit = test_chapter_limit if test_chapter_limit > 0 else default_test_limit
    resolved_start = max(1, start_chapter)
    resolved_end = max(0, end_chapter)

    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="queued",
        progress="Job queued",
        benchmark={"test_mode": resolved_test_mode, "test_chapter_limit": resolved_test_limit},
        options={
            "source_language": source_language.strip() or "Auto-detect",
            "target_language": target_language.strip(),
            "test_mode": resolved_test_mode,
            "test_chapter_limit": resolved_test_limit,
            "start_chapter": resolved_start,
            "end_chapter": resolved_end,
            "input_filename": file.filename,
        },
        checkpoint={"next_chapter_cursor": 0},
        pause_requested=False,
    )
    _save_job(job_id)
    _launch_job(job_id)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> JobStatus:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/jobs/{job_id}/pause")
async def pause_job(job_id: str) -> dict[str, str]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "running":
        return {"status": job.status, "message": "Job is not running"}
    job.pause_requested = True
    job.progress = "Pause requested. Finishing current step..."
    _save_job(job_id)
    return {"status": "ok", "message": "Pause requested"}


@app.post("/api/jobs/{job_id}/resume")
async def resume_job(job_id: str) -> dict[str, str]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in {"paused", "failed"}:
        return {"status": job.status, "message": "Job is not paused/failed"}
    if not job.options:
        raise HTTPException(status_code=400, detail="Missing job options for resume")
    _launch_job(job_id)
    return {"status": "ok", "message": "Resume started"}


@app.get("/api/download/{job_id}")
async def download_result(job_id: str) -> FileResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed" or not job.output_file:
        raise HTTPException(status_code=409, detail="Translation is not ready")

    output_path = Path(job.output_file)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file no longer exists")

    return FileResponse(path=output_path, filename=f"translated_{job_id}.epub", media_type="application/epub+zip")


def _launch_job(job_id: str) -> None:
    existing = active_tasks.get(job_id)
    if existing and not existing.done():
        return

    job = jobs[job_id]
    options = job.options or {}
    input_epub = _job_dir(job_id) / "input.epub"
    if not input_epub.exists():
        job.status = "failed"
        job.error = "Missing input.epub for job"
        _save_job(job_id)
        return

    task = asyncio.create_task(
        _run_translation_job(
            job_id=job_id,
            input_epub=input_epub,
            source_language=options.get("source_language", "Auto-detect"),
            target_language=options.get("target_language", "Brazilian Portuguese"),
            test_mode=bool(options.get("test_mode", False)),
            test_chapter_limit=int(options.get("test_chapter_limit", 5)),
            start_chapter=int(options.get("start_chapter", 1)),
            end_chapter=int(options.get("end_chapter", 0)),
        )
    )
    active_tasks[job_id] = task



def _build_batches(
    segments: list[dict[str, Any]],
    max_chars: int = CHUNK_MAX_CHARS,
    max_items: int = CHUNK_MAX_ITEMS,
) -> list[list[dict[str, Any]]]:
    # Sequential batching keeps chapter order deterministic and token usage bounded.
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0

    for seg in segments:
        seg_len = len(seg["core"])
        would_exceed = (current_chars + seg_len > max_chars) or (len(current) >= max_items)
        if current and would_exceed:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(seg)
        current_chars += seg_len

    if current:
        batches.append(current)
    return batches


async def _run_translation_job(
    job_id: str,
    input_epub: Path,
    source_language: str,
    target_language: str,
    test_mode: bool,
    test_chapter_limit: int,
    start_chapter: int,
    end_chapter: int,
) -> None:
    # Long-running workflow: extract -> translate by chapter/chunk -> repackage -> checkpoint.
    job = jobs[job_id]
    started_at = time.time()
    run_start_perf = time.perf_counter()

    chapter_stats: list[dict[str, Any]] = []
    total_chunks = 0
    total_segments = 0
    total_source_chars = 0
    total_chapters_in_book = 0
    processed_chapters = 0

    try:
        job.status = "running"
        job.error = None
        job.pause_requested = False
        job.progress = "Preparing EPUB"
        _save_job(job_id)

        extract_dir = input_epub.parent / "extracted"
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(input_epub, "r") as zf:
                zf.extractall(extract_dir)

        chapter_files = _resolve_spine_html_files(extract_dir)
        if not chapter_files:
            raise RuntimeError("No HTML/XHTML chapters found in EPUB")

        total_chapters_in_book = len(chapter_files)
        start_idx = max(1, start_chapter) - 1
        end_idx = total_chapters_in_book if end_chapter <= 0 else min(total_chapters_in_book, end_chapter)
        if start_idx >= end_idx:
            raise RuntimeError("Invalid chapter range")

        selected = chapter_files[start_idx:end_idx]
        if test_mode:
            selected = selected[: max(1, test_chapter_limit)]

        checkpoint = job.checkpoint or {"next_chapter_cursor": 0}
        next_cursor = max(0, int(checkpoint.get("next_chapter_cursor", 0)))
        if next_cursor > len(selected):
            next_cursor = len(selected)

        translator = LLMTranslator(source_language=source_language, target_language=target_language)

        for cursor in range(next_cursor, len(selected)):
            if job.pause_requested:
                job.status = "paused"
                job.progress = f"Paused at chapter cursor {cursor}."
                job.checkpoint = {"next_chapter_cursor": cursor}
                _save_job(job_id)
                return

            chapter_idx = cursor + 1
            chapter_path = selected[cursor]
            rel = chapter_path.relative_to(extract_dir)
            chapter_start = time.perf_counter()
            chapter_segments = 0
            chapter_chars = 0

            job.progress = f"Translating chapter {chapter_idx}/{len(selected)}: {rel}"
            _save_job(job_id)

            original = chapter_path.read_text(encoding="utf-8", errors="ignore")
            is_xhtml = chapter_path.suffix.lower() == ".xhtml" or original.lstrip().startswith("<?xml")
            segments: list[dict[str, Any]] = []

            if is_xhtml:
                # XHTML path is strict-XML to keep Apple Books/EPUB readers compatible.
                parser = etree.XMLParser(recover=False, resolve_entities=False)
                root = etree.fromstring(original.encode("utf-8"), parser=parser)

                for elem in root.iter():
                    elem_tag = _local_name(elem.tag)
                    if elem_tag in SKIP_PARENT_TAGS:
                        continue

                    if elem.text:
                        leading, core, trailing = _split_edge_whitespace(str(elem.text))
                        if core and _looks_translatable(core):
                            segments.append({"element": elem, "slot": "text", "leading": leading, "core": core, "trailing": trailing})

                    if elem.tail:
                        parent = elem.getparent()
                        parent_tag = _local_name(parent.tag) if parent is not None else ""
                        if parent_tag in SKIP_PARENT_TAGS:
                            continue
                        leading, core, trailing = _split_edge_whitespace(str(elem.tail))
                        if core and _looks_translatable(core):
                            segments.append({"element": elem, "slot": "tail", "leading": leading, "core": core, "trailing": trailing})
            else:
                soup = BeautifulSoup(original, "html.parser")
                for node in soup.find_all(string=True):
                    if not isinstance(node, NavigableString):
                        continue
                    parent = node.parent
                    if parent and parent.name and parent.name.lower() in SKIP_PARENT_TAGS:
                        continue

                    leading, core, trailing = _split_edge_whitespace(str(node))
                    if core and _looks_translatable(core):
                        segments.append({"node": node, "leading": leading, "core": core, "trailing": trailing})

            if segments:
                batches = _build_batches(segments)
                total_chunks += len(batches)
                for batch_idx, batch in enumerate(batches, start=1):
                    if job.pause_requested:
                        job.status = "paused"
                        job.progress = f"Paused at chapter {chapter_idx}."
                        job.checkpoint = {"next_chapter_cursor": cursor}
                        _save_job(job_id)
                        return

                    job.progress = f"Chapter {chapter_idx}/{len(selected)} - chunk {batch_idx}/{len(batches)}"
                    _save_job(job_id)

                    texts = [seg["core"] for seg in batch]
                    chapter_segments += len(texts)
                    chapter_chars += sum(len(t) for t in texts)
                    translated = await translator.translate_batch(texts)

                    for seg, translated_text in zip(batch, translated, strict=True):
                        merged = f"{seg['leading']}{_sanitize_xml_text(translated_text)}{seg['trailing']}"
                        if "node" in seg:
                            seg["node"].replace_with(merged)
                        elif seg["slot"] == "text":
                            seg["element"].text = merged
                        else:
                            seg["element"].tail = merged

                if is_xhtml:
                    xml_out = etree.tostring(
                        root,
                        encoding="utf-8",
                        xml_declaration=original.lstrip().startswith("<?xml"),
                        pretty_print=False,
                    )
                    chapter_path.write_bytes(xml_out)
                else:
                    chapter_path.write_text(str(soup), encoding="utf-8")

            chapter_duration = round(time.perf_counter() - chapter_start, 2)
            chapter_stats.append(
                {
                    "chapter": chapter_idx,
                    "path": str(rel),
                    "seconds": chapter_duration,
                    "segments": chapter_segments,
                    "source_chars": chapter_chars,
                }
            )
            processed_chapters = chapter_idx
            total_segments += chapter_segments
            total_source_chars += chapter_chars

            job.checkpoint = {"next_chapter_cursor": cursor + 1}
            _save_job(job_id)

        output_epub = input_epub.parent / "translated.epub"
        _repackage_epub(extract_dir, output_epub)

        shutil.rmtree(extract_dir, ignore_errors=True)

        total_seconds = time.perf_counter() - run_start_perf
        job.status = "completed"
        job.progress = "Completed"
        job.output_file = str(output_epub)
        job.benchmark = {
            "started_at_unix": int(started_at),
            "completed_at_unix": int(time.time()),
            "test_mode": test_mode,
            "test_chapter_limit": test_chapter_limit,
            "start_chapter": start_chapter,
            "end_chapter": end_chapter,
            "chapters_processed": processed_chapters,
            "chapters_in_book": total_chapters_in_book,
            "chunks_processed": total_chunks,
            "segments_translated": total_segments,
            "source_chars_translated": total_source_chars,
            "duration_seconds": round(total_seconds, 2),
            "avg_seconds_per_chapter": round(total_seconds / max(processed_chapters, 1), 2),
            "avg_chars_per_second": round(total_source_chars / max(total_seconds, 0.001), 2),
            "per_chapter": chapter_stats,
        }
        _save_job(job_id)
    except Exception as exc:
        total_seconds = time.perf_counter() - run_start_perf
        job.status = "failed"
        job.error = str(exc)
        job.progress = "Failed"
        job.benchmark = {
            "started_at_unix": int(started_at),
            "failed_at_unix": int(time.time()),
            "test_mode": test_mode,
            "test_chapter_limit": test_chapter_limit,
            "start_chapter": start_chapter,
            "end_chapter": end_chapter,
            "chapters_processed": processed_chapters,
            "chapters_in_book": total_chapters_in_book,
            "chunks_processed": total_chunks,
            "segments_translated": total_segments,
            "source_chars_translated": total_source_chars,
            "duration_seconds": round(total_seconds, 2),
            "per_chapter": chapter_stats,
        }
        _save_job(job_id)
    finally:
        current = asyncio.current_task()
        if active_tasks.get(job_id) is current:
            active_tasks.pop(job_id, None)


def _repackage_epub(extract_dir: Path, output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()

    mimetype = extract_dir / "mimetype"
    with zipfile.ZipFile(output_path, "w") as zf:
        if mimetype.exists():
            zf.write(mimetype, arcname="mimetype", compress_type=zipfile.ZIP_STORED)

        for path in sorted(extract_dir.rglob("*")):
            if not path.is_file():
                continue
            arcname = str(path.relative_to(extract_dir))
            if arcname == "mimetype":
                continue
            zf.write(path, arcname=arcname, compress_type=zipfile.ZIP_DEFLATED)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
