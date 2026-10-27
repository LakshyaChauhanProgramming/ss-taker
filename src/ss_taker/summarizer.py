from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field

import httpx

LOGGER = logging.getLogger("ss_taker")

OPENROUTER_URL = os.environ.get("SS_TAKER_LLM_URL", "https://openrouter.ai/api/v1/chat/completions")
# openrouter/free picks any free model that fits the request; pin a model here
# once we settle on one.
DEFAULT_MODEL = os.environ.get("SS_TAKER_LLM_MODEL", "openrouter/free")
LLM_TIMEOUT_SECONDS = float(os.environ.get("SS_TAKER_LLM_TIMEOUT", "120"))
# Transcripts longer than this are summarized in parts first, then merged.
# Free models have small or unknown context windows, so stay well below them.
CHUNK_CHARS = int(os.environ.get("SS_TAKER_LLM_CHUNK_CHARS", "24000"))
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """You summarize meetings from an automatically captured transcript.
Each line is "[mm:ss] (source) text". Sources:
- screen: OCR of the meeting window (captions, chat, shared slides). Expect OCR noise and UI text such as button labels; ignore anything that is not meeting content.
- audio:others / audio:me: speech recognition of the other participants / of the user. Expect misheard words.
Speech may mix languages (for example Hindi and English). Use both sources together; when they disagree, prefer the reading that makes sense in context.
Only report what the transcript supports. Do not invent names, numbers or decisions.

Reply with a single JSON object and nothing else:
{
  "summary": "5-8 sentence overview of the meeting",
  "key_topics": [{"topic": "short title", "points": ["what was said about it"]}],
  "decisions": ["each decision that was actually agreed"]
}
Use empty lists when there is nothing to report."""

MERGE_PROMPT = """The meeting was too long for one pass, so its parts were summarized separately.
Below are the part summaries as JSON, in order. Merge them into one summary of the whole meeting:
combine repeated topics, keep every distinct decision once, and write the overview for the whole meeting."""


class SummarizerError(RuntimeError):
    pass


class SummarizerInputError(SummarizerError):
    """The request cannot be summarized as-is (no transcript, no API key)."""


class _FatalError(SummarizerError):
    """A failure that retrying cannot fix, such as a bad API key."""


@dataclass(slots=True)
class MeetingSummary:
    summary: str
    key_topics: list[dict[str, object]] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    model: str = ""
    """The model OpenRouter actually used (openrouter/free picks one per call)."""
    parts: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def configured() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def summarize(
    transcript: str,
    *,
    title: str = "",
    language: str = "English",
    client: httpx.Client | None = None,
) -> MeetingSummary:
    if not transcript.strip():
        raise SummarizerInputError("Transcript is empty; nothing to summarize")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SummarizerInputError("OPENROUTER_API_KEY is not set")

    own_client = client is None
    client = client or httpx.Client(timeout=LLM_TIMEOUT_SECONDS)
    try:
        llm = _OpenRouter(client, api_key)
        instructions = f"{SYSTEM_PROMPT}\nWrite the summary in {language}."
        header = f"Meeting title: {title}\n\n" if title else ""
        chunks = split_transcript(transcript, CHUNK_CHARS)
        if len(chunks) == 1:
            result = llm.complete_json(instructions, header + "Transcript:\n" + chunks[0])
            return _to_summary(result, llm.last_model, parts=1)

        partials = []
        for i, chunk in enumerate(chunks, 1):
            LOGGER.info("summarizing meeting part %d/%d", i, len(chunks))
            partials.append(
                llm.complete_json(instructions, f"{header}Transcript part {i} of {len(chunks)}:\n{chunk}")
            )
        merged = llm.complete_json(
            f"{instructions}\n\n{MERGE_PROMPT}",
            header + "\n\n".join(json.dumps(p, ensure_ascii=False) for p in partials),
        )
        return _to_summary(merged, llm.last_model, parts=len(chunks))
    finally:
        if own_client:
            client.close()


def split_transcript(transcript: str, max_chars: int) -> list[str]:
    """Split on line boundaries into pieces of at most max_chars (one long line may exceed it)."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in transcript.splitlines():
        if current and size + len(line) + 1 > max_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def parse_json_reply(text: str) -> dict[str, object]:
    """Free models often wrap JSON in ``` fences or add a sentence around it."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise SummarizerError("Model reply contained no JSON object")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SummarizerError(f"Model reply was not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SummarizerError("Model reply was not a JSON object")
    return value


def _to_summary(data: dict[str, object], model: str, *, parts: int) -> MeetingSummary:
    topics = []
    for item in data.get("key_topics") or []:
        if isinstance(item, dict) and item.get("topic"):
            points = item.get("points") or []
            topics.append(
                {
                    "topic": str(item["topic"]),
                    "points": [str(p) for p in points] if isinstance(points, list) else [str(points)],
                }
            )
        elif isinstance(item, str):
            topics.append({"topic": item, "points": []})
    decisions = data.get("decisions") or []
    return MeetingSummary(
        summary=str(data.get("summary") or "").strip(),
        key_topics=topics,
        decisions=[str(d) for d in decisions] if isinstance(decisions, list) else [str(decisions)],
        model=model,
        parts=parts,
    )


class _OpenRouter:
    def __init__(self, client: httpx.Client, api_key: str) -> None:
        self.client = client
        self.api_key = api_key
        self.last_model = DEFAULT_MODEL

    def complete_json(self, system: str, user: str) -> dict[str, object]:
        """Ask for JSON; retry on rate limits, server errors and unparsable replies."""
        error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            if attempt:
                time.sleep(2**attempt)
            try:
                return parse_json_reply(self._chat(system, user))
            except _FatalError:
                raise
            except (SummarizerError, httpx.TransportError) as exc:
                error = exc
                LOGGER.warning("LLM attempt %d failed: %s", attempt + 1, exc)
        raise SummarizerError(f"Summarization failed after {MAX_ATTEMPTS} attempts: {error}")

    def _chat(self, system: str, user: str) -> str:
        response = self.client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "X-Title": "ss-taker",
            },
            json={
                "model": DEFAULT_MODEL,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
        )
        if response.status_code in (401, 402, 403):
            # Not worth retrying: bad key or no credit.
            raise _FatalError(f"OpenRouter rejected the request ({response.status_code}): {response.text[:300]}")
        if response.status_code >= 400:
            raise SummarizerError(f"OpenRouter returned {response.status_code}: {response.text[:300]}")
        body = response.json()
        if "error" in body:
            raise SummarizerError(f"OpenRouter error: {body['error']}")
        self.last_model = str(body.get("model") or DEFAULT_MODEL)
        try:
            return str(body["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise SummarizerError(f"Unexpected OpenRouter response: {str(body)[:300]}") from exc


