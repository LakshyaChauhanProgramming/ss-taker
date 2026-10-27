import json

import httpx
import pytest

from ss_taker import summarizer


def _client(replies: list[httpx.Response], seen: list[dict]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return replies.pop(0)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _reply(content: str, model: str = "some/free-model:free") -> httpx.Response:
    return httpx.Response(200, json={"model": model, "choices": [{"message": {"content": content}}]})


GOOD = json.dumps(
    {
        "summary": "Team planned the launch.",
        "key_topics": [{"topic": "Launch", "points": ["Ship on Friday"]}],
        "decisions": ["Launch on Friday"],
    }
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(summarizer.time, "sleep", lambda _s: None)


def test_summarize_parses_reply_and_reports_model() -> None:
    seen: list[dict] = []
    result = summarizer.summarize("[00:01] (screen) launch friday", client=_client([_reply(GOOD)], seen))
    assert result.summary == "Team planned the launch."
    assert result.key_topics == [{"topic": "Launch", "points": ["Ship on Friday"]}]
    assert result.decisions == ["Launch on Friday"]
    assert result.model == "some/free-model:free"
    assert seen[0]["model"] == summarizer.DEFAULT_MODEL


def test_fenced_json_is_accepted() -> None:
    assert summarizer.parse_json_reply(f"Here you go:\n```json\n{GOOD}\n```")["decisions"] == ["Launch on Friday"]


def test_bad_reply_is_retried() -> None:
    seen: list[dict] = []
    replies = [_reply("sorry, I cannot"), httpx.Response(429, text="rate limited"), _reply(GOOD)]
    result = summarizer.summarize("[00:01] (screen) hi", client=_client(replies, seen))
    assert result.decisions == ["Launch on Friday"]
    assert len(seen) == 3


def test_auth_error_is_not_retried() -> None:
    seen: list[dict] = []
    with pytest.raises(summarizer.SummarizerError, match="401"):
        summarizer.summarize("[00:01] (screen) hi", client=_client([httpx.Response(401, text="bad key")], seen))
    assert len(seen) == 1


def test_long_transcript_is_summarized_in_parts(monkeypatch) -> None:
    monkeypatch.setattr(summarizer, "CHUNK_CHARS", 70)
    seen: list[dict] = []
    transcript = "\n".join(f"[00:0{i}] (screen) line number {i}" for i in range(4))
    result = summarizer.summarize(transcript, client=_client([_reply(GOOD)] * 3, seen))
    assert result.parts == 2
    assert len(seen) == 3
    assert "summarized separately" in seen[2]["messages"][0]["content"]


def test_missing_key_and_empty_transcript_are_input_errors(monkeypatch) -> None:
    with pytest.raises(summarizer.SummarizerInputError):
        summarizer.summarize("   ")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(summarizer.SummarizerInputError):
        summarizer.summarize("[00:01] (screen) hi")
