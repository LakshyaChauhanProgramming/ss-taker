import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ss_taker import app as app_module
from ss_taker import meeting as meeting_module
from ss_taker.audio import SpeechSegment
from ss_taker.meeting import MeetingConfig, MeetingManager


class FakeOCR:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls: list[dict] = []

    def capture_and_ocr(self, **kwargs):
        self.calls.append(kwargs)
        text = self.texts[min(len(self.calls), len(self.texts)) - 1]
        return SimpleNamespace(text=text)


class FakeRecorder:
    def __init__(self, name, on_segment, **_kwargs) -> None:
        self.name = name
        self.on_segment = on_segment
        self.error = None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.on_segment(self.name, SpeechSegment(start=2.0, text="let's ship on friday"))


def _wait_for(condition, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        time.sleep(0.01)


def test_session_combines_screen_and_audio(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(meeting_module, "audio_available", lambda: None)
    ocr = FakeOCR(["Agenda: launch", "Agenda: launch\nBob: friday works"])
    manager = MeetingManager(tmp_path)
    config = MeetingConfig(title="Launch sync", ocr_interval_seconds=0.01, audio_sources=["others"])
    session = manager.start(config, ocr, recorder_factory=FakeRecorder)
    _wait_for(lambda: len(ocr.calls) >= 2)
    session.stop()

    assert ocr.calls[0]["persist"] is False
    text = session.transcript_text()
    assert "(screen) Agenda: launch" in text
    assert "(screen) Bob: friday works" in text
    assert "(audio:others) let's ship on friday" in text
    assert text.count("Agenda: launch") == 1
    saved = json.loads((tmp_path / f"{session.id}.json").read_text())
    assert saved["status"] == "stopped"
    assert len(saved["segments"]) == 3


def test_only_one_meeting_records_at_a_time(tmp_path) -> None:
    manager = MeetingManager(tmp_path)
    first = manager.start(MeetingConfig(audio=False, ocr_interval_seconds=0.01), FakeOCR(["x"]))
    with pytest.raises(meeting_module.MeetingBusyError):
        manager.start(MeetingConfig(audio=False), FakeOCR(["x"]))
    first.stop()
    manager.start(MeetingConfig(audio=False, ocr_interval_seconds=0.01), FakeOCR(["x"])).stop()


def test_missing_audio_support_is_reported_not_fatal(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(meeting_module, "audio_available", lambda: "parec not found")
    session = MeetingManager(tmp_path).start(MeetingConfig(ocr=False), None)
    session.stop()
    assert session.info()["errors"] == ["audio disabled: parec not found"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("SS_TAKER_TOKEN", raising=False)
    monkeypatch.setattr(app_module, "meetings", MeetingManager(tmp_path))
    monkeypatch.setattr(app_module, "_get_service", lambda: FakeOCR(["Decision: launch friday"]))
    monkeypatch.setattr(meeting_module, "audio_available", lambda: "disabled in tests")
    return TestClient(app_module.app)


def test_meeting_api_flow(client, monkeypatch) -> None:
    started = client.post("/meetings", json={"title": "Sync", "ocr_interval_seconds": 1, "audio": False})
    assert started.status_code == 201
    meeting_id = started.json()["id"]
    assert client.post("/meetings", json={"audio": False}).status_code == 409

    _wait_for(lambda: client.get(f"/meetings/{meeting_id}").json()["segments"] >= 1)
    stopped = client.post(f"/meetings/{meeting_id}/stop").json()
    assert stopped["status"] == "stopped"
    assert "(screen) Decision: launch friday" in client.get(f"/meetings/{meeting_id}/transcript").text

    def fake_summarize(transcript, **_kwargs):
        assert "launch friday" in transcript
        return meeting_module.summarizer.MeetingSummary(summary="Launch agreed.", decisions=["Launch friday"])

    monkeypatch.setattr(meeting_module.summarizer, "summarize", fake_summarize)
    summary = client.post(f"/meetings/{meeting_id}/summarize", json={"language": "Hinglish"})
    assert summary.status_code == 200
    assert summary.json()["decisions"] == ["Launch friday"]
    assert client.get(f"/meetings/{meeting_id}").json()["summary"]["summary"] == "Launch agreed."


def test_meeting_api_errors(client, monkeypatch) -> None:
    assert client.get("/meetings/nope").status_code == 404
    assert client.post("/meetings", json={"ocr": False, "audio": False}).status_code == 400
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    meeting_id = client.post("/meetings", json={"ocr": False, "audio": True}).json()["id"]
    client.post(f"/meetings/{meeting_id}/stop")
    response = client.post(f"/meetings/{meeting_id}/summarize")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_meeting_endpoints_need_token(client, monkeypatch) -> None:
    monkeypatch.setenv("SS_TAKER_TOKEN", "secret")
    assert client.get("/meetings").status_code == 401
    assert client.post("/meetings", json={}).status_code == 401
    assert client.get("/meetings", headers={"X-API-Key": "secret"}).status_code == 200
