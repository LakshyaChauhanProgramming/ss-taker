from ss_taker.transcript import Segment, Transcript, format_timestamp, render


def test_repeated_screen_lines_are_added_once() -> None:
    transcript = Transcript()
    assert transcript.add_screen_text(0, "Alice: hello everyone\nBob: hi there") == 2
    assert transcript.add_screen_text(5, "Alice: hello everyone\nBob: hi there\nAlice: lets start") == 1
    assert [s.text for s in transcript.segments()] == ["Alice: hello everyone", "Bob: hi there", "Alice: lets start"]


def test_growing_caption_replaces_earlier_segment() -> None:
    transcript = Transcript()
    transcript.add_screen_text(0, "we should ship")
    transcript.add_screen_text(5, "we should ship on friday")
    segments = transcript.segments()
    assert len(segments) == 1
    assert segments[0].text == "we should ship on friday"
    assert segments[0].t == 0


def test_ocr_noise_variant_and_short_junk_are_ignored() -> None:
    transcript = Transcript()
    transcript.add_screen_text(0, "Budget review for the next quarter")
    assert transcript.add_screen_text(5, "Budget reveiw for the next quarter.\n@ x\n|") == 0


def test_render_orders_sources_by_time() -> None:
    transcript = Transcript()
    transcript.add(Segment(t=65, source="audio:others", text="sounds good"))
    transcript.add_screen_text(3, "Agenda: launch plan")
    assert render(transcript.segments()) == "[00:03] (screen) Agenda: launch plan\n[01:05] (audio:others) sounds good"


def test_format_timestamp_with_hours() -> None:
    assert format_timestamp(3725) == "1:02:05"
