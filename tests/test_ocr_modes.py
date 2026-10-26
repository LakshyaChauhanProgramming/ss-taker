import numpy as np

from ss_taker.ocr import OCRService


def test_balanced_mode_uses_practical_candidate_set() -> None:
    service = OCRService(output_dir="output-test", start_pool=False)
    try:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        processed = np.zeros((20, 20), dtype=np.uint8)
        variants = service._quality_variants(image, processed, "balanced")
        assert [name for name, _ in variants] == ["raw", "processed"]
    finally:
        service.close()


def _stub_ocr(service: OCRService, results: dict[int, tuple[str, float, int]]) -> list[int]:
    """Make OCR return canned results keyed by image width; record calls."""
    calls: list[int] = []

    def fake(image, **_kwargs):
        calls.append(image.shape[1])
        text, conf, words = results.get(image.shape[1], ("", -1.0, 0))
        return text, conf, [{"text": "w", "confidence": conf}] * words

    service._ocr_scored = fake
    return calls


def test_balanced_mode_skips_preprocessing_when_raw_reads_well() -> None:
    service = OCRService(output_dir="output-test", start_pool=False)
    calls = _stub_ocr(service, {20: ("clear text here", 90.0, 10)})
    timings: dict[str, float] = {}
    candidates = service._ocr_candidates(
        np.zeros((20, 20, 3), dtype=np.uint8),
        lang="eng", psm=6, whitelist=None, preserve_layout=False, mode="balanced", timings=timings,
    )
    assert [c["variant"] for c in candidates] == ["raw"]
    assert calls == [20]
    assert "preprocess_ms" not in timings


def test_balanced_mode_adds_processed_pass_when_raw_is_weak() -> None:
    service = OCRService(output_dir="output-test", start_pool=False)
    _stub_ocr(service, {20: ("??", 20.0, 1)})
    candidates = service._ocr_candidates(
        np.zeros((20, 20, 3), dtype=np.uint8),
        lang="eng", psm=6, whitelist=None, preserve_layout=False, mode="balanced", timings={},
    )
    assert [c["variant"] for c in candidates] == ["raw", "processed"]


def test_fast_mode_reads_raw_only() -> None:
    service = OCRService(output_dir="output-test", start_pool=False)
    _stub_ocr(service, {20: ("??", 10.0, 1)})
    candidates = service._ocr_candidates(
        np.zeros((20, 20, 3), dtype=np.uint8),
        lang="eng", psm=6, whitelist=None, preserve_layout=False, mode="fast", timings={},
    )
    assert [c["variant"] for c in candidates] == ["raw"]
