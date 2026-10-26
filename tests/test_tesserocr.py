import cv2
import numpy as np
import pytest

from ss_taker import ocr


def _text_image() -> np.ndarray:
    image = np.full((120, 900, 3), 255, dtype=np.uint8)
    cv2.putText(image, "Quarterly review moved to Friday", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    return image


def test_parse_tsv_matches_pytesseract_dict_shape() -> None:
    tsv = "1\t1\t0\t0\t0\t0\t0\t0\t10\t10\t-1\t\n5\t1\t1\t1\t1\t1\t2\t3\t4\t5\t96.5\tHello"
    data = ocr._parse_tsv(tsv)
    assert data["text"] == ["", "Hello"]
    assert data["conf"] == ["-1", "96.5"]
    assert data["line_num"] == ["0", "1"]


def test_tesserocr_and_pytesseract_read_the_same_text(monkeypatch) -> None:
    if ocr.tesserocr is None or ocr._find_tessdata() is None:
        pytest.skip("tesserocr or tessdata not available")
    image = _text_image()

    engine = ocr.TesserocrEngine(ocr._find_tessdata())
    monkeypatch.setattr(ocr, "get_tesserocr_engine", lambda: engine)
    fast_text, fast_conf, _ = ocr._ocr_worker_scored(image.tobytes(), image.shape, "eng", 6, None, False)

    monkeypatch.setattr(ocr, "get_tesserocr_engine", lambda: None)
    cli_text, _, _ = ocr._ocr_worker_scored(image.tobytes(), image.shape, "eng", 6, None, False)

    assert "Friday" in fast_text
    assert fast_text == cli_text
    assert fast_conf > 50


def test_tesserocr_engine_resets_whitelist_between_calls() -> None:
    if ocr.tesserocr is None or ocr._find_tessdata() is None:
        pytest.skip("tesserocr or tessdata not available")
    engine = ocr.TesserocrEngine(ocr._find_tessdata())
    image = _text_image()
    digits = engine.image_to_data(image, lang="eng", psm=6, whitelist="0123456789", preserve_layout=False)
    plain = engine.image_to_data(image, lang="eng", psm=6, whitelist=None, preserve_layout=False)
    assert not any(ch.isalpha() for ch in "".join(digits["text"]))
    assert "Friday" in " ".join(plain["text"])
