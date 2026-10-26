from ss_taker.ocr import OCRService


def test_cleanup_normalizes_whitespace() -> None:
    service = OCRService(output_dir="output-test")
    try:
        assert service._cleanup_text("a   b\r\nc\n\n", preserve_layout=False) == "a b c"
    finally:
        service.close()



def test_cleanup_keeps_rn_inside_words() -> None:
    service = OCRService(output_dir="output-test")
    try:
        assert service._cleanup_text("return learn", preserve_layout=False) == "return learn"
    finally:
        service.close()
