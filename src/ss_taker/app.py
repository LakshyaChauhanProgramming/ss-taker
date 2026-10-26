from __future__ import annotations

import json
import logging
import os
import secrets
import time
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from .ocr import OCRService, OCRResult, ocr_engine_name

LOGGER = logging.getLogger("ss_taker")
app = FastAPI(title="ss-taker")
service: OCRService | None = None


def require_token(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> None:
    """Every endpoint except /health reads the screen or describes it, so gate
    them behind SS_TAKER_TOKEN whenever one is configured."""
    expected = os.environ.get("SS_TAKER_TOKEN")
    if not expected:
        return
    supplied = x_api_key
    if supplied is None and authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid API token")


protected = [Depends(require_token)]


class PickSSRequest(BaseModel):
    region: Optional[tuple[int, int, int, int]] = Field(
        default=None, description="Optional x, y, width, height"
    )
    lang: str = "eng"
    psm: int = 6
    whitelist: Optional[str] = None
    preserve_layout: bool = False
    delay_seconds: float = Field(default=0, ge=0, le=30, description="Seconds to wait before capture")
    mode: str = Field(
        default="balanced",
        description=(
            "OCR mode. fast reads the raw capture only; balanced adds a preprocessed pass when the raw "
            "read is weak; quality_hd_ultra runs all HD candidates."
        ),
    )


class PickSSResponse(BaseModel):
    mode: str
    text: str
    chosen_text: str
    candidate_texts: list[dict[str, object]]
    capture_stats: dict[str, object]
    cached: bool
    image_hash: str
    timings_ms: dict[str, float]
    json_path: str


class HealthDisplayResponse(BaseModel):
    display: str | None
    connected: bool


class HealthX11Response(BaseModel):
    socket: bool
    permissions: bool


class HealthBackendResponse(BaseModel):
    backend: str | None


class HealthOCRResponse(BaseModel):
    tesseract: bool
    languages: list[str]
    engine: str


class DebugResponse(BaseModel):
    display: str | None
    wayland_display: str | None
    x11: bool
    backend: str | None
    ocr: bool
    version: str


class MetricsResponse(BaseModel):
    screenshots: int
    ocr_requests: int
    errors: int
    avg_latency_ms: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/display", response_model=HealthDisplayResponse, dependencies=protected)
def health_display() -> HealthDisplayResponse:
    svc = _get_service()
    diag = svc.diagnostics()
    return HealthDisplayResponse(display=diag.display, connected=diag.x11_connected)


@app.get("/health/x11", response_model=HealthX11Response, dependencies=protected)
def health_x11() -> HealthX11Response:
    svc = _get_service()
    diag = svc.diagnostics()
    return HealthX11Response(socket=diag.x11_socket, permissions=diag.x11_connected)


@app.get("/health/backend", response_model=HealthBackendResponse, dependencies=protected)
def health_backend() -> HealthBackendResponse:
    svc = _get_service()
    return HealthBackendResponse(backend=svc.capture_backend_name())


@app.get("/health/ocr", response_model=HealthOCRResponse, dependencies=protected)
def health_ocr() -> HealthOCRResponse:
    svc = _get_service()
    diag = svc.diagnostics()
    return HealthOCRResponse(tesseract=diag.tesseract, languages=diag.languages, engine=ocr_engine_name())


@app.get("/debug", response_model=DebugResponse, dependencies=protected)
def debug() -> DebugResponse:
    svc = _get_service()
    diag = svc.diagnostics()
    return DebugResponse(
        display=diag.display,
        wayland_display=diag.wayland_display,
        x11=diag.x11_connected,
        backend=diag.backend,
        ocr=diag.tesseract,
        version=app.version or "1.2.0",
    )


@app.get("/metrics", response_model=MetricsResponse, dependencies=protected)
def metrics() -> MetricsResponse:
    svc = _get_service()
    return MetricsResponse(
        screenshots=svc.stats["screenshots"],
        ocr_requests=svc.stats["ocr_requests"],
        errors=svc.stats["errors"],
        avg_latency_ms=svc.stats["avg_latency_ms"],
    )


@app.on_event("shutdown")
def _shutdown() -> None:
    global service
    if service is not None:
        service.close()
        service = None


@app.post("/pick-ss", response_model=PickSSResponse, dependencies=protected)
def pick_ss(req: PickSSRequest) -> Response:
    svc = _get_service()
    started = time.perf_counter()
    try:
        if req.delay_seconds:
            time.sleep(req.delay_seconds)
        result: OCRResult = svc.capture_and_ocr(
            region=req.region,
            lang=req.lang,
            psm=req.psm,
            whitelist=req.whitelist,
            preserve_layout=req.preserve_layout,
            mode=req.mode,
        )
    except RuntimeError as exc:
        svc.stats["errors"] += 1
        _raise_screenshot_error(str(exc))

    total_ms = (time.perf_counter() - started) * 1000
    LOGGER.info(
        "[display] %s [x11] %s [backend] %s [ocr] ok total_ms=%.1f",
        svc.diagnostics().display,
        "connected" if svc.diagnostics().x11_connected else "disconnected",
        svc.capture_backend_name(),
        total_ms,
    )
    svc.stats["screenshots"] += 1
    svc.stats["ocr_requests"] += 1
    svc.stats["total_latency_ms"] += total_ms
    svc.stats["avg_latency_ms"] = svc.stats["total_latency_ms"] / max(1, svc.stats["ocr_requests"])
    payload = PickSSResponse(
        mode=result.mode,
        text=result.text,
        chosen_text=result.text,
        candidate_texts=[
            {
                "variant": candidate.get("variant"),
                "mean_confidence": candidate.get("mean_confidence"),
                "score": candidate.get("score"),
                "text_length": len(str(candidate.get("cleaned_text", ""))),
                "width": candidate.get("width"),
                "height": candidate.get("height"),
            }
            for candidate in result.candidate_texts
        ],
        capture_stats=result.capture_stats,
        cached=result.cached,
        image_hash=result.image_hash,
        timings_ms=result.timings_ms,
        json_path=str(result.output_path),
    )
    return Response(content=payload.model_dump_json(indent=2), media_type="application/json")


def create_app() -> FastAPI:
    return app


def _get_service() -> OCRService:
    global service
    if service is None:
        service = OCRService()
    return service


def _raise_screenshot_error(message: str) -> None:
    if "DISPLAY is not set" in message:
        raise HTTPException(status_code=400, detail=message)
    if "X11 socket not mounted" in message:
        raise HTTPException(status_code=400, detail=message)
    if "Cannot connect to display" in message:
        raise HTTPException(status_code=403, detail=message)
    if "Screenshot backend unavailable" in message:
        raise HTTPException(status_code=503, detail=message)
    if "OCR unavailable" in message:
        raise HTTPException(status_code=503, detail=message)
    raise HTTPException(status_code=503, detail=message)
