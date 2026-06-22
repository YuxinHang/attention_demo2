from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import VisionEngine


engine = VisionEngine()
ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.start()
    yield
    engine.stop()


app = FastAPI(title="Privacy-first TV Attention Demo", lifespan=lifespan, docs_url=None, redoc_url=None)


class SourceRequest(BaseModel):
    kind: str
    value: str | int | None = None


@app.get("/api/health")
def health():
    return engine.metrics()


@app.post("/api/source")
def source(request: SourceRequest):
    try:
        engine.set_source(request.kind, request.value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.post("/api/calibration/start")
def calibration_start():
    engine.start_calibration()
    return {"ok": True}


@app.post("/api/calibration/skip")
def calibration_skip():
    engine.skip_calibration()
    return {"ok": True}


@app.post("/api/session/reset")
def session_reset():
    engine.reset_session()
    return {"ok": True}


@app.get("/stream")
async def stream():
    async def frames():
        while True:
            jpeg = engine.jpeg()
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(0.04)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/metrics")
async def metrics_socket(socket: WebSocket):
    await socket.accept()
    try:
        while True:
            await socket.send_json(engine.metrics())
            await asyncio.sleep(0.25)
    except (WebSocketDisconnect, RuntimeError):
        return


if STATIC.is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/{path:path}")
    def frontend(path: str):
        candidate = STATIC / path
        return FileResponse(candidate if candidate.is_file() else STATIC / "index.html")
else:
    @app.get("/")
    def missing_frontend():
        return JSONResponse({"message": "Frontend is not built. Run the setup script."}, status_code=503)

