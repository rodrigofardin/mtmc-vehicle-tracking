"""API HTTP/WebSocket para o frontend. Camada fina sobre `live.LiveService`.

GET  /api/health                      estado do serviço
GET  /api/cameras                     lista dinâmica (estado confirmado pelo backend)
GET  /api/cameras/{id}                uma câmera
GET  /api/cameras/{id}/snapshot.jpg   último frame (sem overlays)
WS   /api/cameras/{id}/ws             pacotes binários [u32 BE hdr_len][header JSON][JPEG]
"""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from .config import load_config
from .live import LiveService
from .videos import VideoRegistry


def create_app(cfg: dict, mode: str | None = None) -> FastAPI:
    registry = VideoRegistry.from_config(cfg)
    svc = LiveService(cfg, registry, mode)

    @asynccontextmanager
    async def lifespan(_):
        svc.start()
        yield
        svc.stop()

    app = FastAPI(title="MTMC streaming API", lifespan=lifespan)
    app.state.service = svc
    clients = {"n": 0}

    @app.get("/api/health")
    def health():
        return {"status": "ok", "mode": svc.mode, "environment": cfg["server"]["environment"],
                "device": cfg["detector"]["device"] if svc.mode == "live" else "none (replay)",
                "uptime_s": round(time.time() - svc.started, 1), "cameras": len(svc.workers),
                "ws_clients": clients["n"], "notes": registry.notes,
                "association": {"status": svc.assoc_note, **(svc.assoc.stats if svc.assoc else {})}}

    @app.get("/api/runtime")
    def runtime():
        """Modelos, parâmetros e estado de configuração em execução. SOMENTE LEITURA; não toca nos streams."""
        from .runtime_info import build_runtime
        return build_runtime(svc, cfg, svc.started)

    @app.get("/api/cameras")
    def cameras():
        return [w.status() for w in svc.workers.values()]

    def _worker(cam_id: str):
        w = svc.workers.get(cam_id)
        if w is None:
            raise HTTPException(404, f"câmera desconhecida: {cam_id}")
        return w

    @app.get("/api/cameras/{cam_id}")
    def camera(cam_id: str):
        return _worker(cam_id).status()

    @app.get("/api/cameras/{cam_id}/snapshot.jpg")
    def snapshot(cam_id: str):
        w = _worker(cam_id)
        pkt = w.hub.packet
        if pkt is None:
            raise HTTPException(503, "ainda sem frames")
        n = int.from_bytes(pkt[:4], "big")
        return Response(pkt[4 + n:], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.websocket("/api/cameras/{cam_id}/ws")
    async def ws(sock: WebSocket, cam_id: str):
        w = svc.workers.get(cam_id)
        if w is None:
            await sock.close(code=4404)
            return
        await sock.accept()
        clients["n"] += 1
        last = 0
        try:
            while True:
                if w.hub.seq != last and w.hub.packet is not None:
                    last = w.hub.seq
                    await sock.send_bytes(w.hub.packet)     # só o frame mais recente: sem fila
                else:
                    await asyncio.sleep(0.01)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            clients["n"] -= 1

    return app


def app_from_env() -> FastAPI:
    """Entrypoint para `uvicorn mtmc.server:app_from_env --factory`."""
    return create_app(load_config(os.environ.get("MTMC_CONFIG", "configs/s01_gpu.yaml")),
                      os.environ.get("MTMC_MODE") or None)
