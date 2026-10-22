"""Processamento contínuo para streaming (serviço `serve`).

Um worker por câmera, com YOLO/ByteTrack PRÓPRIOS (nada compartilhado). Todos os workers seguem um
relógio GLOBAL comum, então os vídeos permanecem alinhados pelo `cam_timestamp` (não pelo índice de
frame). Se a inferência não acompanha o tempo real, frames são descartados (nunca acumulados).

Cada frame publicado é um pacote binário:  [uint32 BE: tamanho do header][header JSON][JPEG]
O header traz as detecções (bbox em pixels do vídeo ORIGINAL) e metadados; o desenho é feito no cliente.
"""
from __future__ import annotations

import json
import logging
import struct
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .detect_track import _write_tracker_yaml, det_paths, read_detections, resolve_classes
from .videos import VideoInfo, VideoRegistry

log = logging.getLogger(__name__)


class SessionClock:
    """Relógio global em loop: g(t) percorre [start, end) e recomeça (loop += 1)."""

    def __init__(self, start: float, end: float, speed: float = 1.0):
        self.start, self.span, self.speed = start, max(end - start, 1e-6), speed
        self._wall0 = time.monotonic()

    def now(self):
        e = (time.monotonic() - self._wall0) * self.speed
        return self.start + (e % self.span), int(e // self.span)


class FrameHub:
    """Último pacote publicado por câmera (os clientes só recebem o mais recente: sem fila)."""

    def __init__(self):
        self.seq = 0
        self.packet: Optional[bytes] = None
        self.last_pub = 0.0
        self.cond = threading.Condition()

    def publish(self, packet: bytes):
        with self.cond:
            self.seq += 1
            self.packet = packet
            self.last_pub = time.monotonic()
            self.cond.notify_all()


class LiveSource:
    """YOLO + ByteTrack em tempo real para UMA câmera."""

    def __init__(self, cfg: dict):
        from ultralytics import YOLO
        d = cfg["detector"]
        if d.get("disable_mkldnn", True) and str(d["device"]).startswith("cpu"):
            import torch
            torch.backends.mkldnn.enabled = False
        self.d = d
        self.model = YOLO(d["model"])
        self.names = dict(self.model.names)
        self.classes = resolve_classes(d.get("classes"), self.names)
        self.tracker = _write_tracker_yaml(cfg, Path(cfg["output"]["dir"]))

    def reset(self):
        """Reinicia o ByteTrack (novo loop do vídeo). Nunca usar persist=False: o Ultralytics registra o
        callback com o valor da 1ª chamada e passaria a recriar o tracker a cada frame."""
        for t in getattr(self.model.predictor, "trackers", None) or []:
            t.reset()

    def detect(self, frame, frame_idx: int) -> List[dict]:
        res = self.model.track(frame, persist=True, tracker=self.tracker, conf=self.d["conf"],
                               iou=self.d["iou"], imgsz=self.d["imgsz"], device=self.d["device"],
                               classes=self.classes, half=self.d.get("half", False), verbose=False)[0]
        b = res.boxes
        out = []
        if b is not None and len(b) and b.id is not None:
            xyxy, conf = b.xyxy.cpu().numpy(), b.conf.cpu().numpy()
            cls, ids = b.cls.cpu().numpy().astype(int), b.id.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                out.append({"id": int(ids[i]), "cls": int(cls[i]), "name": self.names[int(cls[i])],
                            "conf": round(float(conf[i]), 3), "bbox": [round(float(v), 1) for v in xyxy[i]],
                            "gid": None, "uncertain": False})
        return out


# TensorRT/CUDA não toleram bem a 1ª execução (captura/alocação lazy) de vários engines em threads simultâneas
# ("CUDA error ... during capture"). Inicialização e inferência de GPU dos detectores são serializadas.
GPU_LOCK = threading.Lock()


class StepperSource:
    """Detector unificado (PyTorch/TensorRT/RF-DETR/D-FINE) + BYTETracker próprio da câmera."""

    def __init__(self, cfg: dict, fps: float):
        from .detect_track import _UnifiedStepper
        with GPU_LOCK:
            self.s = _UnifiedStepper(cfg, fps)
            self.s.det.detect(np.zeros((540, 960, 3), np.uint8), 0.99)     # aquece (aloca contexto/buffers)

    def reset(self):
        self.s.tracker.reset()

    def detect(self, frame, frame_idx: int) -> List[dict]:
        with GPU_LOCK:
            xyxy, conf, cls, ids = self.s.track(frame)
        return [{"id": int(ids[i]), "cls": int(cls[i]), "name": self.s.names[int(cls[i])],
                 "conf": round(float(conf[i]), 3), "bbox": [round(float(v), 1) for v in xyxy[i]],
                 "gid": None, "uncertain": False} for i in range(len(xyxy))]


class ReplaySource:
    """Reproduz detecções JÁ salvas (sem inferência). Inclui IDs globais se houver associação salva."""

    def __init__(self, cfg: dict, cam: str):
        p, _ = det_paths(cfg, cam)
        if not p.is_file():
            raise FileNotFoundError(f"sem detecções salvas para {cam}: {p} (rode `detect`)")
        gmap, unc = {}, set()
        gp = Path(cfg["output"]["dir"]) / "global_identities.json"
        if gp.is_file():
            for i in json.loads(gp.read_text(encoding="utf-8")):
                for u in i["members"]:
                    gmap[u] = i["gid"]
                    if i.get("uncertain"):
                        unc.add(u)
        self.by_frame = defaultdict(list)
        for r in read_detections(p):
            uid = f"{cam}:{r['track_id']}"
            self.by_frame[r["frame"]].append({
                "id": r["track_id"], "cls": r["cls_id"], "name": r["cls_name"], "conf": r["conf"],
                "bbox": r["bbox"], "gid": gmap.get(uid), "uncertain": uid in unc})

    def reset(self):
        pass

    def detect(self, frame, frame_idx):
        return self.by_frame.get(frame_idx, [])


class CameraWorker(threading.Thread):
    def __init__(self, info: VideoInfo, cfg: dict, clock: SessionClock, mode: str, assoc=None):
        super().__init__(daemon=True, name=f"cam-{info.cam_id}")
        self.info, self.cfg, self.clock, self.mode = info, cfg, clock, mode
        self.assoc = assoc
        self.source = None            # fonte de detecção/tracking desta câmera (usada por /api/runtime, só leitura)
        self.hub = FrameHub()
        self.stop_evt = threading.Event()
        self.error: Optional[str] = None
        self.phase = "starting"       # starting | running | waiting | error
        self.frames_published = 0
        self.frames_dropped = 0
        self.proc_fps = 0.0
        self.infer_ms = 0.0
        self.loop_id = 0
        self.last_frame_idx = -1

    # estado exposto pela API (somente o que o backend confirmou)
    def status(self) -> dict:
        i = self.info
        if not i.accessible:
            state = "unavailable"
        elif self.error:
            state = "error"
        elif self.frames_published == 0:
            state = "waiting" if self.phase == "waiting" else "starting"
        elif time.monotonic() - self.hub.last_pub > 3.0 and self.phase != "waiting":
            state = "stalled"
        else:
            state = "running" if self.phase != "waiting" else "waiting"
        return {"id": i.cam_id, "state": state, "error": self.error or (
                "; ".join(i.warnings) if not i.accessible else None),
                "width": i.width, "height": i.height, "fps": i.fps, "offset_s": i.offset,
                "frames_total": i.n_frames, "frame": self.last_frame_idx, "loop": self.loop_id,
                "processing_fps": round(self.proc_fps, 1), "inference_ms": round(self.infer_ms, 1),
                "frames_published": self.frames_published, "frames_dropped": self.frames_dropped,
                "mode": self.mode}

    def _open(self):
        cap = cv2.VideoCapture(self.info.path)
        if not cap.isOpened():
            raise IOError(f"não foi possível abrir {self.info.path}")
        return cap

    def run(self):
        info, sc = self.info, self.cfg["server"]
        if not info.accessible:
            self.error = "; ".join(info.warnings) or "vídeo inacessível"
            return
        try:
            if self.mode != "live":
                source = ReplaySource(self.cfg, info.cam_id)
            elif self.cfg["detector"].get("type", "ultralytics_track") != "ultralytics_track":
                source = StepperSource(self.cfg, info.fps)
            else:
                source = LiveSource(self.cfg)
        except Exception as e:  # modelo/arquivo ausente: erro visível, API continua de pé
            log.exception("%s: falha ao iniciar fonte de detecção", info.cam_id)
            self.error = f"{type(e).__name__}: {e}"
            return
        self.source = source
        cap, pos, cur_loop = None, -1, -1
        ema = None
        while not self.stop_evt.is_set():
            try:
                g, loop = self.clock.now()
                if loop != cur_loop or cap is None:
                    if cap:
                        cap.release()
                    cap, pos, cur_loop = self._open(), -1, loop
                    self.loop_id = loop
                    source.reset()
                fi = info.to_local_frame(g)
                if fi < 0 or fi >= info.n_frames:
                    self.phase = "waiting"
                    time.sleep(0.05)
                    continue
                if fi <= pos:
                    time.sleep(0.004)
                    continue
                while pos < fi - 1:               # descarta frames atrasados
                    if not cap.grab():
                        break
                    pos += 1
                    self.frames_dropped += 1
                ok, frame = cap.read()
                if not ok:
                    raise IOError(f"falha de leitura no frame {pos + 1}")
                pos += 1
                self.phase, self.error = "running", None
                t0 = time.perf_counter()
                dets = source.detect(frame, pos)
                if self.assoc is not None:
                    self.assoc.update(info.cam_id, pos, info.to_global(pos), dets, frame, loop)
                    self.assoc.annotate(info.cam_id, dets)
                self.infer_ms = (time.perf_counter() - t0) * 1000
                self._publish(frame, pos, g, dets, loop)
                now = time.perf_counter()
                dt = now - (ema or now)
                ema = now
                if dt > 0:
                    self.proc_fps = 0.8 * self.proc_fps + 0.2 * (1 / dt) if self.proc_fps else 1 / dt
            except Exception as e:
                log.warning("%s: %s", info.cam_id, e)
                self.error = str(e)
                if cap:
                    cap.release()
                cap = None
                time.sleep(1.0)
        if cap:
            cap.release()

    def _publish(self, frame, idx, g, dets, loop):
        sc = self.cfg["server"]
        h, w = frame.shape[:2]
        sw = min(int(sc["stream_width"]), w)
        small = cv2.resize(frame, (sw, int(h * sw / w)), interpolation=cv2.INTER_AREA) if sw < w else frame
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, int(sc["jpeg_quality"])])
        if not ok:
            raise IOError("falha ao codificar JPEG")
        header = json.dumps({
            "cam": self.info.cam_id, "seq": self.hub.seq + 1, "frame": idx, "loop": loop,
            "t_global": round(g, 3), "t_local": round(idx / self.info.fps, 3),
            "width": w, "height": h, "stream_width": small.shape[1], "stream_height": small.shape[0],
            "proc_fps": round(self.proc_fps, 1), "inference_ms": round(self.infer_ms, 1),
            "mode": self.mode, "detections": dets}, separators=(",", ":")).encode()
        self.hub.publish(struct.pack(">I", len(header)) + header + jpg.tobytes())
        self.frames_published += 1
        self.last_frame_idx = idx


class LiveService:
    def __init__(self, cfg: dict, registry: VideoRegistry, mode: Optional[str] = None):
        self.cfg, self.registry = cfg, registry
        self.mode = mode or cfg["server"]["mode"]
        if self.mode not in ("live", "replay"):
            raise ValueError(f"server.mode inválido: {self.mode!r}")
        tl = registry.common_timeline() or (0.0, 1.0)
        self.clock = SessionClock(tl[0], tl[1] + 0.1, float(cfg["server"]["speed"]))
        self.assoc, self.assoc_note = self._build_assoc()
        self.workers: Dict[str, CameraWorker] = {
            c: CameraWorker(registry[c], cfg, self.clock, self.mode, self.assoc) for c in registry.cam_ids}
        self.started = time.time()

    def _build_assoc(self):
        """Associação ao vivo só no modo live e se houver cue decisivo (Re-ID e/ou homografias)."""
        sc = self.cfg["server"].get("live_association", {})
        if self.mode != "live" or not sc.get("enabled", False):
            return None, "desabilitada (server.live_association.enabled=false ou modo replay)"
        from .live_assoc import LiveAssociator
        from .reid import ReIDUnavailable, build_reid
        try:
            ext = build_reid(self.cfg)
        except ReIDUnavailable as e:
            ext, note = None, f"Re-ID indisponível: {e}"
            log.error(note)
        if ext is None and not self.cfg["spatial"].get("homographies") and not self.cfg["spatial"].get("transitions"):
            return None, "sem Re-ID nem informação espacial: nada decisivo para associar"
        return LiveAssociator(self.cfg, self.registry, ext), "ativa"

    def start(self):
        if self.assoc:
            self.assoc.start()
        for w in self.workers.values():
            w.start()

    def stop(self):
        if self.assoc:
            self.assoc.stop()
        for w in self.workers.values():
            w.stop_evt.set()
