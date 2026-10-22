"""Associação global incremental para o modo ao vivo (`mtmc serve --serve-mode live`).

Reaproveita EXATAMENTE a lógica offline (`association.associate`): a cada `interval_s` os tracks ativos
de todas as câmeras viram `Tracklet`s parciais (com embedding de Re-ID dos melhores recortes), e a
associação é recalculada. Como `associate` renumera as identidades a cada execução, aqui há uma camada
de IDs ESTÁVEIS: um cluster herda o ID global já mostrado a seus membros (o menor, se vários).

Só tracklets em clusters com >= 2 câmeras recebem `gid` visível (`linked=True`); os demais seguem sem
ID global no vídeo (identidade própria, sem associação). Tudo é reiniciado quando o vídeo reinicia (loop),
pois o ByteTrack também reinicia.

Limites conhecidos: o tracking ao vivo descarta frames para acompanhar o tempo real, então os tracks são
mais fragmentados que na avaliação offline; as métricas medidas com GT (README) são do pipeline offline.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .association import associate
from .reid import ReIDExtractor
from .tracklets import Tracklet

log = logging.getLogger(__name__)

MAX_HISTORY = 400      # observações guardadas por track (limita memória)


class _Track:
    def __init__(self, cam: str, local_id: int):
        self.cam, self.local_id = cam, local_id
        self.frames: List[int] = []
        self.tg: List[float] = []
        self.boxes: List[list] = []
        self.confs: List[float] = []
        self.votes: Counter = Counter()
        self.n_obs = 0
        self.embs: List[Tuple[float, np.ndarray]] = []     # (score, embedding), melhores K
        self.pending: List[Tuple[float, np.ndarray]] = []  # (score, recorte BGR) ainda sem embedding
        self.last_tg = 0.0

    @property
    def uid(self) -> str:
        return f"{self.cam}:{self.local_id}"


class LiveAssociator:
    def __init__(self, cfg: dict, registry, extractor: Optional[ReIDExtractor]):
        sc = cfg["server"].get("live_association", {})
        self.cfg, self.registry, self.ext = cfg, registry, extractor
        self.interval = float(sc.get("interval_s", 2.0))
        self.retention = float(sc.get("retention_s", 60.0))
        self.max_crops = int(sc.get("max_crops", 6))
        self.min_side = int(sc.get("min_crop_side", 40))
        self.lock = threading.RLock()
        self.tracks: Dict[str, _Track] = {}
        self.assign: Dict[str, dict] = {}       # uid -> {gid, linked, uncertain}  (troca atômica)
        self.stable: Dict[str, str] = {}        # uid -> gid estável já atribuído
        self.next_id = 1
        self.loop = 0
        self.stats = {"runs": 0, "last_run_ms": 0.0, "active_tracks": 0, "linked_identities": 0, "last_error": None}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="live-assoc")

    # ------------------------------------------------------------------ ciclo de vida
    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def reset(self, loop: int):
        with self.lock:
            if loop == self.loop:
                return
            self.loop = loop
            self.tracks.clear()
            self.assign = {}
            self.stable.clear()
            self.next_id = 1
            log.info("associador ao vivo reiniciado (loop %d)", loop)

    # ------------------------------------------------------------------ chamado pelos workers
    def update(self, cam: str, frame_idx: int, tg: float, dets: List[dict], frame, loop: int):
        if loop != self.loop:
            self.reset(loop)
        h, w = frame.shape[:2]
        with self.lock:
            for d in dets:
                t = self.tracks.get(f"{cam}:{d['id']}")
                if t is None:
                    t = self.tracks[f"{cam}:{d['id']}"] = _Track(cam, d["id"])
                t.frames.append(frame_idx); t.tg.append(tg); t.boxes.append(d["bbox"]); t.confs.append(d["conf"])
                t.votes[d["name"]] += d["conf"]
                t.n_obs += 1
                t.last_tg = tg
                if len(t.frames) > MAX_HISTORY:
                    for L in (t.frames, t.tg, t.boxes, t.confs):
                        del L[0]
                self._maybe_crop(t, d, frame, w, h)

    def _maybe_crop(self, t: _Track, d: dict, frame, w: int, h: int):
        if self.ext is None or t.n_obs % 3 != 1:
            return
        x1, y1, x2, y2 = (int(round(v)) for v in d["bbox"])
        if x2 - x1 < self.min_side or y2 - y1 < self.min_side:
            return
        if x1 < 3 or y1 < 3 or x2 > w - 3 or y2 > h - 3:            # caixas cortadas pela borda: embedding ruim
            return
        score = d["conf"] * (x2 - x1) * (y2 - y1)
        worst = min([s for s, _ in t.embs] + [s for s, _ in t.pending], default=-1.0) \
            if len(t.embs) + len(t.pending) >= self.max_crops else -1.0
        if score > worst and len(t.pending) < 3:
            t.pending.append((score, frame[max(y1, 0):y2, max(x1, 0):x2].copy()))

    def annotate(self, cam: str, dets: List[dict]) -> List[dict]:
        a = self.assign
        for d in dets:
            info = a.get(f"{cam}:{d['id']}")
            if info and info["linked"]:
                d["gid"], d["uncertain"] = info["gid"], info["uncertain"]
        return dets

    # ------------------------------------------------------------------ thread periódica
    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self._step()
                self.stats["last_error"] = None
            except Exception as e:  # nunca derruba o servidor
                log.exception("associação ao vivo falhou")
                self.stats["last_error"] = f"{type(e).__name__}: {e}"

    def _snapshot(self):
        with self.lock:
            now = max((t.last_tg for t in self.tracks.values()), default=0.0)
            for uid in [u for u, t in self.tracks.items() if now - t.last_tg > self.retention]:
                del self.tracks[uid]
            pend = [(t, list(t.pending)) for t in self.tracks.values() if t.pending]
            for t, _ in pend:
                t.pending.clear()
            return pend

    def _embed_pending(self, pend):
        if not pend or self.ext is None:
            return
        crops = [c for _, items in pend for _, c in items]
        emb = self.ext.embed(crops)
        k = 0
        with self.lock:
            for t, items in pend:
                for score, _ in items:
                    t.embs.append((score, emb[k])); k += 1
                t.embs.sort(key=lambda x: -x[0])
                del t.embs[self.max_crops:]

    def _tracklets(self) -> List[Tracklet]:
        acfg = self.cfg["association"]
        out = []
        with self.lock:
            for t in self.tracks.values():
                if t.n_obs < acfg["min_tracklet_frames"] or not t.frames:
                    continue
                info = self.registry[t.cam]
                cls = max(t.votes, key=t.votes.get)
                boxes = [list(b) for b in t.boxes]
                emb = None
                if t.embs:
                    e = np.mean([v for _, v in t.embs], axis=0)
                    emb = (e / max(np.linalg.norm(e), 1e-12)).tolist()
                out.append(Tracklet(
                    uid=t.uid, cam=t.cam, local_id=t.local_id, cls_name=cls,
                    cls_conf=t.votes[cls] / max(sum(t.votes.values()), 1e-9),
                    first_frame=t.frames[0], last_frame=t.frames[-1],
                    t_local_first=t.frames[0] / info.fps, t_local_last=t.frames[-1] / info.fps,
                    t_global_first=t.tg[0], t_global_last=t.tg[-1], n_obs=len(t.frames),
                    frames=list(t.frames), boxes=boxes,
                    centers=[[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes],
                    confs=list(t.confs), mean_conf=float(np.mean(t.confs)), embedding=emb))
        return out

    def _step(self):
        t0 = time.perf_counter()
        self._embed_pending(self._snapshot())
        tl = self._tracklets()
        self.stats["active_tracks"] = len(tl)
        if len(tl) < 2:
            return
        identities, _ = associate(tl, self.cfg)
        new_assign, linked_n = self._stabilize(identities)
        self.assign = new_assign                          # troca atômica; workers só leem
        self.stats.update(runs=self.stats["runs"] + 1, linked_identities=linked_n,
                          last_run_ms=round((time.perf_counter() - t0) * 1000, 1))

    # ------------------------------------------------------------------ IDs estáveis
    def _stabilize(self, identities) -> Tuple[Dict[str, dict], int]:
        by_uid = {t.uid: t for t in self.tracks.values()}
        out: Dict[str, dict] = {}
        used = set()
        linked_n = 0
        # clusters multi-câmera, os maiores primeiro (herdam IDs com mais "votos")
        for ident in sorted(identities, key=lambda i: -len(i.members)):
            cams = {by_uid[u].cam for u in ident.members if u in by_uid}
            if len(cams) < 2:
                continue
            prev = Counter(self.stable[u] for u in ident.members if u in self.stable and self.stable[u] not in used)
            gid = None
            if prev:
                gid = min(prev, key=lambda g: (-prev[g], int(g[1:])))     # o mais "votado"; empate: o mais antigo
            else:
                gid = f"G{self.next_id:04d}"
                self.next_id += 1
            used.add(gid)
            linked_n += 1
            for u in ident.members:
                self.stable[u] = gid
                out[u] = {"gid": gid, "linked": True, "uncertain": ident.uncertain}
        # tracklets sem associação: descartam o ID estável antigo (se o cluster se desfez)
        for u in list(self.stable):
            if u not in out:
                del self.stable[u]
        return out, linked_n
