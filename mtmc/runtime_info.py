"""Descrição SOMENTE LEITURA do que está realmente em execução (modelos, tracker, Re-ID, parâmetros).

Princípios:
  * Nada aqui altera inferência, tracker ou configuração; só lê atributos de objetos já carregados.
  * "configurado" = valor no arquivo YAML relido do disco agora; "aplicado" = valor lido do OBJETO em execução
    (ex.: `predictor.args`, argumentos do BYTETracker instanciado). Se não há como ler do runtime => `applied=None`
    e `status="unconfirmed"` (a UI mostra "não confirmado pelo runtime").
  * Formato/versão do modelo vêm de metadados do runtime (AutoBackend.format, ICudaEngine, sessão ONNX), nunca do
    nome do arquivo.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1
_UNCONF = object()


# ----------------------------------------------------------------------------- versões do runtime
def runtime_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name, mod in (("python", None), ("torch", "torch"), ("ultralytics", "ultralytics"), ("tensorrt", "tensorrt"),
                      ("onnxruntime", "onnxruntime"), ("opencv", "cv2"), ("numpy", "numpy")):
        try:
            if mod is None:
                import platform
                out[name] = platform.python_version()
            else:
                out[name] = str(__import__(mod).__version__)
        except Exception:  # noqa: BLE001 - módulo ausente neste ambiente
            out[name] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        out["cuda"] = out["gpu"] = None
    return out


# ----------------------------------------------------------------------------- inspeção de objetos
def _engine_io(engine) -> Dict[str, Any]:
    """Tensores de E/S de um ICudaEngine (formas e tipos reais)."""
    import tensorrt as trt
    ins, outs = [], []
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        item = {"name": n, "shape": [int(v) for v in engine.get_tensor_shape(n)],
                "dtype": str(engine.get_tensor_dtype(n)).split(".")[-1]}
        (ins if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT else outs).append(item)
    return {"inputs": ins, "outputs": outs}


def describe_ultralytics(model) -> Dict[str, Any]:
    """`model` = objeto `ultralytics.YOLO` já usado em pelo menos uma inferência (senão: carregamento pendente)."""
    d: Dict[str, Any] = {"loaded": False}
    pred = getattr(model, "predictor", None)
    be = getattr(pred, "model", None) if pred is not None else None
    d["task"] = getattr(model, "task", None)
    if be is None:
        d["load_state"] = "carregado (aguardando 1ª inferência para expor o backend)"
        d["names"] = dict(getattr(model, "names", {}) or {})
        return d
    d["loaded"] = True
    d["load_state"] = "carregado"
    fmt = getattr(be, "format", None)
    d["format"] = {"pt": "PyTorch", "engine": "TensorRT (engine)", "onnx": "ONNX", "torchscript": "TorchScript"}.get(fmt, fmt)
    d["backend_class"] = type(getattr(be, "backend", be)).__name__
    d["device"] = str(getattr(be, "device", "") or "") or None
    d["names"] = {int(k): v for k, v in dict(getattr(be, "names", {}) or {}).items()}
    d["stride"] = getattr(be, "stride", None)
    d["end2end"] = bool(getattr(be, "end2end", False))
    args = vars(pred.args) if getattr(pred, "args", None) is not None else {}
    d["applied"] = {k: args.get(k) for k in ("conf", "iou", "imgsz", "max_det", "half", "classes", "device", "agnostic_nms")}
    eng = getattr(getattr(be, "backend", None), "model", None)
    if eng is not None and type(eng).__name__ == "ICudaEngine":
        try:
            d["io"] = _engine_io(eng)
            d["input_shape"] = d["io"]["inputs"][0]["shape"]
            d["io_dtype"] = d["io"]["inputs"][0]["dtype"]
        except Exception as e:  # noqa: BLE001
            d["io_error"] = f"{type(e).__name__}: {e}"
        meta = getattr(getattr(be, "backend", None), "metadata", None) or getattr(be, "metadata", None)
        if isinstance(meta, dict):
            d["engine_metadata"] = {k: meta.get(k) for k in ("imgsz", "batch", "args", "version", "description", "date") if k in meta}
    elif fmt == "pt":
        try:
            d["params"] = int(sum(p.numel() for p in be.backend.model.parameters()))
        except Exception:  # noqa: BLE001
            pass
        d["io_dtype"] = "float16" if getattr(be, "fp16", False) else "float32"
    return d


def describe_detr(det) -> Dict[str, Any]:
    """`DetrTRTDetector`: engine TensorRT carregado por `TRTEngine` (formato confirmado pelo próprio carregador)."""
    d: Dict[str, Any] = {"loaded": True, "load_state": "carregado", "task": "detect", "format": "TensorRT (engine)",
                         "backend_class": "TRTEngine"}
    try:
        import torch
        d["device"] = f"cuda:{torch.cuda.current_device()}"
    except Exception:  # noqa: BLE001
        d["device"] = None
    try:
        d["io"] = _engine_io(det.eng.engine)
        d["input_shape"] = d["io"]["inputs"][0]["shape"]
        d["io_dtype"] = d["io"]["inputs"][0]["dtype"]
    except Exception as e:  # noqa: BLE001
        d["io_error"] = f"{type(e).__name__}: {e}"
    from .detectors import COCO80
    d["names"] = {int(c): COCO80.get(int(c), str(c)) for c in det.cls_map.values()}      # só as classes de veículo mapeadas
    d["label_space_size"] = int(next(iter(det._masks.values())).shape[0]) if getattr(det, "_masks", None) else None
    d["applied"] = {"top_k": det.topk, "nms": "não se aplica (DETR sem NMS)"}
    return d


def describe_tracker(trk) -> Dict[str, Any]:
    """`trk` = instância de BYTETracker (a que realmente roda)."""
    if trk is None:
        return {"loaded": False}
    a = getattr(trk, "args", None)
    return {"loaded": True, "type": type(trk).__name__, "module": type(trk).__module__,
            "applied": {k: getattr(a, k, None) for k in ("track_high_thresh", "track_low_thresh", "new_track_thresh",
                                                          "track_buffer", "match_thresh", "fuse_score")} if a is not None else {},
            "max_time_lost": next((getattr(trk, n) for n in ("max_time_lost", "max_frames_lost") if hasattr(trk, n)), None),
            "frame_rate": getattr(trk, "frame_rate", None)}


def describe_reid(ext) -> Optional[Dict[str, Any]]:
    if ext is None:
        return None
    d: Dict[str, Any] = {"loaded": True, "load_state": "carregado", "task": "re-identificação (embedding)",
                         "class": type(ext).__name__}
    sess = getattr(ext, "sess", None)
    if sess is not None:                                   # onnxruntime
        d["format"] = "ONNX"
        d["providers"] = list(sess.get_providers())
        d["device"] = "cuda" if "CUDAExecutionProvider" in d["providers"] else "cpu"
        i, o = sess.get_inputs()[0], sess.get_outputs()[0]
        d["io"] = {"inputs": [{"name": i.name, "shape": [s if isinstance(s, int) else str(s) for s in i.shape], "dtype": i.type}],
                   "outputs": [{"name": o.name, "shape": [s if isinstance(s, int) else str(s) for s in o.shape], "dtype": o.type}]}
        d["input_size_hw"] = list(getattr(ext, "size", [])) or None
        pre = getattr(ext, "pre", None)
        if pre is not None:
            d["preprocess"] = {"color": pre.color, "scale": pre.scale, "mean": [round(float(x), 4) for x in pre.mean],
                               "std": [round(float(x), 4) for x in pre.std]}
        try:
            import onnxruntime
            d["runtime_version"] = f"onnxruntime {onnxruntime.__version__}"
        except Exception:  # noqa: BLE001
            pass
    return d


def describe_source(source) -> Dict[str, Any]:
    """Descreve a fonte de detecção/tracking de UMA câmera (a instância real que ela usa)."""
    if source is None:
        return {"kind": None, "loaded": False}
    cls = type(source).__name__
    if cls == "ReplaySource":
        return {"kind": "replay", "loaded": True, "detector": None, "tracker": None,
                "note": "reproduz detecções salvas; nenhum modelo é executado"}
    if cls == "LiveSource":                               # YOLO + model.track (caminho original)
        trk = (getattr(getattr(source.model, "predictor", None), "trackers", None) or [None])[0]
        return {"kind": "ultralytics_track", "loaded": True, "detector": describe_ultralytics(source.model),
                "tracker": describe_tracker(trk)}
    if cls == "StepperSource":                             # detector unificado + BYTETracker próprio
        s = source.s
        det = s.det
        dd = describe_detr(det) if type(det).__name__ == "DetrTRTDetector" else describe_ultralytics(det.model)
        dd["applied_conf_arg"] = s.conf                    # confiança mínima passada a cada chamada
        dd["name"] = getattr(det, "name", None)
        dd["detector_class"] = type(det).__name__
        return {"kind": "unified", "loaded": True, "detector": dd, "tracker": describe_tracker(getattr(s.tracker, "t", None)),
                "allowed_class_ids": sorted(s.allowed) if s.allowed is not None else None}
    return {"kind": cls, "loaded": True}


# ----------------------------------------------------------------------------- configuração (arquivo x runtime)
def _flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if str(k).startswith("_"):
                continue
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(d, list) and d and all(isinstance(x, dict) for x in d):
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = d
    return out


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def config_state(runtime_cfg: dict, path: Optional[str]) -> Dict[str, Any]:
    """Compara a configuração em memória (aplicada na partida) com o arquivo relido agora."""
    st: Dict[str, Any] = {"file": path, "file_mtime": None, "sha256": None, "changed_on_disk": None, "diff": []}
    if not path or not os.path.isfile(path):
        return st
    st["file_mtime"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(path)))
    st["sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    try:
        from .config import load_config
        disk = _flatten(load_config(path))
        mem = _flatten(runtime_cfg)
        diff = [{"key": k, "running": mem.get(k), "on_disk": disk.get(k)}
                for k in sorted(set(mem) | set(disk)) if not _same(mem.get(k), disk.get(k))]
        st["diff"], st["changed_on_disk"] = diff[:200], bool(diff)
    except Exception as e:  # noqa: BLE001
        st["error"] = f"{type(e).__name__}: {e}"
    return st


_DISK_CACHE: Dict[str, Any] = {"t": 0.0, "path": None, "cfg": None}


def disk_config(runtime_cfg: dict, ttl: float = 2.0) -> dict:
    """Configuração relida do arquivo (valor "configurado"); sem arquivo ou com erro, cai na cópia em memória."""
    path = runtime_cfg.get("_config_path")
    if not path or not os.path.isfile(path):
        return runtime_cfg
    now = time.time()
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)           # o arquivo mudou => não usa o cache
    if _DISK_CACHE["path"] == key and now - _DISK_CACHE["t"] < ttl and _DISK_CACHE["cfg"] is not None:
        return _DISK_CACHE["cfg"]
    try:
        from .config import load_config
        cfg = load_config(path)
    except Exception:  # noqa: BLE001 - arquivo em edição/inválido: mostra o que está em memória
        return runtime_cfg
    _DISK_CACHE.update(t=now, path=key, cfg=cfg)
    return cfg


def config_revision(cfg: dict) -> str:
    blob = json.dumps(_flatten(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# ----------------------------------------------------------------------------- tabela de parâmetros
def _param(pid, label, category, scope, configured, applied=_UNCONF, *, desc="", source="config", unit=None,
           fixed_in_code=False):
    if applied is _UNCONF:
        status, applied_v = "unconfirmed", None
    else:
        applied_v = applied
        status = "confirmed" if _same(configured, applied) else "diverges"
    return {"id": pid, "label": label, "category": category, "scope": scope, "configured": configured,
            "applied": applied_v, "status": status, "source": source, "description": desc, "unit": unit,
            "editable": False, "fixed_in_code": fixed_in_code}


def build_runtime(svc, cfg: dict, started_at: float) -> Dict[str, Any]:
    descs = {c: describe_source(getattr(w, "source", None)) for c, w in svc.workers.items()}
    first = next((d for d in descs.values() if d.get("detector")), None)
    ass = svc.assoc
    reid = describe_reid(getattr(ass, "ext", None)) if ass is not None else None
    disk = disk_config(cfg)                    # "configurado" = arquivo relido; "aplicado" = objetos em execução
    dcfg, tcfg, acfg, scfg = disk["detector"], disk["tracker"], disk["association"], disk["server"]
    det = (first or {}).get("detector") or {}
    trk = (first or {}).get("tracker") or {}
    ap = det.get("applied") or {}
    ta = trk.get("applied") or {}
    kind = (first or {}).get("kind")
    unified = kind == "unified"
    params: List[Dict[str, Any]] = []
    add = params.append

    # --- Inferência
    cls_cfg = dcfg.get("classes")
    cfg_type = dcfg.get("type", "ultralytics_track")
    applied_type = _UNCONF
    if kind == "ultralytics_track":
        applied_type = "ultralytics_track"
    elif unified:
        dc = det.get("detector_class")
        applied_type = "ultralytics" if dc == "UltralyticsDetector" else (cfg_type if dc == "DetrTRTDetector" and cfg_type in ("rfdetr", "dfine") else dc)
    add(_param("detector.type", "Tipo de detector", "Inferência", "global", cfg_type,
               applied_type, desc="Caminho de inferência em uso (ultralytics_track = YOLO + model.track; unified = detector + BYTETracker próprio).",
               source="runtime" if kind else "config"))
    conf_applied = det.get("applied_conf_arg") if unified else ap.get("conf")
    add(_param("detector.conf", "Confiança mínima", "Inferência", "modelo", dcfg.get("conf"),
               conf_applied if conf_applied is not None else _UNCONF,
               desc="Detecções abaixo deste valor são descartadas antes do tracker.", source="runtime"))
    if det.get("applied", {}).get("nms") == "não se aplica (DETR sem NMS)":
        add(_param("detector.iou", "IoU da NMS", "Inferência", "modelo", "não se aplica", "não se aplica",
                   desc="DETRs não usam NMS.", source="runtime", fixed_in_code=True))
    else:
        add(_param("detector.iou", "IoU da NMS", "Inferência", "modelo", dcfg.get("iou") if not unified else 0.7,
                   ap.get("iou", _UNCONF), desc="Limiar de IoU da NMS." + (" No caminho unificado o valor é fixo no código (0,7)." if unified else ""),
                   source="runtime", fixed_in_code=unified))
    imgsz_applied = det.get("input_shape")
    cfg_imgsz = (dcfg.get("shape") or dcfg.get("size")) if dcfg.get("type") in ("rfdetr", "dfine") else dcfg.get("imgsz")
    add(_param("detector.imgsz", "Tamanho de entrada", "Inferência", "modelo", cfg_imgsz,
               ([imgsz_applied[2], imgsz_applied[3]] if imgsz_applied and len(imgsz_applied) == 4 else ap.get("imgsz", _UNCONF)),
               desc="Tamanho da imagem de entrada do detector (alto × largura em engines TensorRT).", source="runtime", unit="px"))
    maxdet = ap.get("max_det", ap.get("top_k", _UNCONF))
    add(_param("detector.max_det", "Máximo de detecções por frame", "Inferência", "modelo", None if maxdet is _UNCONF else maxdet, maxdet,
               desc="Não há parâmetro no YAML; o valor mostrado é o aplicado pelo runtime (padrão da biblioteca ou fixo no código).",
               source="runtime", fixed_in_code=True))
    names_applied = None
    if det.get("names") is not None and (cls_cfg is not None):
        names_applied = sorted(det["names"].values()) if not unified or det.get("applied", {}).get("nms") else None
    allowed = (first or {}).get("allowed_class_ids")
    if unified and allowed is not None:
        from .detectors import COCO80
        names_applied = sorted(COCO80.get(i, str(i)) for i in allowed)
    elif not unified and ap.get("classes") is not None and det.get("names"):
        names_applied = sorted(det["names"][i] for i in ap["classes"] if i in det["names"])
    add(_param("detector.classes", "Classes habilitadas", "Inferência", "modelo",
               sorted(map(str, cls_cfg)) if cls_cfg is not None else None,
               names_applied if names_applied is not None else _UNCONF,
               desc="Classes de veículo entregues ao tracker (as demais são descartadas).", source="runtime"))
    add(_param("detector.device", "Dispositivo", "Inferência", "modelo", dcfg.get("device"),
               (det.get("device") if det.get("device") else _UNCONF), desc="Dispositivo de execução do detector.", source="runtime"))
    add(_param("detector.trt", "Engine TensorRT", "Inferência", "modelo", bool(dcfg.get("trt", False)),
               (det.get("format", "").startswith("TensorRT") if det.get("format") else _UNCONF),
               desc="Se o detector carregado é um engine TensorRT (confirmado pelo formato real do backend).", source="runtime"))

    # --- Tracking
    add(_param("tracker.type", "Tipo de tracker", "Tracking", "câmera", trk.get("type"), trk.get("type", _UNCONF),
               desc="Fixo no código (BYTETracker do Ultralytics = ByteTrack); uma instância independente por câmera.",
               source="runtime", fixed_in_code=True))
    labels = {"track_high_thresh": ("Limiar de confiança alta (1ª associação)", "Detecções acima deste valor entram na 1ª associação."),
              "track_low_thresh": ("Limiar de confiança baixa (2ª associação)", "Detecções entre este valor e o alto são usadas na 2ª associação (efetivo só se a confiança mínima do detector for menor)."),
              "new_track_thresh": ("Limiar para iniciar track", "Confiança mínima para criar uma nova trilha."),
              "track_buffer": ("Buffer de frames (perda de track)", "Frames que uma trilha perdida é mantida antes de ser removida."),
              "match_thresh": ("Limiar de associação (IoU)", "Limiar de casamento entre trilhas e detecções."),
              "fuse_score": ("Fundir score na distância", "Combina a confiança da detecção com o IoU na associação.")}
    for k, (lab, d) in labels.items():
        add(_param(f"tracker.{k}", lab, "Tracking", "câmera", tcfg.get(k), ta.get(k, _UNCONF), desc=d, source="runtime"))
    add(_param("tracker.max_time_lost", "Máx. frames perdidos (derivado)", "Tracking", "câmera", trk.get("max_time_lost"),
               trk.get("max_time_lost") if trk.get("max_time_lost") is not None else _UNCONF, desc="Calculado pelo BYTETracker a partir do buffer e da taxa de quadros (somente leitura).",
               source="runtime", fixed_in_code=True, unit="frames"))

    # --- Processamento
    add(_param("server.speed", "Velocidade do relógio global", "Processamento", "global", scfg.get("speed"), svc.clock.speed,
               desc="1,0 = tempo real; vídeos reiniciam juntos em loop.", source="runtime", unit="x"))
    add(_param("server.stream_width", "Largura do stream (JPEG)", "Processamento", "global", scfg.get("stream_width"), scfg.get("stream_width"),
               desc="Largura máxima dos frames enviados ao navegador (as caixas vêm em pixels do vídeo original).", unit="px"))
    add(_param("server.jpeg_quality", "Qualidade JPEG do stream", "Processamento", "global", scfg.get("jpeg_quality"), scfg.get("jpeg_quality"),
               desc="Qualidade de compressão dos frames do stream."))
    bs = det.get("input_shape", [None])[0] if det.get("input_shape") else None
    add(_param("detector.batch", "Tamanho do lote", "Processamento", "modelo", bs, bs if bs is not None else _UNCONF,
               desc="Cada câmera infere um frame por vez; a GPU é serializada entre câmeras.", source="runtime", fixed_in_code=True))

    # --- Re-ID e associação
    rcfg = disk["reid"]
    add(_param("reid.model", "Modelo de Re-ID", "Re-ID", "modelo", rcfg.get("model"), _UNCONF if reid is None else rcfg.get("model"),
               desc="Arquivo carregado pelo extrator de Re-ID.", source="runtime" if reid else "config"))
    add(_param("reid.device", "Dispositivo do Re-ID", "Re-ID", "modelo", "cuda" if str(rcfg.get("device", "cpu")).startswith("cuda") else "cpu",
               reid.get("device") if reid and reid.get("device") else _UNCONF,
               desc="Dispositivo efetivo do onnxruntime" + (f" (provedores: {', '.join(reid['providers'])})." if reid and reid.get("providers") else "."),
               source="runtime"))
    live = scfg.get("live_association", {})
    for k, lab, d in (("interval_s", "Intervalo da associação ao vivo", "Segundos entre reavaliações das identidades globais."),
                      ("retention_s", "Retenção de tracks", "Tempo que um track inativo permanece disponível para associação."),
                      ("max_crops", "Recortes por track", "Máximo de recortes embutidos por track."),
                      ("min_crop_side", "Lado mínimo do recorte", "Recortes menores são ignorados.")):
        applied = getattr(ass, {"interval_s": "interval", "retention_s": "retention", "max_crops": "max_crops", "min_crop_side": "min_side"}[k], _UNCONF) if ass else _UNCONF
        add(_param(f"live_association.{k}", lab, "Associação", "global", live.get(k), applied, desc=d, source="runtime" if ass else "config"))
    for k, lab, d in (("appearance_threshold", "Limiar de aparência (cosseno)", "Cosseno mínimo do Re-ID para aceitar o par."),
                      ("appearance_threshold_transition", "Limiar de aparência em transições", "Se vazio, usa o limiar de aparência."),
                      ("appearance_scale", "Escala do cosseno [mín, máx]", "Mapeia o cosseno para score 0–1."),
                      ("spatial_threshold", "Limiar espacial", "Score mínimo do cue espacial."),
                      ("accept_score", "Score mínimo para aceitar", "Média ponderada dos cues disponíveis."),
                      ("ambiguity_margin", "Margem de ambiguidade", "Candidatos com score dentro da margem viram ambíguos."),
                      ("min_overlap_s", "Sobreposição temporal mínima", "Segundos de sobreposição para tratar como simultâneo."),
                      ("max_gap_s", "Intervalo máximo em transições", "Segundos entre saída e entrada."),
                      ("min_tracklet_frames", "Observações mínimas por tracklet", "Tracklets mais curtos não são associados."),
                      ("require_decisive_cue", "Exigir cue decisivo", "Sem aparência ou espacial, o par fica `insufficient_evidence`.")):
        applied = ass.cfg["association"].get(k, _UNCONF) if ass else _UNCONF
        add(_param(f"association.{k}", lab, "Associação", "global", acfg.get(k), applied, desc=d,
                   source="runtime" if ass else "config"))
    sp = cfg["spatial"]
    add(_param("spatial.max_ground_dist", "Distância máxima no plano de chão", "Associação", "global", sp.get("max_ground_dist"),
               ass.cfg["spatial"].get("max_ground_dist", _UNCONF) if ass else _UNCONF, desc="Distância (m) entre posições no plano comum.", unit="m",
               source="runtime" if ass else "config"))

    # --- Regras / regiões
    add(_param("spatial.zones", "Zonas (polígonos)", "Regras e regiões", "câmera", {c: len(z) for c, z in (sp.get("zones") or {}).items()},
               desc="Usadas só para rotular entrada/saída no pipeline OFFLINE; o associador ao vivo não as usa.", source="config"))
    add(_param("spatial.transitions", "Transições entre zonas", "Regras e regiões", "global", len(sp.get("transitions") or []),
               len(ass.cfg["spatial"].get("transitions") or []) if ass else _UNCONF,
               desc="Regras de transição configuradas (cue espacial por zonas).", source="runtime" if ass else "config"))
    add(_param("spatial.homographies", "Homografias por câmera", "Regras e regiões", "câmera", sorted((sp.get("homographies") or {}).keys()),
               sorted((ass.cfg["spatial"].get("homographies") or {}).keys()) if ass else _UNCONF,
               desc="Câmeras com homografia pixel→metros carregada.", source="runtime" if ass else "config"))
    add(_param("evaluation.roi", "ROI de avaliação", "Regras e regiões", "câmera", disk["evaluation"].get("roi_pattern"),
               desc="Usada apenas na avaliação offline contra o GT; não afeta o processamento ao vivo.", source="config"))

    # --- por câmera (valores efetivos e origem)
    cams = []
    drift = (cfg.get("alignment") or {}).get("drift_ppm") or {}
    for c, w in svc.workers.items():
        d = descs[c]
        info = svc.registry[c]
        cam_cfg = next((x for x in cfg["cameras"] if str(x["id"]) == c), {})
        zones = (sp.get("zones") or {}).get(c)
        cams.append({
            "id": c, "state": w.status()["state"], "error": w.error,
            "model": (d.get("detector") or {}).get("name") or (d.get("detector") or {}).get("format"),
            "model_load": (d.get("detector") or {}).get("load_state") if d.get("detector") else ("replay" if d.get("kind") == "replay" else "não carregado"),
            "tracker": (d.get("tracker") or {}).get("type"),
            "params": {
                "detector.conf": {"value": conf_applied if conf_applied is not None else dcfg.get("conf"), "origin": "global"},
                "detector.imgsz": {"value": dcfg.get("imgsz"), "origin": "global"},
                "tracker.track_buffer": {"value": ta.get("track_buffer", tcfg.get("track_buffer")), "origin": "global"},
                "tracker.match_thresh": {"value": ta.get("match_thresh", tcfg.get("match_thresh")), "origin": "global"},
                "video.fps": {"value": info.fps, "origin": "vídeo"},
                "time.offset_s": {"value": info.offset, "origin": "camera" if cam_cfg.get("offset_adjust") else "metadados (cam_timestamp)"},
                "time.drift_ppm": {"value": info.drift_ppm, "origin": "camera" if c in drift else "global (padrão 0)"},
                "spatial.zones": {"value": len(zones) if zones else 0, "origin": "camera" if zones else "não configurado"},
                "spatial.homography": {"value": c in (sp.get("homographies") or {}), "origin": "calibração" if c in (sp.get("homographies") or {}) else "não configurado"},
            },
        })

    models: List[Dict[str, Any]] = []
    if first:
        users = [c for c, d in descs.items() if d.get("detector")]
        models.append({"id": "detector", "role": "detector", "name": det.get("name") or dcfg.get("model") or "detector",
                       "artifact": dcfg.get("model") if dcfg.get("type", "ultralytics_track") in ("ultralytics", "ultralytics_track") else None,
                       "kind": kind, **{k: det.get(k) for k in ("task", "format", "backend_class", "device", "load_state", "input_shape", "io_dtype", "io",
                                                               "engine_metadata", "stride", "end2end", "params", "io_error")},
                       "classes": [{"id": i, "name": n} for i, n in sorted((det.get("names") or {}).items())],
                       "n_classes": len(det.get("names") or {}), "cameras": users})
    if reid:
        models.append({"id": "reid", "role": "re-id", "name": os.path.basename(str(rcfg.get("model"))), "artifact": rcfg.get("model"),
                       "n_classes": None, "classes": [], "cameras": list(svc.workers), **reid})
    if trk:
        models.append({"id": "tracker", "role": "tracker", "name": trk.get("type"), "artifact": trk.get("module"), "task": "tracking",
                       "format": "algoritmo (sem pesos)", "load_state": "ativo", "classes": [], "n_classes": None,
                       "cameras": [c for c, d in descs.items() if d.get("tracker")]})

    return {
        "schema_version": SCHEMA_VERSION, "read_only": True, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": svc.mode, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_at)),
        "revision": config_revision(cfg), "config": config_state(cfg, cfg.get("_config_path")),
        "versions": runtime_versions(), "history": {"supported": False, "reason": "o backend não registra alterações (somente leitura)"},
        "models": models, "parameters": params, "cameras": cams,
        "association": {"status": svc.assoc_note, **(ass.stats if ass else {})},
    }
