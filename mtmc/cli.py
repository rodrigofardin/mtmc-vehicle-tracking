"""CLI: python -m mtmc <comando> --config configs/xxx.yaml"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

from .config import load_config
from .videos import VideoRegistry

log = logging.getLogger("mtmc")


def out(cfg, *p) -> Path:
    d = Path(cfg["output"]["dir"]).joinpath(*p)
    d.parent.mkdir(parents=True, exist_ok=True)
    return d


def cmd_inspect(cfg, a):
    reg = VideoRegistry.from_config(cfg, count_frames=a.count_frames or cfg["processing"]["count_frames"])
    out(cfg, "registry.json").write_text(json.dumps(reg.to_dict(), indent=2), encoding="utf-8")
    print(f"{'cam':6} {'ok':3} {'fps':>5} {'res':>10} {'frames':>7} {'meta_fn':>8} {'ts_raw':>7} "
          f"{'offset':>7} {'g_start':>8} {'g_end':>8}")
    for c, i in reg.infos.items():
        print(f"{c:6} {str(i.accessible)[0]:3} {i.fps:5.1f} {f'{i.width}x{i.height}':>10} "
              f"{i.n_frames:7d} {str(i.meta_frames):>8} {str(i.meta_timestamp_raw):>7} "
              f"{i.offset:7.3f} {i.global_start:8.3f} {i.global_end:8.3f}")
        for w in i.warnings:
            print(f"   ! {w}")
    for n in reg.notes:
        print(f"NOTA: {n}")
    print("timeline global:", reg.common_timeline(), "| sobreposição de todas:", reg.synchronized_overlap())
    return reg


def _registry(cfg):
    return VideoRegistry.from_config(cfg, count_frames=cfg["processing"]["count_frames"])


def cmd_detect(cfg, a):
    from .detect_track import run_camera
    reg = _registry(cfg)
    cams = a.cameras or reg.cam_ids
    for c in cams:
        try:
            s = run_camera(cfg, reg[c], force=a.force)
            print(c, {k: s[k] for k in ("frames_processed", "rows", "frames_without_detections",
                                        "read_failures") if k in s}, "(reutilizado)" if s.get("reused") else "")
        except Exception as e:  # uma câmera com falha não derruba as demais
            log.error("%s: falhou: %s", c, e)


def cmd_tracklets(cfg, a):
    from .detect_track import det_paths, read_detections
    from .tracklets import build_tracklets, extract_crops, save_tracklets
    reg = _registry(cfg)
    allt = []
    for c in reg.cam_ids:
        p, _ = det_paths(cfg, c)
        if not p.is_file():
            log.warning("%s: sem detecções (%s); rode `detect`", c, p)
            continue
        info = reg[c]
        allt += build_tracklets(read_detections(p), c, (info.width, info.height), cfg, str(p))
    extract_crops(allt, reg, cfg)
    save_tracklets(allt, out(cfg, "tracklets.json"))
    print(f"{len(allt)} tracklets salvos em {out(cfg, 'tracklets.json')}")


def cmd_associate(cfg, a):
    from .association import associate, save_association
    from .reid import ReIDUnavailable, attach_embeddings, build_reid
    from .tracklets import load_tracklets, save_tracklets
    ts = load_tracklets(out(cfg, "tracklets.json"))
    try:
        ext = build_reid(cfg)
    except ReIDUnavailable as e:
        log.error("Re-ID indisponível: %s", e)
        return 2
    if ext is None:
        log.warning("Re-ID desabilitado: associação sem aparência (ver README)")
    else:
        print("embeddings:", attach_embeddings(ts, ext), "de", len(ts), "tracklets")
        save_tracklets(ts, out(cfg, "tracklets.json"))
    ids, decs = associate(ts, cfg)
    save_association(ids, decs, out(cfg))
    from collections import Counter
    print("decisões por status:", dict(Counter(d.status for d in decs)))
    print(f"{len(ids)} identidades globais; {sum(len(i.members) > 1 for i in ids)} com >1 câmera/tracklet")


def cmd_visualize(cfg, a):
    from .association import uid_to_gid
    from .tracklets import load_tracklets
    from .visualize import build_lookup, render_camera, render_mosaic
    reg = _registry(cfg)
    ts = load_tracklets(out(cfg, "tracklets.json"))
    gp = out(cfg, "global_identities.json")
    gmap, unc = None, set()
    if gp.is_file():
        from .association import GlobalIdentity
        ids = [GlobalIdentity(**d) for d in json.loads(gp.read_text(encoding="utf-8"))]
        gmap = uid_to_gid(ids)
        unc = {u for i in ids if i.uncertain for u in i.members}
    else:
        log.warning("sem global_identities.json; visualização só com IDs locais")
    v = cfg["visualization"]
    cams = [c for c in (a.cameras or v.get("cameras") or reg.usable())]
    lookup = build_lookup(cfg, cams, ts, gmap, unc)
    mode = a.mode or v["mode"]
    ms = a.max_seconds or v.get("max_seconds")
    for c in cams:
        p = out(cfg, "viz", f"{c}_{mode}.mp4")
        render_camera(cfg, reg, c, lookup, mode, p, ms)
        print("gerado", p)
    if not a.no_mosaic:
        p = out(cfg, "viz", f"mosaic_{mode}.mp4")
        n = render_mosaic(cfg, reg, cams, lookup, mode, p, v["cell_width"], v.get("fps"), ms, a.start)
        print("gerado", p, f"({n} passos)")


def _gt_setup(cfg, reg, ts):
    """Carrega GT, ROIs, detecções do tracking local e o avaliador global pré-computado."""
    from .detect_track import det_paths, read_detections
    from .evaluation import GlobalEvaluator, load_gt, load_rois, tracklet_to_gt
    ev = cfg["evaluation"]
    gt = load_gt(ev["gt_dir"], reg.cam_ids)
    rois = load_rois(ev["roi_pattern"], reg.cam_ids) if ev.get("roi_pattern") else None
    rows = []
    for c in reg.cam_ids:
        p, _ = det_paths(cfg, c)
        if p.is_file():
            rows += read_detections(p)
    keep = {t.uid for t in ts}                      # só detecções de tracklets mantidos
    rows = [r for r in rows if f"{r['cam']}:{r['track_id']}" in keep]
    return gt, rois, rows, GlobalEvaluator(rows, gt, ev["iou_threshold"], rois), tracklet_to_gt(ts, gt, ev["iou_threshold"])


def cmd_evaluate(cfg, a):
    from .association import GlobalIdentity, uid_to_gid
    from .evaluation import association_metrics, export_mot, idf1_and_switches, to_frame_dict
    from .tracklets import load_tracklets
    reg = _registry(cfg)
    ts = load_tracklets(out(cfg, "tracklets.json"))
    ids = [GlobalIdentity(**d) for d in json.loads(out(cfg, "global_identities.json").read_text(encoding="utf-8"))]
    gmap = uid_to_gid(ids)
    ev = cfg["evaluation"]
    if not ev.get("gt_dir"):
        from .detect_track import det_paths, read_detections
        rows = []
        for c in reg.cam_ids:
            p, _ = det_paths(cfg, c)
            if p.is_file():
                rows += [r for r in read_detections(p) if f"{c}:{r['track_id']}" in gmap]
        gnum = lambda r: int(gmap[f"{r['cam']}:{r['track_id']}"][1:])
        export_mot(rows, gnum, out(cfg, "mot_global"))
        export_mot(rows, lambda r: r["track_id"], out(cfg, "mot_local"))
        print("SEM ground truth (evaluation.gt_dir ausente): métricas NÃO calculadas.\n"
              f"Resultados exportados para revisão manual/TrackEval em {out(cfg, 'mot_global')}")
        return
    gt, rois, rows, ge, u2g = _gt_setup(cfg, reg, ts)
    if not gt:
        print("gt_dir sem GT para as câmeras configuradas; nada avaliado")
        return
    gnum = lambda r: int(gmap[f"{r['cam']}:{r['track_id']}"][1:])
    export_mot(rows, gnum, out(cfg, "mot_global"))
    export_mot(rows, lambda r: r["track_id"], out(cfg, "mot_local"))
    res = {"global": ge.idf1(gmap, ev.get("drop_single_camera", True)),
           "association": association_metrics(ts, gmap, u2g),
           "local_tracking": idf1_and_switches(gt, to_frame_dict(rows, lambda r: r["track_id"]), ev["iou_threshold"]),
           "notes": ["IDF1 global: implementação própria (não é o eval.py oficial); protocolo AIC22: ROI + só IDs em >=2 câmeras.",
                     "local_tracking: o GT só anota veículos vistos em >=2 câmeras; veículos sem anotação contam como FP (IDF1 local subestimado)."]}
    out(cfg, "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


def cmd_tune(cfg, a):
    """Varre limiares de associação contra o GT (usa tracklets.json com embeddings já calculados)."""
    import copy
    import itertools
    from .association import associate, uid_to_gid
    from .evaluation import association_metrics
    from .tracklets import load_tracklets
    reg = _registry(cfg)
    ts = load_tracklets(out(cfg, "tracklets.json"))
    if not cfg["evaluation"].get("gt_dir"):
        print("tune exige evaluation.gt_dir")
        return 2
    _, _, _, ge, u2g = _gt_setup(cfg, reg, ts)
    grid = cfg.get("tuning", {}).get("grid") or {}
    keys = list(grid)
    results = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        c = copy.deepcopy(cfg)
        for k, v in zip(keys, combo):
            sec, name = k.split(".", 1)
            c[sec][name] = v
        ids, _ = associate(ts, c)
        gmap = uid_to_gid(ids)
        r = ge.idf1(gmap, cfg["evaluation"].get("drop_single_camera", True))
        m = association_metrics(ts, gmap, u2g)
        results.append({"params": dict(zip(keys, combo)), "IDF1": r["IDF1"], "IDP": r["IDP"], "IDR": r["IDR"],
                        "pair_precision": m["pair_precision"], "pair_recall": m["pair_recall"],
                        "linked": sum(len(i.members) > 1 for i in ids)})
        print(json.dumps(results[-1]), flush=True)
    results.sort(key=lambda r: -r["IDF1"])
    out(cfg, "tuning.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nTOP 5 por IDF1:")
    for r in results[:5]:
        print(json.dumps(r))


def cmd_snapshot(cfg, a):
    reg = _registry(cfg)
    for c in reg.usable():
        info = reg[c]
        cap = cv2.VideoCapture(info.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
        ok, img = cap.read()
        cap.release()
        if ok:
            p = out(cfg, "snapshots", f"{c}_f{a.frame}.jpg")
            cv2.imwrite(str(p), img)
            print("gerado", p, f"{info.width}x{info.height}")


def cmd_reid_check(cfg, a):
    from .reid_check import run_check
    reg = _registry(cfg)
    res = run_check(cfg, reg, per_track=a.per_track)
    out(cfg, "reid_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"{'variante':14} {'AUC':>6} {'rank1':>6} {'mAP':>6} {'pos':>6} {'neg':>6}")
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["auc"]):
        print(f"{k:14} {v['auc']:6.3f} {v['rank1']:6.3f} {v['mAP']:6.3f} {v['pos_mean']:6.3f} {v['neg_mean']:6.3f}")


def cmd_serve(cfg, a):
    import uvicorn
    from .server import create_app
    s = cfg["server"]
    uvicorn.run(create_app(cfg, a.serve_mode), host=a.host or s["host"], port=a.port or s["port"],
                log_level="info")


def cmd_run(cfg, a):
    cmd_inspect(cfg, a)
    cmd_detect(cfg, a)
    cmd_tracklets(cfg, a)
    cmd_associate(cfg, a)
    cmd_evaluate(cfg, a)
    if not a.no_viz:
        cmd_visualize(cfg, a)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mtmc")
    ap.add_argument("command", choices=["inspect", "snapshot", "detect", "tracklets", "associate",
                                        "visualize", "evaluate", "run", "serve", "reid-check", "tune"])
    ap.add_argument("--per-track", type=int, default=3, help="reid-check: recortes por (câmera, veículo)")
    ap.add_argument("--serve-mode", choices=["live", "replay"], help="serve: live|replay")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--cameras", nargs="*")
    ap.add_argument("--force", action="store_true", help="ignora detecções salvas")
    ap.add_argument("--count-frames", action="store_true", help="decodifica para contar frames reais")
    ap.add_argument("--mode", choices=["local", "global", "compare"])
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--start", type=float, help="início do mosaico (tempo global, s)")
    ap.add_argument("--no-mosaic", action="store_true")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    fn = {"inspect": cmd_inspect, "snapshot": cmd_snapshot, "detect": cmd_detect,
          "tracklets": cmd_tracklets, "associate": cmd_associate, "visualize": cmd_visualize,
          "evaluate": cmd_evaluate, "run": cmd_run, "serve": cmd_serve,
          "reid-check": cmd_reid_check, "tune": cmd_tune}[a.command]
    r = fn(cfg, a)
    return r if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())
