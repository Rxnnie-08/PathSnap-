"""
PathSnap API (Laptop 1).  Save as backend/main.py

Install:   pip install fastapi uvicorn pandas numpy "shapely>=2" pyproj
Run (from inside the backend folder):
    PowerShell:  $env:DATA_DIR="C:\\path\\to\\trips"; uvicorn main:app --host 0.0.0.0 --port 8000
    cmd:         set DATA_DIR=C:\\path\\to\\trips && uvicorn main:app --host 0.0.0.0 --port 8000

Environment variables (no paths or secrets are stored in the code):
    DATA_DIR      folder containing the trip folders (<id>.track/.nodes/.arcs[/.route]).
                  Use the student-test trips while developing, the evaluator trips only at final evaluation.
    PREDICTOR     "baseline" (default) -> nearest-arc baseline
                  "model"              -> the team's classifier + HMM pipeline (see MODEL_ENTRY)
    MODEL_ENTRY   "module.path:function" of the model pipeline, default "hmm.pipeline:reconstruct".
                  The function must have the signature  reconstruct(net, x, y, t) -> int array
                  (one predicted arc_id per input point, -1 where none).

If PREDICTOR=model and the import fails, the API refuses to start. There is deliberately no silent
fallback to the baseline, so the predictor name in the logs and UI always tells the truth.

Reference routes (.route) are read server-side only to compute metrics and are never sent to the UI.
"""
import os, sys, glob, time, logging, importlib
from functools import lru_cache

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from pyproj import Transformer
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# repo root on the import path, so the model code in sibling folders (hmm/, models/) can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_DIR = os.environ.get("DATA_DIR", "data")
PREDICTOR = os.environ.get("PREDICTOR", "baseline").lower()
MODEL_ENTRY = os.environ.get("MODEL_ENTRY", "hmm.pipeline:reconstruct")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("pathsnap")


# --------------------------------------------------------------------------- data access
def index_trips():
    idx = {}
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.track"), recursive=True)):
        base = p[: -len(".track")]
        tid = os.path.basename(base)
        if tid in idx:  # same file stem in different folders -> make the id unique
            tid = os.path.basename(os.path.dirname(base)) + "_" + tid
        idx[tid] = base
    return idx


TRIPS = index_trips()


@lru_cache(maxsize=4)
def load_network(nodes_p, arcs_p):
    nod = pd.read_csv(nodes_p, sep=r"\s+", header=None, names=["lon", "lat"])
    arc = pd.read_csv(arcs_p, sep=r"\s+", header=None, names=["u", "v"])
    lon0, lat0 = nod.lon.mean(), nod.lat.mean()
    epsg = (32600 if lat0 >= 0 else 32700) + int((lon0 + 180) // 6) + 1
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    nx, ny = tr.transform(nod.lon.values, nod.lat.values)
    u, v = arc.u.values, arc.v.values
    ax, ay, bx, by = nx[u], ny[u], nx[v], ny[v]
    segs = shapely.linestrings(np.stack([np.stack([ax, ay], 1), np.stack([bx, by], 1)], 1))
    return dict(
        tr=tr, epsg=epsg, u=u, v=v, lon=nod.lon.values, lat=nod.lat.values,
        segs=segs, tree=STRtree(segs),
        length=np.hypot(bx - ax, by - ay), bearing=np.arctan2(by - ay, bx - ax),
    )


# --------------------------------------------------------------------------- predictor
def _baseline(net, x, y, t):
    """Nearest-arc baseline: each point independently takes the closest arc. No sequence model."""
    pts = shapely.points(x, y)
    pair = net["tree"].query_nearest(pts, all_matches=False)  # shape (2, n)
    out = np.full(len(x), -1, dtype=np.int64)
    out[pair[0]] = pair[1]
    return out


if PREDICTOR == "model":
    _mod, _fn = MODEL_ENTRY.split(":")
    _reconstruct = getattr(importlib.import_module(_mod), _fn)  # raises -> API will not start
    PREDICTOR_NAME = "classifier_hmm"
elif PREDICTOR == "baseline":
    _reconstruct = _baseline
    PREDICTOR_NAME = "nearest_arc_baseline"
else:
    raise SystemExit(f"PREDICTOR must be 'baseline' or 'model', got '{PREDICTOR}'")


def predict_arcs(net, x, y, t):
    """One predicted arc_id per input point (-1 if none). x, y in metres; t in seconds."""
    out = np.asarray(_reconstruct(net, x, y, t), dtype=np.int64)
    if out.shape != (len(x),):
        raise ValueError(f"predictor returned shape {out.shape}, expected ({len(x)},)")
    return out


# --------------------------------------------------------------------------- pipeline for one trip
def _clean(v):
    if v is None:
        return None
    v = float(v)
    return None if np.isnan(v) else round(v, 4)


def run_trip(trip_id):
    base = TRIPS[trip_id]
    net = load_network(f"{base}.nodes", f"{base}.arcs")
    trk = pd.read_csv(f"{base}.track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    valid = ~(trk[["lon", "lat", "t"]].isna().any(axis=1)
              | ~trk.lon.between(-180, 180) | ~trk.lat.between(-90, 90))
    trk = trk[valid].reset_index(drop=True)
    x, y = net["tr"].transform(trk.lon.values, trk.lat.values)

    pred = predict_arcs(net, np.asarray(x), np.asarray(y), trk.t.values)
    seq = pred[pred >= 0]
    route = seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq  # collapse repeated arcs

    segments = [[[float(net["lat"][net["u"][a]]), float(net["lon"][net["u"][a]])],
                 [float(net["lat"][net["v"][a]]), float(net["lon"][net["v"][a]])]] for a in route]

    metrics = {"route_continuity": None, "arc_precision": None, "arc_recall": None, "arc_f1": None}
    if len(route) > 1:
        metrics["route_continuity"] = _clean(np.mean(net["v"][route[:-1]] == net["u"][route[1:]]))
    elif len(route) == 1:
        metrics["route_continuity"] = 1.0

    ref_p = f"{base}.route"  # ground truth: server-side only, never returned
    if os.path.exists(ref_p) and len(route):
        ref = set(pd.read_csv(ref_p, header=None)[0].astype(int).tolist())
        ps = set(route.tolist())
        inter = len(ps & ref)
        prec = inter / len(ps)
        rec = inter / len(ref) if ref else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics.update(arc_precision=_clean(prec), arc_recall=_clean(rec), arc_f1=_clean(f1))

    return {
        "trip_id": trip_id,
        "predictor": PREDICTOR_NAME,
        "n_points": int(len(trk)),
        "gps": trk[["lat", "lon"]].values.tolist(),
        "pred_arc_per_point": [int(a) for a in pred],
        "route_arcs": [int(a) for a in route],
        "segments": segments,
        "metrics": metrics,
    }


# --------------------------------------------------------------------------- API
app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
log.info("DATA_DIR=%s | %d trips indexed | predictor=%s", os.path.abspath(DATA_DIR), len(TRIPS), PREDICTOR_NAME)


class PredictReq(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "predictor": PREDICTOR_NAME, "n_trips": len(TRIPS)}


@app.get("/trips")
def trips():
    return {"trips": list(TRIPS.keys())}


@app.post("/predict")
def predict(req: PredictReq, request: Request):
    if req.trip_id not in TRIPS:
        log.warning("UNKNOWN trip_id=%s from %s", req.trip_id, request.client.host)
        raise HTTPException(404, f"unknown trip_id {req.trip_id}")
    t0 = time.perf_counter()
    try:
        out = run_trip(req.trip_id)
    except Exception as e:
        log.exception("PREDICT FAILED trip=%s", req.trip_id)
        raise HTTPException(500, f"pipeline error: {e}")
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    m = out["metrics"]
    log.info("PREDICT trip=%s client=%s predictor=%s points=%d arc_f1=%s continuity=%s latency=%dms",
             req.trip_id, request.client.host, PREDICTOR_NAME, out["n_points"],
             m["arc_f1"], m["route_continuity"], out["latency_ms"])
    return out