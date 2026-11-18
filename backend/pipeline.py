"""
Model hook for the PathSnap API.  Save as backend/pipeline.py

Works on ANY folder of raw trips (student-test or evaluator): main.py reads the raw .track/.nodes/.arcs,
cleans the GPS rows and projects them to metres; this file turns that into the inputs that
preProcessing/features.py expects, builds the 19 features, scores candidates with the trained model and
returns one arc_id per GPS point. It never reads .route or any label.

Start the API with:
    PREDICTOR=model  MODEL_ENTRY=pipeline:reconstruct

Env vars (all optional):
    MODEL_PATH          default <repo>/models/random_forest.joblib   (.pkl also works)
    META_PATH           default <repo>/models/meta.json              (feature order, used if the file has no list)
    SCALER_PATH         set ONLY if the model was trained on scaled features (the npz X); e.g. models/scaler.pkl
    PATHSNAP_GAP_S      time gap (s) that starts a new track segment, default 60
"""
import os, json, pickle, logging
import numpy as np
import pandas as pd
import joblib

from preProcessing.trainingCode.features import build_candidates, FEATURES   # repo root is on sys.path (main.py adds it)

log = logging.getLogger("pathsnap")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT, "models", "random_forest.joblib"))
META_PATH = os.environ.get("META_PATH", os.path.join(ROOT, "models", "meta.json"))
SCALER_PATH = os.environ.get("SCALER_PATH")
GAP_S = float(os.environ.get("PATHSNAP_GAP_S", "60"))


def _load(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


_obj = _load(MODEL_PATH)
MODEL = _obj["model"] if isinstance(_obj, dict) else _obj
if isinstance(_obj, dict) and "features" in _obj:
    FEATS = list(_obj["features"])
elif os.path.exists(META_PATH):
    FEATS = json.load(open(META_PATH))["features"]
else:
    FEATS = list(FEATURES)
assert set(FEATS) <= set(FEATURES), f"model needs features that features.py does not build: {set(FEATS) - set(FEATURES)}"
SCALER = _load(SCALER_PATH) if SCALER_PATH else None
if hasattr(MODEL, "n_jobs"):
    MODEL.n_jobs = -1
log.info("pipeline loaded: model=%s | features=%d | scaler=%s | gap=%.0fs",
         type(MODEL).__name__, len(FEATS), "yes" if SCALER is not None else "no", GAP_S)


def _feature_net(net):
    """Network in the structure features.py expects. Built once and cached on the network dict."""
    fn = net.get("_feat")
    if fn is None:
        nx, ny = net["tr"].transform(net["lon"], net["lat"])
        arc_len = np.asarray(net["length"], dtype=float)
        fn = dict(
            arcs=np.c_[net["u"], net["v"]].astype(np.int64),
            node_xy=np.c_[nx, ny],
            arc_len=arc_len,
            valid=arc_len > 0,                      # zero-length arcs are not routable
        )
        net["_feat"] = fn
    return fn


def _points(x, y, t):
    """Points table for one trip: point_idx = position in the x/y/t arrays passed in."""
    n = len(x)
    t = np.asarray(t, dtype=float)
    dt = np.r_[0.0, np.diff(t)]
    brk = np.r_[False, (dt[1:] > GAP_S) | (dt[1:] < 0)]          # new segment after a gap or backwards time
    seg_id = np.cumsum(brk)
    dt = np.where(np.r_[True, brk[1:]], 0.0, dt)                 # first point of a segment has dt = 0
    return pd.DataFrame(dict(point_idx=np.arange(n), x=np.asarray(x, float), y=np.asarray(y, float),
                             t=t, seg_id=seg_id, dt=dt, subject_id="trip"))


def reconstruct(net, x, y, t):
    """Return one predicted arc_id per input GPS point (-1 where no candidate exists)."""
    n = len(x)
    out = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return out
    cand = build_candidates(_points(x, y, t), _feature_net(net))
    if cand is None or len(cand) == 0:
        return out

    X = cand[FEATS].to_numpy()
    if SCALER is not None:
        X = SCALER.transform(X)
    pos = list(MODEL.classes_).index(1)                           # never assume column 1 is the positive class
    cand = cand.assign(p=MODEL.predict_proba(X)[:, pos])

    best = cand.loc[cand.groupby("point_idx")["p"].idxmax()]      # classifier-only: best candidate per point
    out[best["point_idx"].to_numpy().astype(int)] = best["arc_id"].to_numpy().astype(np.int64)
    if (out < 0).any():
        log.warning("%d of %d GPS points had no candidate arc", int((out < 0).sum()), n)
    return out