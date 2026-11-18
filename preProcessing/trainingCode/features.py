"""
Serving-safe feature code.  Save as preProcessing/features.py

The functions below are copied from featureEngineering.ipynb (sections 2.4 to 2.7) so that the API builds
exactly the same 19 features that the models were trained on. The ONLY difference: the KD-tree index can be
cached on the network, because rebuilding it on every request is slow. Nothing here reads .route or labels.

Recommendation: the notebook should import these functions from this file instead of keeping its own copy,
so training and serving can never drift apart.

Input contract (both come from dataProcessing):
  pts : DataFrame with columns point_idx, x, y, t, seg_id, dt, subject_id   (cleaned, projected, one trip)
  net : dict with arcs (A,2 int), node_xy (N,2), valid (A, bool), arc_len (A,)   (same projected frame as pts)
Output of build_candidates: DataFrame with point_idx, seg_id, subject_id, arc_id, the 19 FEATURES and head_diff_deg.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

K_CAND = 10
SEARCH_R = 120.0
DENSIFY_M = 15.0
STATIONARY_M = 3.0
MAX_SPEED = 60.0

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "n_within_50m", "frac_along", "arc_len",
    "head_diff_cos", "head_valid", "arc_sin", "arc_cos", "end_outdeg", "start_indeg",
    "dist_to_start", "dist_to_end", "speed", "dt", "accel", "turn_rate", "seg_pos",
]


def add_kinematics(df):
    df = df.copy()
    n = len(df)
    xy = df[["x", "y"]].to_numpy()
    seg = df["seg_id"].to_numpy()
    dt = df["dt"].to_numpy()
    new_seg = np.r_[True, seg[1:] != seg[:-1]]

    step = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    step[new_seg] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, 0.0)
    speed = np.clip(speed, 0, MAX_SPEED)
    with np.errstate(divide="ignore", invalid="ignore"):
        accel = np.r_[0.0, np.diff(speed)] / np.where(dt > 0, dt, np.nan)
    accel[new_seg | np.r_[True, new_seg[:-1]]] = 0.0
    accel = np.clip(np.nan_to_num(accel), -10, 10)

    idx = np.arange(n)
    prev_i = np.where(new_seg, idx, idx - 1)
    next_i = np.minimum(idx + 1, n - 1)
    next_i = np.where(seg[next_i] != seg, idx, next_i)
    d = xy[next_i] - xy[prev_i]
    heading = np.arctan2(d[:, 1], d[:, 0])
    head_valid = (np.linalg.norm(d, axis=1) > STATIONARY_M).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        turn = np.r_[0.0, np.abs(np.angle(np.exp(1j * np.diff(heading))))] / np.where(dt > 0, dt, np.nan)
    turn_rate = np.clip(np.nan_to_num(turn), 0, np.pi)
    turn_rate[new_seg] = 0.0

    g = df.groupby("seg_id")["t"]
    span = (g.transform("max") - g.transform("min")).replace(0, 1)

    df["dist_prev"], df["speed"], df["accel"] = step, speed, accel
    df["heading"], df["head_valid"], df["turn_rate"] = heading, head_valid, turn_rate
    df["seg_pos"] = (df["t"] - g.transform("min")) / span
    return df


def add_network_features(net):
    u, v = net["arcs"][:, 0], net["arcs"][:, 1]
    A, B = net["node_xy"][u], net["node_xy"][v]
    n_nodes = len(net["node_xy"])
    net["ang"] = np.arctan2(B[:, 1] - A[:, 1], B[:, 0] - A[:, 0])
    net["outdeg"] = np.bincount(u[net["valid"]], minlength=n_nodes)
    net["indeg"] = np.bincount(v[net["valid"]], minlength=n_nodes)
    net["A"], net["B"] = A, B
    return net


def build_index(net):
    ids = np.flatnonzero(net["valid"])
    n = np.maximum(np.ceil(net["arc_len"][ids] / DENSIFY_M).astype(int), 1) + 1
    arc_of = np.repeat(ids, n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n - 1, n)
    verts = net["A"][arc_of] + (net["B"][arc_of] - net["A"][arc_of]) * t[:, None]
    return cKDTree(verts), arc_of


def point_to_arcs(p, arc_ids, net):
    A, B = net["A"][arc_ids], net["B"][arc_ids]
    AB = B - A
    t = np.clip(((p - A) * AB).sum(1) / (AB ** 2).sum(1), 0, 1)
    return np.linalg.norm(p - (A + AB * t[:, None]), axis=1), t


def search_candidates(pts, net):
    if "_index" not in net:                      # cached on the network: built once per trip, not per request
        net["_index"] = build_index(net)
    tree, arc_of = net["_index"]
    P = pts[["x", "y"]].to_numpy()
    balls = tree.query_ball_point(P, r=SEARCH_R + DENSIFY_M / 2)
    out_pt, out_arc, out_d, out_t, out_rank = [], [], [], [], []
    for i, (p, hits) in enumerate(zip(P, balls)):
        cand = np.unique(arc_of[hits]) if len(hits) else np.array([], int)
        if len(cand) < K_CAND:
            _, nn = tree.query(p, k=min(64, tree.n))
            cand = np.unique(np.r_[cand, arc_of[np.atleast_1d(nn)]]).astype(int)
        d, t = point_to_arcs(p, cand, net)
        o = np.argsort(d)[:K_CAND]
        out_pt.append(np.full(len(o), i)); out_arc.append(cand[o]); out_d.append(d[o]); out_t.append(t[o])
        out_rank.append(np.arange(len(o)))
    return pd.DataFrame({"pt": np.concatenate(out_pt), "arc_id": np.concatenate(out_arc),
                         "dist": np.concatenate(out_d), "frac_along": np.concatenate(out_t),
                         "dist_rank": np.concatenate(out_rank)})


def add_candidate_features(c, pts, net):
    arc = c["arc_id"].to_numpy()
    g = c.groupby("pt")["dist"]
    c["dist_minus_min"] = c["dist"] - g.transform("min")
    c["n_within_50m"] = g.transform(lambda s: (s <= 50).sum())
    c["arc_len"] = net["arc_len"][arc]
    c["dist_to_start"] = c["frac_along"] * c["arc_len"]
    c["dist_to_end"] = (1 - c["frac_along"]) * c["arc_len"]
    ang = net["ang"][arc]
    c["arc_sin"], c["arc_cos"] = np.sin(ang), np.cos(ang)
    c["end_outdeg"] = net["outdeg"][net["arcs"][arc, 1]]
    c["start_indeg"] = net["indeg"][net["arcs"][arc, 0]]

    pt = c["pt"].to_numpy()
    hv = pts["head_valid"].to_numpy()[pt]
    diff = np.angle(np.exp(1j * (pts["heading"].to_numpy()[pt] - ang)))
    c["head_valid"] = hv
    c["head_diff_cos"] = np.cos(diff) * hv
    c["head_diff_deg"] = np.degrees(np.abs(diff)) * hv
    for col in ["speed", "dt", "accel", "turn_rate", "seg_pos"]:
        c[col] = pts[col].to_numpy()[pt]
    return c


def build_candidates(pts, net):
    """Label-free candidate table for one trip. Safe to call at inference (never touches .route)."""
    net = add_network_features(net)
    pts = add_kinematics(pts.reset_index(drop=True))
    c = search_candidates(pts, net)
    c = add_candidate_features(c, pts, net)
    pt = c["pt"].to_numpy()
    c["point_idx"] = pts["point_idx"].to_numpy()[pt]
    c["seg_id"] = pts["seg_id"].to_numpy()[pt]
    c["subject_id"] = pts["subject_id"].iloc[0]
    return c.drop(columns="pt")