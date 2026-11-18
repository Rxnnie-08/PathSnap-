"""
Task 2: HMM + Viterbi Route Reconstruction — Evaluation.

Compares classifier-only vs classifier+HMM on the 10 student-test trips.

Classifier-only:  argmax of Random Forest P(on_route=1) per GPS point independently.
Classifier+HMM:   same RF probabilities as emissions, Viterbi decoding with
                  road-network transition scores.

Run from the repo root:
    python models/05_HMM_Viterbi_Evaluation.py

Outputs (models/results/):
    hmm_comparison.csv          per-trip and overall metrics
    hmm_comparison.png           comparison plots
    hmm_config.json              HMM hyperparameters
"""

import os, sys, json, pickle, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, log_loss

sys.path.insert(0, os.path.dirname(__file__))
from hmm.viterbi import HMMRouteReconstructor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROCESSED = "preProcessing"
RAW_TEST = "data/raw/student_test"
OUT = "models/results"
os.makedirs(OUT, exist_ok=True)

EARTH_R = 6371008.8
EPS = 1e-12


# --------------------------------------------------------------------------- data
def load_split(split):
    z = np.load(os.path.join(PROCESSED, split, "model_data.npz"), allow_pickle=True)
    meta = pd.DataFrame({
        "subject_id": z["groups"].astype(str),
        "point_idx": z["point_idx"],
        "seg_id": z["seg_id"],
        "arc_id": z["arc_id"],
        "y": z["y"].astype(int),
        "y_true_arc": z["y_true_arc"].astype(int),
    })
    return z["X"], meta, [str(f) for f in z["feature_names"]]


def load_road_network(sid):
    """Load .nodes and .arcs for one subject from the raw data folder."""
    p = os.path.join(RAW_TEST, sid, sid)
    nodes = np.loadtxt(p + ".nodes", ndmin=2)
    arcs = np.loadtxt(p + ".arcs", dtype=int, ndmin=2)
    return nodes, arcs


def load_track(sid):
    """Load the cleaned GPS track for one subject."""
    p = os.path.join(RAW_TEST, sid, sid)
    trk = np.loadtxt(p + ".track", ndmin=2)  # (T, 3): lon, lat, t
    return trk


def project(lon, lat, lon0, lat0):
    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return np.c_[x, y]


# --------------------------------------------------------------------------- load RF model
print("Loading Random Forest model...")
with open(os.path.join(OUT, "RandomForest.pkl"), "rb") as f:
    rf_model = pickle.load(f)

# Verify classes_ mapping
print(f"  classes_: {rf_model.classes_}")
pos_label_idx = int(np.where(rf_model.classes_ == 1)[0][0])
print(f"  P(on_route=1) is column {pos_label_idx} of predict_proba")

# --------------------------------------------------------------------------- load test data
Xte, mte, FEATURES = load_split("test")
print(f"Test: {Xte.shape}, {mte['subject_id'].nunique()} subjects")

# Get RF probabilities for ALL test candidates
print("Computing RF probabilities for test candidates...")
p_all = rf_model.predict_proba(Xte)[:, pos_label_idx]
mte = mte.assign(score=p_all).reset_index(drop=True)


# --------------------------------------------------------------------------- route continuity
def route_continuity(selected_arcs, seg_ids, arcs_arr, canon_id):
    """
    Fraction of consecutive (same-segment) predicted arc pairs that are
    the same arc or end-to-start connected in the road network.

    Returns float or np.nan if there are no valid consecutive pairs.
    """
    if len(selected_arcs) < 2:
        return np.nan
    same = seg_ids[1:] == seg_ids[:-1]
    prev, cur = selected_arcs[:-1][same], selected_arcs[1:][same]
    if len(prev) == 0:
        return np.nan
    # same arc or connected
    same_arc = (prev == cur) | (canon_id[prev] == canon_id[cur])
    connected = arcs_arr[prev, 1] == arcs_arr[cur, 0]
    ok = int((same_arc | connected).sum())
    tot = int(len(prev))
    return ok / tot if tot > 0 else np.nan


# --------------------------------------------------------------------------- per-subject evaluation
def evaluate_subject(sid, subj_meta, subj_X):
    """Evaluate classifier-only and HMM for one subject."""
    nodes, arcs = load_road_network(sid)
    recon = HMMRouteReconstructor(nodes, arcs)

    # Load the raw track for GPS coordinates and timestamps
    trk = load_track(sid)
    lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
    gps_xy = project(trk[:, 0], trk[:, 1], lon0, lat0)

    # Group candidates by point_idx
    subj_meta = subj_meta.sort_values("point_idx").reset_index(drop=True)
    point_indices = subj_meta["point_idx"].unique()
    n_points = len(point_indices)

    # Build candidate lists and emission probabilities per GPS point
    # point_idx maps to the row in the track file
    candidate_lists = []
    emission_probs = []
    arc_id_per_candidate = []
    y_per_candidate = []
    y_true_arc_per_candidate = []
    dt_per_point = []

    for pi in point_indices:
        mask = subj_meta["point_idx"].values == pi
        arcs_cand = subj_meta.loc[mask, "arc_id"].values
        scores = subj_meta.loc[mask, "score"].values
        ys = subj_meta.loc[mask, "y"].values
        yta = subj_meta.loc[mask, "y_true_arc"].values

        candidate_lists.append(arcs_cand.tolist())
        emission_probs.append(scores)
        arc_id_per_candidate.append(arcs_cand)
        y_per_candidate.append(ys)
        y_true_arc_per_candidate.append(yta)

    # Compute dt per point (from track timestamps)
    t_arr = trk[:, 2]
    dt_arr = np.r_[0.0, np.diff(t_arr)]
    # Also mark segment starts (large gaps → dt=0)
    GAP_S = 60.0
    new_seg = np.r_[True, dt_arr[1:] > GAP_S]
    dt_arr[new_seg] = 0.0

    # ---- Classifier-only: argmax of RF probability per point ----
    clf_selected = []
    clf_scores = []
    for t in range(n_points):
        if len(emission_probs[t]) == 0:
            clf_selected.append(-1)
            clf_scores.append(0.0)
        else:
            best = int(np.argmax(emission_probs[t]))
            clf_selected.append(candidate_lists[t][best])
            clf_scores.append(emission_probs[t][best])

    # ---- HMM + Viterbi ----
    hmm_start = time.perf_counter()
    hmm_selected = recon.reconstruct(
        candidate_lists, emission_probs, gps_xy[:n_points], dt_arr[:n_points]
    )
    hmm_time = time.perf_counter() - hmm_start

    # ---- Build per-candidate predictions for metrics ----
    def make_pred_list(selected_arcs, candidate_lists, emission_probs):
        """Expand per-point selections back to per-candidate binary predictions."""
        preds = []
        probs = []
        for t, sel in enumerate(selected_arcs):
            if sel == -1 or len(candidate_lists[t]) == 0:
                preds.extend([0] * len(candidate_lists[t]))
                probs.extend([0.0] * len(candidate_lists[t]))
            else:
                for j, aid in enumerate(candidate_lists[t]):
                    preds.append(1 if aid == sel else 0)
                    probs.append(emission_probs[t][j])
        return np.array(preds), np.array(probs)

    # Flatten ground truth
    y_flat = np.concatenate([np.array(y_per_candidate[t]) for t in range(n_points)])
    y_true_arc_flat = np.concatenate([np.array(y_true_arc_per_candidate[t]) for t in range(n_points)])

    clf_pred, clf_prob = make_pred_list(clf_selected, candidate_lists, emission_probs)
    hmm_pred, hmm_prob = make_pred_list(hmm_selected, candidate_lists, emission_probs)

    # ---- Candidate-level metrics ----
    def cand_metrics(y, pred, prob):
        return dict(
            accuracy=accuracy_score(y, pred),
            macro_f1=f1_score(y, pred, average="macro", zero_division=0),
            log_loss=log_loss(y, np.clip(prob, EPS, 1 - EPS), labels=[0, 1]) if len(set(y)) > 1 else np.nan,
        )

    clf_cm = cand_metrics(y_flat, clf_pred, clf_prob)
    hmm_cm = cand_metrics(y_flat, hmm_pred, hmm_prob)

    # ---- Point-level top-1 ----
    # Top-1 on route: is the selected arc on the true route?
    def top1_metrics(selected_arcs, candidate_lists, y_per_candidate):
        hits = 0
        true_arc_hits = 0
        total = 0
        miss = 0
        for t in range(len(selected_arcs)):
            if len(candidate_lists[t]) == 0:
                miss += 1
                continue
            total += 1
            sel = selected_arcs[t]
            ys = y_per_candidate[t]
            arcs = candidate_lists[t]
            # on_route: is the selected arc labeled as on_route?
            on_route = False
            true_arc = False
            for j, aid in enumerate(arcs):
                if aid == sel:
                    on_route = ys[j] == 1
                    true_arc = y_per_candidate[t][j] == 1 and y_true_arc_per_candidate[t][j] == 1
                    break
            if on_route:
                hits += 1
            if true_arc:
                true_arc_hits += 1
        return dict(
            top1_on_route=hits / total if total > 0 else np.nan,
            top1_true_arc=true_arc_hits / total if total > 0 else np.nan,
            candidate_miss_rate=miss / (total + miss) if (total + miss) > 0 else np.nan,
        )

    clf_pm = top1_metrics(clf_selected, candidate_lists, y_per_candidate)
    hmm_pm = top1_metrics(hmm_selected, candidate_lists, y_per_candidate)

    # ---- Route continuity ----
    seg_ids = np.array([subj_meta.loc[subj_meta["point_idx"] == pi, "seg_id"].values[0]
                        for pi in point_indices])
    clf_cont = route_continuity(np.array(clf_selected), seg_ids, arcs, recon.canon_id)
    hmm_cont = route_continuity(np.array(hmm_selected), seg_ids, arcs, recon.canon_id)

    return dict(
        subject_id=sid,
        n_points=n_points,
        # classifier-only
        clf_accuracy=clf_cm["accuracy"],
        clf_macro_f1=clf_cm["macro_f1"],
        clf_log_loss=clf_cm["log_loss"],
        clf_top1_on_route=clf_pm["top1_on_route"],
        clf_top1_true_arc=clf_pm["top1_true_arc"],
        clf_miss_rate=clf_pm["candidate_miss_rate"],
        clf_continuity=clf_cont,
        # HMM
        hmm_accuracy=hmm_cm["accuracy"],
        hmm_macro_f1=hmm_cm["macro_f1"],
        hmm_log_loss=hmm_cm["log_loss"],
        hmm_top1_on_route=hmm_pm["top1_on_route"],
        hmm_top1_true_arc=hmm_pm["top1_true_arc"],
        hmm_miss_rate=hmm_pm["candidate_miss_rate"],
        hmm_continuity=hmm_cont,
        # timing
        hmm_inference_time_s=hmm_time,
    )


# --------------------------------------------------------------------------- run for all subjects
print("\nEvaluating classifier-only vs HMM on 10 student-test trips...")
results = []
for i, sid in enumerate(sorted(mte["subject_id"].unique())):
    subj = mte[mte["subject_id"] == sid]
    subj_X = Xte[mte["subject_id"].values == sid]
    print(f"  [{i+1}/10] {sid}  ({subj['point_idx'].nunique()} points, {len(subj)} candidates)...")
    r = evaluate_subject(sid, subj, subj_X)
    results.append(r)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT, "hmm_comparison.csv"), index=False)


# --------------------------------------------------------------------------- overall summary
print("\n=== Overall (student-test, 10 trips) ===")
overall = dict(
    subject_id="OVERALL",
    n_points=results_df["n_points"].sum(),
    clf_accuracy=results_df["clf_accuracy"].mean(),
    clf_macro_f1=results_df["clf_macro_f1"].mean(),
    clf_log_loss=results_df["clf_log_loss"].mean(),
    clf_top1_on_route=results_df["clf_top1_on_route"].mean(),
    clf_top1_true_arc=results_df["clf_top1_true_arc"].mean(),
    clf_miss_rate=results_df["clf_miss_rate"].mean(),
    clf_continuity=results_df["clf_continuity"].mean(),
    hmm_accuracy=results_df["hmm_accuracy"].mean(),
    hmm_macro_f1=results_df["hmm_macro_f1"].mean(),
    hmm_log_loss=results_df["hmm_log_loss"].mean(),
    hmm_top1_on_route=results_df["hmm_top1_on_route"].mean(),
    hmm_top1_true_arc=results_df["hmm_top1_true_arc"].mean(),
    hmm_miss_rate=results_df["hmm_miss_rate"].mean(),
    hmm_continuity=results_df["hmm_continuity"].mean(),
    hmm_inference_time_s=results_df["hmm_inference_time_s"].sum(),
)
overall_df = pd.DataFrame([overall])
full_df = pd.concat([results_df, overall_df], ignore_index=True)
full_df.to_csv(os.path.join(OUT, "hmm_comparison.csv"), index=False)

summary_cols = ["subject_id", "n_points",
                "clf_top1_on_route", "hmm_top1_on_route",
                "clf_top1_true_arc", "hmm_top1_true_arc",
                "clf_macro_f1", "hmm_macro_f1",
                "clf_continuity", "hmm_continuity"]
print(full_df[summary_cols].round(4).to_string(index=False))

# Save HMM config
hmm_config = dict(
    beta_d=0.06,
    beta_t=0.1,
    not_connected_penalty=15.0,
    eps=1e-12,
    description="Transition = connectivity bonus + distance mismatch penalty + time gap penalty. "
                "Emission = RF P(on_route=1) in log-space. Viterbi with back-pointers.",
)
with open(os.path.join(OUT, "hmm_config.json"), "w") as f:
    json.dump(hmm_config, f, indent=2)


# --------------------------------------------------------------------------- plots
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1. Top-1 on route per trip
ax = axes[0, 0]
x = np.arange(len(results_df))
w = 0.35
ax.bar(x - w/2, results_df["clf_top1_on_route"], w, label="Classifier-only", color="steelblue")
ax.bar(x + w/2, results_df["hmm_top1_on_route"], w, label="Classifier+HMM", color="coral")
ax.set_xticks(x); ax.set_xticklabels(results_df["subject_id"], rotation=45, fontsize=7)
ax.set_ylabel("Top-1 On Route"); ax.set_title("Top-1 On-Route Accuracy per Trip")
ax.legend(fontsize=8)

# 2. Route continuity per trip
ax = axes[0, 1]
ax.bar(x - w/2, results_df["clf_continuity"].fillna(0), w, label="Classifier-only", color="steelblue")
ax.bar(x + w/2, results_df["hmm_continuity"].fillna(0), w, label="Classifier+HMM", color="coral")
ax.set_xticks(x); ax.set_xticklabels(results_df["subject_id"], rotation=45, fontsize=7)
ax.set_ylabel("Route Continuity"); ax.set_title("Route Continuity per Trip")
ax.legend(fontsize=8)

# 3. Macro F1 per trip
ax = axes[1, 0]
ax.bar(x - w/2, results_df["clf_macro_f1"], w, label="Classifier-only", color="steelblue")
ax.bar(x + w/2, results_df["hmm_macro_f1"], w, label="Classifier+HMM", color="coral")
ax.set_xticks(x); ax.set_xticklabels(results_df["subject_id"], rotation=45, fontsize=7)
ax.set_ylabel("Macro F1"); ax.set_title("Candidate-level Macro F1 per Trip")
ax.legend(fontsize=8)

# 4. Overall comparison
ax = axes[1, 1]
metrics = ["Top-1\nOn Route", "Top-1\nTrue Arc", "Macro F1", "Route\nContinuity"]
clf_vals = [overall["clf_top1_on_route"], overall["clf_top1_true_arc"],
            overall["clf_macro_f1"], overall["clf_continuity"] or 0]
hmm_vals = [overall["hmm_top1_on_route"], overall["hmm_top1_true_arc"],
            overall["hmm_macro_f1"], overall["hmm_continuity"] or 0]
xm = np.arange(len(metrics))
ax.bar(xm - w/2, clf_vals, w, label="Classifier-only", color="steelblue")
ax.bar(xm + w/2, hmm_vals, w, label="Classifier+HMM", color="coral")
ax.set_xticks(xm); ax.set_xticklabels(metrics, fontsize=8)
ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title("Overall Comparison")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "hmm_comparison.png"), dpi=150)
print(f"\nPlots saved to {OUT}/hmm_comparison.png")
print("\nTask 2 complete.")
