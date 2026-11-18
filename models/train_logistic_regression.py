"""
Model 1: Logistic Regression (linear family) for PathSnap candidate scoring.
Save as models/train_logistic_regression.py

Reads the teammate's processed folder:
    <root>/meta.json
    <root>/train/candidates.csv.gz, <root>/train/points.csv.gz      (80 training trips)
    <root>/val/candidates.csv.gz,   <root>/val/points.csv.gz        (development comparison set)
    <root>/test/candidates.csv.gz,  <root>/test/points.csv.gz       (optional test evaluation set)

Task : binary classification per (GPS point, candidate arc): is_true_arc = 1 if it is the reference arc.
CV   : 5-fold GroupKFold grouped by trip (subject_id) on the training trips.
Final: fit on all training trips, evaluate on val and test (if present).

Install: pip install scikit-learn pandas numpy joblib
Run:     python train_logistic_regression.py --root "C:\\path\\to\\processed" --out-models models/artifacts --out-results results
"""
import os, json, time, argparse
import numpy as np
import pandas as pd
import joblib, sklearn
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, accuracy_score, precision_score, recall_score, confusion_matrix

SEED = 42
LABEL = "is_true_arc"
GROUP = "subject_id"                                   # one id per trip
KEY = ["subject_id", "seg_id", "point_idx"]            # identifies one GPS observation
FORBIDDEN = {"on_route", "is_true_arc", "arc_id", "point_idx", "seg_id", "subject_id"}  # leakage / id columns


def load_split(root, split, feats):
    cols = list(dict.fromkeys(feats + [LABEL] + KEY))
    cand = pd.read_csv(os.path.join(root, split, "candidates.csv.gz"), usecols=cols)
    pts = pd.read_csv(os.path.join(root, split, "points.csv.gz"), usecols=KEY)
    assert not pts.duplicated(KEY).any(), f"{split}: observation key {KEY} is not unique in points"
    n_pts = pts.groupby(GROUP).size()                  # GPS observations per trip
    return cand, n_pts


def make_model(class_weight):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),                   # fitted on training folds only
        ("clf", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
                                   class_weight=class_weight, random_state=SEED)),
    ])


def obs_top1(df, proba, n_points):
    """Observation-level Top-1: per GPS point, take the highest-scoring candidate and compare with the
    reference. Points whose reference arc is not among the candidates count as wrong (candidate miss)."""
    d = df[KEY + [LABEL]].copy()
    d["p"] = proba
    best = d.loc[d.groupby(KEY)["p"].idxmax()]
    correct = int(best[LABEL].sum())
    pts_with_positive = int(d.groupby(KEY)[LABEL].max().sum())
    return correct / n_points, 1.0 - pts_with_positive / n_points


def evaluate(y, proba, df, n_points):
    pred = (proba >= 0.5).astype(int)                  # fixed threshold, not tuned
    top1, miss = obs_top1(df, proba, n_points)
    return dict(
        macro_f1=f1_score(y, pred, average="macro"),
        accuracy=accuracy_score(y, pred),
        precision_pos=precision_score(y, pred, zero_division=0),
        recall_pos=recall_score(y, pred, zero_division=0),
        log_loss=log_loss(y, np.clip(proba, 1e-15, 1 - 1e-15)),
        top1_obs=top1,
        candidate_miss_rate=miss,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="processed folder containing meta.json, train/, val/, test/")
    ap.add_argument("--out-models", default="models/artifacts")
    ap.add_argument("--out-results", default="results")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out_models, exist_ok=True)
    os.makedirs(args.out_results, exist_ok=True)

    feats = json.load(open(os.path.join(args.root, "meta.json")))["features"]
    bad = FORBIDDEN & set(feats)
    assert not bad, f"leakage/id columns in feature list: {bad}"

    tr, tr_n = load_split(args.root, "train", feats)
    va, va_n = load_split(args.root, "val", feats)
    
    # Check if test split exists
    has_test = os.path.exists(os.path.join(args.root, "test"))
    if has_test:
        te, te_n = load_split(args.root, "test", feats)

    print(f"features ({len(feats)}): {feats}")
    print(f"train: {len(tr):,} candidates | {tr[GROUP].nunique()} trips | {int(tr_n.sum()):,} GPS points | "
          f"positive rate {tr[LABEL].mean():.4f}")
    print(f"val  : {len(va):,} candidates | {va[GROUP].nunique()} trips | {int(va_n.sum()):,} GPS points | "
          f"positive rate {va[LABEL].mean():.4f}")
    if has_test:
        print(f"test : {len(te):,} candidates | {te[GROUP].nunique()} trips | {int(te_n.sum()):,} GPS points | "
              f"positive rate {te[LABEL].mean():.4f}")
    print("NaN in features (train):", int(tr[feats].isna().sum().sum()))

    X, y, g = tr[feats].to_numpy(), tr[LABEL].to_numpy().astype(int), tr[GROUP].to_numpy()
    Xv, yv = va[feats].to_numpy(), va[LABEL].to_numpy().astype(int)
    if has_test:
        Xt, yt = te[feats].to_numpy(), te[LABEL].to_numpy().astype(int)

    configs = {"logreg_default": None, "logreg_balanced": "balanced"}
    summary, fold_rows, test_summary = [], [], []
    
    for name, cw in configs.items():
        for k, (a, b) in enumerate(GroupKFold(n_splits=args.folds).split(X, y, g)):
            m = make_model(cw).fit(X[a], y[a])
            proba = m.predict_proba(X[b])[:, 1]
            val_fold = tr.iloc[b]
            n_pts = int(tr_n.reindex(val_fold[GROUP].unique()).sum())
            r = evaluate(y[b], proba, val_fold, n_pts)
            r.update(config=name, fold=k)
            fold_rows.append(r)
        cv = pd.DataFrame([r for r in fold_rows if r["config"] == name])

        t0 = time.perf_counter()
        final = make_model(cw).fit(X, y)
        train_s = time.perf_counter() - t0
        
        # Evaluate on Validation
        t0 = time.perf_counter()
        pv = final.predict_proba(Xv)[:, 1]
        infer_s = time.perf_counter() - t0
        res = evaluate(yv, pv, va, int(va_n.sum()))

        summary_row = dict(
            model=name, family="Linear", class_weight=str(cw), C=1.0, penalty="l2", seed=SEED,
            cv_macro_f1_mean=cv.macro_f1.mean(), cv_macro_f1_std=cv.macro_f1.std(),
            cv_top1_mean=cv.top1_obs.mean(), cv_log_loss_mean=cv.log_loss.mean(),
            train_time_s=train_s, infer_ms_per_1k_cands=infer_s * 1e6 / len(va),
            **{f"val_{k}": v for k, v in res.items()},
        )

        # Evaluate on Test (if folder exists)
        if has_test:
            pt = final.predict_proba(Xt)[:, 1]
            res_test = evaluate(yt, pt, te, int(te_n.sum()))
            summary_row.update({f"test_{k}": v for k, v in res_test.items()})

            cm_test = confusion_matrix(yt, (pt >= 0.5).astype(int))
            pd.DataFrame(cm_test, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
                os.path.join(args.out_results, f"{name}_test_confusion.csv"))
            
            test_summary.append({
                "model": name,
                "class_weight": str(cw),
                **{f"test_{k}": v for k, v in res_test.items()}
            })

        summary.append(summary_row)

        cm = confusion_matrix(yv, (pv >= 0.5).astype(int))
        pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
            os.path.join(args.out_results, f"{name}_val_confusion.csv"))
        
        joblib.dump({"model": final, "features": feats, "seed": SEED, "class_weight": cw,
                     "sklearn_version": sklearn.__version__},
                    os.path.join(args.out_models, f"{name}.joblib"))
        
        print(f"\n[{name}] CV macro F1 {cv.macro_f1.mean():.4f} +/- {cv.macro_f1.std():.4f} | "
              f"val macro F1 {res['macro_f1']:.4f} | top1 {res['top1_obs']:.4f} | "
              f"log loss {res['log_loss']:.4f} | miss {res['candidate_miss_rate']:.4f} | train {train_s:.1f}s")
        if has_test:
            print(f"       -> test macro F1 {res_test['macro_f1']:.4f} | test accuracy {res_test['accuracy']:.4f} | test top1 {res_test['top1_obs']:.4f}")

    pd.DataFrame(fold_rows).to_csv(os.path.join(args.out_results, "logreg_cv_folds.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(args.out_results, "logreg_summary.csv"), index=False)
    
    if has_test and test_summary:
        pd.DataFrame(test_summary).to_csv(os.path.join(args.out_results, "logreg_test_summary.csv"), index=False)

    with open(os.path.join(args.out_results, "logreg_config.json"), "w") as f:
        json.dump(dict(seed=SEED, features=feats, folds=args.folds, sklearn=sklearn.__version__,
                       configs={k: str(v) for k, v in configs.items()}), f, indent=2)
    print("\nsaved summary, per-fold CV, confusion matrices, test metrics, config and model artifacts")


if __name__ == "__main__":
    main()