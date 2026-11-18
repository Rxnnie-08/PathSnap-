"""
Task 1: Bayesian Hyperparameter Optimisation for Random Forest.

Uses Optuna with TPE sampler to optimise CV Macro F1 under GroupKFold
(by subject / trip ID).  Compares the untuned model with the tuned model
on the student-test set and saves all results.

Run from the repo root:
    python models/04_Bayesian_Optimization_RandomForest.py

Outputs (written to models/results/):
    RandomForest_Bayesian_Tuned.pkl      tuned model
    rf_bayes_best_params.json             best hyperparameters
    rf_bayes_trials.csv                   every Optuna trial
    rf_bayes_comparison.csv               untuned vs tuned comparison
    rf_bayes_comparison.png               comparison plots
"""

import json, os, pickle, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             precision_score, recall_score)
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROCESSED = "preProcessing"
OUT = "models/results"
N_FOLDS = 5
N_TRIALS = 30
SEED = 42
os.makedirs(OUT, exist_ok=True)


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


Xtr, mtr, FEATURES = load_split("train")
Xva, mva, _ = load_split("val")
Xte, mte, _ = load_split("test")
print(f"train {Xtr.shape} | val {Xva.shape} | test {Xte.shape}")

# CV pool = train + val subjects (78 subjects)
Xcv = np.vstack([Xtr, Xva])
mcv = pd.concat([mtr, mva], ignore_index=True)
ycv = mcv["y"].to_numpy()
groups = mcv["subject_id"].to_numpy()
print(f"CV pool: {len(set(groups))} subjects, {len(Xcv)} rows")


# --------------------------------------------------------------------------- metrics
def candidate_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return dict(
        accuracy=accuracy_score(y, pred),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        macro_f1=f1_score(y, pred, average="macro", zero_division=0),
        log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
    )


def point_metrics(meta, p):
    """Top-1 accuracy: for each GPS point, take the highest-scoring candidate."""
    d = meta.assign(score=p).sort_values(["subject_id", "point_idx"]).reset_index(drop=True)
    g = d.groupby(["subject_id", "point_idx"], sort=False)
    top = d.loc[g["score"].idxmax()].reset_index(drop=True)
    has_true = g["y"].max().to_numpy()
    return dict(
        n_points=len(top),
        top1_on_route=float(top["y"].mean()),
        top1_true_arc=float(top["y_true_arc"].mean()),
        candidate_miss_rate=float(1 - has_true.mean()),
    )


# --------------------------------------------------------------------------- objective
def objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        max_depth=trial.suggest_int("max_depth", 8, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50),
        max_features=trial.suggest_float("max_features", 0.2, 1.0),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
        max_samples=trial.suggest_float("max_samples", 0.1, 0.4),
        n_jobs=-1,
        random_state=SEED,
    )
    scores = []
    for tr_idx, va_idx in GroupKFold(n_splits=N_FOLDS).split(Xcv, ycv, groups):
        m = RandomForestClassifier(**params)
        m.fit(Xcv[tr_idx], ycv[tr_idx])
        p = m.predict_proba(Xcv[va_idx])[:, 1]
        pred = (p >= 0.5).astype(int)
        scores.append(f1_score(ycv[va_idx], pred, average="macro", zero_division=0))
    cv_f1 = float(np.mean(scores))
    trial.set_user_attr("cv_f1_std", float(np.std(scores)))
    trial.set_user_attr("cv_f1_folds", [float(s) for s in scores])
    return cv_f1


# --------------------------------------------------------------------------- run optimisation
print(f"\nStarting Bayesian optimisation: {N_TRIALS} trials, TPE sampler, "
      f"GroupKFold({N_FOLDS}) by subject, objective = CV Macro F1")

sampler = TPESampler(seed=SEED)
study = optuna.create_study(direction="maximize", sampler=sampler, study_name="rf_bayes")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# save every trial
trial_rows = []
for t in study.trials:
    trial_rows.append(dict(
        trial=t.number,
        value=t.value,
        **t.params,
        cv_f1_std=t.user_attrs.get("cv_f1_std", None),
    ))
trials_df = pd.DataFrame(trial_rows)
trials_df.to_csv(os.path.join(OUT, "rf_bayes_trials.csv"), index=False)
print(f"\nBest trial: #{study.best_trial.number}")
print(f"Best CV Macro F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

with open(os.path.join(OUT, "rf_bayes_best_params.json"), "w") as f:
    json.dump(study.best_params, f, indent=2)


# --------------------------------------------------------------------------- retrain tuned model
best = dict(study.best_params)
best.update(n_jobs=-1, random_state=SEED)
print("\nRetraining tuned Random Forest on train split...")
t0 = time.perf_counter()
tuned_model = RandomForestClassifier(**best)
tuned_model.fit(Xtr, mtr["y"].to_numpy())
tuned_train_time = time.perf_counter() - t0

with open(os.path.join(OUT, "RandomForest_Bayesian_Tuned.pkl"), "wb") as f:
    pickle.dump(tuned_model, f)
print(f"Tuned model saved to {OUT}/RandomForest_Bayesian_Tuned.pkl ({tuned_train_time:.1f}s)")


# --------------------------------------------------------------------------- evaluate both
def evaluate(model, Xte, mte, label):
    t0 = time.perf_counter()
    p = model.predict_proba(Xte)[:, 1]
    inf_time = time.perf_counter() - t0
    cm = candidate_metrics(mte["y"].to_numpy(), p)
    pm = point_metrics(mte, p)
    return dict(
        Model=label,
        CV_Macro_F1=study.best_value if "Tuned" in label else None,
        Test_Accuracy=cm["accuracy"],
        Test_Macro_F1=cm["macro_f1"],
        Test_Precision=cm["precision"],
        Test_Recall=cm["recall"],
        Test_Log_Loss=cm["log_loss"],
        Test_Top1_on_Route=pm["top1_on_route"],
        Test_Top1_True_Arc=pm["top1_true_arc"],
        Test_Candidate_Miss_Rate=pm["candidate_miss_rate"],
        Inference_Time_s=inf_time,
        Training_Time_s=tuned_train_time if "Tuned" in label else None,
    )


# load untuned model
with open(os.path.join(OUT, "RandomForest.pkl"), "rb") as f:
    untuned_model = pickle.load(f)

print("\nEvaluating untuned model on student-test...")
untuned = evaluate(untuned_model, Xte, mte, "RandomForest (untuned)")
print("Evaluating tuned model on student-test...")
tuned = evaluate(tuned_model, Xte, mte, "RandomForest (tuned)")

# also get CV F1 for untuned (from existing cv_folds.csv if available)
cv_path = os.path.join(OUT, "RandomForest_cv_folds.csv")
if os.path.exists(cv_path):
    cv_df = pd.read_csv(cv_path)
    untuned["CV_Macro_F1"] = float(cv_df["macro_f1"].mean())

comp_df = pd.DataFrame([untuned, tuned]).set_index("Model")
comp_df.to_csv(os.path.join(OUT, "rf_bayes_comparison.csv"))
print("\n=== Comparison (student-test) ===")
print(comp_df.round(4).to_string())


# --------------------------------------------------------------------------- plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 1. Optimisation history
ax = axes[0]
ax.plot(range(len(study.trials)), [t.value for t in study.trials], "o-", ms=2, alpha=0.6)
ax.axhline(y=study.best_value, color="r", ls="--", label=f"Best = {study.best_value:.4f}")
ax.set_xlabel("Trial"); ax.set_ylabel("CV Macro F1"); ax.set_title("Optuna Optimisation History")
ax.legend(fontsize=8)

# 2. Parameter importance
ax = axes[1]
try:
    importances = optuna.importance.get_param_importances(study)
    names = list(importances.keys())[::-1]
    vals = list(importances.values())[::-1]
    ax.barh(names, vals, color="steelblue")
    ax.set_xlabel("Importance"); ax.set_title("Hyperparameter Importance")
except Exception:
    ax.text(0.5, 0.5, "Not available", ha="center", va="center", transform=ax.transAxes)

# 3. Untuned vs tuned comparison
ax = axes[2]
metrics_to_plot = ["Test_Accuracy", "Test_Macro_F1", "Test_Top1_on_Route", "Test_Top1_True_Arc"]
x = np.arange(len(metrics_to_plot))
w = 0.35
untuned_vals = [untuned.get(m, 0) or 0 for m in metrics_to_plot]
tuned_vals = [tuned.get(m, 0) or 0 for m in metrics_to_plot]
ax.bar(x - w/2, untuned_vals, w, label="Untuned", color="steelblue")
ax.bar(x + w/2, tuned_vals, w, label="Tuned", color="coral")
ax.set_xticks(x); ax.set_xticklabels([m.replace("Test_", "") for m in metrics_to_plot], rotation=20, fontsize=8)
ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title("Untuned vs Tuned (Student-Test)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "rf_bayes_comparison.png"), dpi=150)
print(f"\nPlots saved to {OUT}/rf_bayes_comparison.png")
print("\nTask 1 complete.")
