Both notebooks ran, and everything you need is in the `train`, `val` and `test` folders.

## Files needed to run the models

| Split | Exact file | Use it for |
|---|---|---|
| Train (62 subjects) | `processed/train/model_data.npz` | Fitting your 5 models |
| Validation (16 subjects) | `processed/val/model_data.npz` | Tuning and comparing models |
| Test (10 subjects) | `processed/test/model_data.npz` | Final scores only, once |

Each `model_data.npz` holds:
- `X`: scaled features (19 columns), the model input
- `y`: target, 1 if the candidate road arc is on the true route
- `groups`: subject id per row, for `GroupKFold`

```python
import numpy as np

tr = np.load("processed/train/model_data.npz", allow_pickle=True)
va = np.load("processed/val/model_data.npz",   allow_pickle=True)
te = np.load("processed/test/model_data.npz",  allow_pickle=True)

X_train, y_train = tr["X"], tr["y"]
X_val,   y_val   = va["X"], va["y"]
X_test,  y_test  = te["X"], te["y"]
groups = tr["groups"]
```

## Everything else in `processed/`

| Item | What it is | Use it for |
|---|---|---|
| `train/`, `val/`, `test/` | Folders for each split (see above) | Training, tuning and final testing |
| `interim/` | Cleaned GPS points and road networks (`interim.pkl`) | Not for training. Needed later for the HMM/Viterbi step |
| `scaler.pkl` | The StandardScaler fit on train | Scaling new data, e.g. in your API, the same way as training |
| `meta.json` | Feature list, settings, and which subjects are in each split | Your report and reproducibility |
| `subject_report.csv` | Per-subject data summary and quality flags | Your report only |
| `*/candidates.csv.gz` | Same rows as `model_data.npz` in readable form | Inspection in pandas |
| `*/points.csv.gz` | Cleaned GPS points with speed, heading and so on | Visualisations and decoding, not training |
