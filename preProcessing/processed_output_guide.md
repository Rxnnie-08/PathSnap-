Both notebooks ran, and everything you need is in the `train`, `val` and `test` folders.

| Item | What it is | Use it for |
|---|---|---|
| `train/` | Training split (62 subjects) | Fitting your 5 models |
| `val/` | Validation split (16 subjects) | Tuning and comparing models |
| `test/` | Test split (10 subjects, from `student_test`) | Final scores only, once |
| `interim/` | Cleaned GPS points and road networks (`interim.pkl`) | Not for training. Needed later for the HMM/Viterbi step |
| `scaler.pkl` | The StandardScaler fit on train | Scaling new data, e.g. in your API, the same way as training |
| `meta.json` | Feature list, settings, and which subjects are in each split | Your report and reproducibility |
| `subject_report.csv` | Per-subject data summary and quality flags | Your report only |
