
# PathSnap — GPS Map Matching using ML and HMM

PathSnap converts noisy GPS observations into a predicted sequence of road segments. The project compares machine-learning models, tunes a selected model, and uses a Hidden Markov Model (HMM) with Viterbi decoding to reconstruct a continuous route.

## Pipeline

1. **Preprocessing:** Clean GPS data and generate candidate-road features.
2. **ML training:** Train and compare at least five models on a shared data split.
3. **Bayesian optimization:** Tune the selected model's hyperparameters.
4. **HMM + Viterbi:** Combine model probabilities and road-network connectivity to decode a continuous route.
5. **Visualization:** Compare observed GPS points and predicted routes using Folium.
6. **Deployment:** Run FastAPI on one laptop and Streamlit on another.

## Repository structure

```text
PathSnap-/
├── preProcessing/       # GPS data preprocessing
├── models/              # Model notebooks, training code, optional saved artifacts
├── hmm/                 # HMM / Viterbi decoding
├── backend/             # FastAPI routes and inference code
├── frontend/            # Streamlit user interface and Folium map
├── results/             # Model comparison, graphs, screenshots
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Requires a compatible Python 3 installation. Clone the repository:

```bash
git clone https://github.com/Rxnnie-08/PathSnap-.git
cd PathSnap-
python -m pip install -r requirements.txt
```

Place processed data, road-network files, and any required saved-model artifacts at the paths expected by the code. These may be stored outside GitHub if they are large. Ensure the model uses the same feature order and preprocessing as during training.

## Run the two-laptop demo

**Laptop 1 — API server**

From the repository root:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Test locally at `http://localhost:8000/docs`.

**Laptop 2 — Streamlit UI**

Install the dependencies and configure the UI's API URL using Laptop 1's actual local IP address, e.g. `http://192.168.1.5:8000`. Then run:

```bash
python -m streamlit run frontend/app.py
```

Both laptops must be connected to a network that allows device-to-device communication. Allow inbound TCP port 8000 on Laptop 1 if required by its firewall.

> The commands above assume `backend/main.py` exposes a FastAPI object named `app`, `frontend/app.py` is the Streamlit entry point, and the API URL is configured in the frontend. Adjust filenames and paths if the implementation differs.

## Evaluation

Compare all models on consistent trip-based splits and metrics. Keep evaluation-only trips separate from model fitting and tuning. Include classifier-only versus HMM/Viterbi results, route visualizations, and two-laptop API/UI evidence in `results/` and the final technical report.

## Deliverables

- Public GitHub repository containing code, instructions, and reproducible results.
- One technical report PDF containing methods, model comparisons, maps, and demo screenshots.
=======
PathSnap — GPS Map Matching using ML and HMM
PathSnap converts noisy GPS observations into a predicted sequence of road segments. The project compares machine-learning models, tunes a selected model, and uses a Hidden Markov Model (HMM) with Viterbi decoding to reconstruct a continuous route.
Pipeline
1. Preprocessing: Clean GPS data and generate candidate-road features.
2. ML training: Train and compare at least five models on a shared data split.
3. Bayesian optimization: Tune the selected model's hyperparameters.
4. HMM + Viterbi: Combine model probabilities and road-network connectivity to decode a continuous route.
5. Visualization: Compare observed GPS points and predicted routes using Folium.
6. Deployment: Run FastAPI on one laptop and Streamlit on another.
Repository structure
PathSnap-/
├── preProcessing/       # GPS data preprocessing
├── models/              # Model notebooks, training code, optional saved artifacts
├── hmm/                 # HMM / Viterbi decoding
├── backend/             # FastAPI routes and inference code
├── frontend/            # Streamlit user interface and Folium map
├── results/             # Model comparison, graphs, screenshots
├── requirements.txt
├── .gitignore
└── README.md
Setup
Requires a compatible Python 3 installation. Clone the repository:
git clone https://github.com/Rxnnie-08/PathSnap-.git
cd PathSnap-
python -m pip install -r requirements.txt
Place processed data, road-network files, and any required saved-model artifacts at the paths expected by the code. These may be stored outside GitHub if they are large. Ensure the model uses the same feature order and preprocessing as during training.
Run the two-laptop demo
Laptop 1 — API server

From the repository root:
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
Test locally at http://localhost:8000/docs.

Laptop 2 — Streamlit UI

Install the dependencies and configure the UI's API URL using Laptop 1's actual local IP address, e.g. http://192.168.1.5:8000. Then run:
python -m streamlit run frontend/app.py
Both laptops must be connected to a network that allows device-to-device communication. Allow inbound TCP port 8000 on Laptop 1 if required by its firewall.
The commands above assume backend/main.py exposes a FastAPI object named app, frontend/app.py is the Streamlit entry point, and the API URL is configured in the frontend. Adjust filenames and paths if the implementation differs.

Evaluation
Compare all models on consistent trip-based splits and metrics. Keep evaluation-only trips separate from model fitting and tuning. Include classifier-only versus HMM/Viterbi results, route visualizations, and two-laptop API/UI evidence in results/ and the final technical report.
Deliverables
- Public GitHub repository containing code, instructions, and reproducible results.
- One technical report PDF containing methods, model comparisons, maps, and demo screenshots.
>>>>>>> 5ecfab5f4e0843d8ad2700695b0964e0369dcf7b
