# Diabetes Prediction API — End-to-End MLOps Pipeline

[![CI/CD](https://github.com/iqranishath/diabetes-prediction-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/iqranishath/diabetes-prediction-mlops/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.14-blue)
![Docker](https://img.shields.io/badge/docker-iqranishath%2Fdiabetes--prediction--model-2496ED)

A machine learning model that predicts whether a patient is diabetic from five health metrics, served as a REST API, packaged in Docker, and shipped automatically through a CI/CD pipeline.

The focus of this project is not the model itself but everything around it: making a model reproducible, testable, deployable, and automatically delivered every time the code changes.

---

## Architecture

```
train.py ──► diabetes_model.pkl ──► FastAPI (main.py) ──► Docker image
                                                              │
   git push ──► GitHub Actions: install → train → test ──► build & push ──► Docker Hub
```

1. **Training** — `train.py` downloads the dataset, trains a Random Forest classifier, evaluates it on a held-out test set, and saves the model.
2. **Serving** — `main.py` exposes the model through a FastAPI endpoint with input validation.
3. **Packaging** — the Dockerfile installs pinned dependencies and trains the model inside the image, so no binary model file is committed to Git.
4. **Automation** — on every push to `main`, GitHub Actions retrains the model and runs the tests. Only if the tests pass is the image built and published to Docker Hub.

---

## Model

**Dataset:** Pima Indians Diabetes Dataset (768 patients)
**Features:** Pregnancies, Glucose, BloodPressure, BMI, Age
**Algorithm:** Random Forest Classifier (scikit-learn)
**Split:** 80/20, stratified, fixed random seed for reproducibility

### Results (154 test patients)

| Metric | Value |
|---|---|
| Accuracy | **0.766** |
| Baseline accuracy (always predict "not diabetic") | 0.649 |
| Precision — diabetic | 0.69 |
| Recall — diabetic | 0.61 |
| F1 — diabetic | 0.65 |

The model beats the naive baseline by about 12 percentage points. The main weakness is **recall for diabetic patients (0.61)**: the model misses roughly 4 in 10 diabetic cases. In a screening context, a missed case is costlier than a false alarm, so improving recall is the first priority for the next iteration (see *Next steps*).

---

## Run it

### Option 1 — Docker Hub (one command)

```bash
docker run -d -p 8000:8000 --name diabetes-container iqranishath/diabetes-prediction-model:latest
```

Open the interactive API docs at **http://127.0.0.1:8000/docs**.

### Option 2 — Run locally

```bash
git clone https://github.com/iqranishath/diabetes-prediction-mlops.git
cd diabetes-prediction-mlops

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
python train.py
uvicorn main:app --reload
```

### Option 3 — Build the image yourself

```bash
docker build -t diabetes-prediction-model .
docker run -d -p 8000:8000 --name diabetes-container diabetes-prediction-model
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Returns a diabetes prediction |
| GET | `/docs` | Interactive Swagger UI |

**Request**

```json
POST /predict
{
  "Pregnancies": 2,
  "Glucose": 130,
  "BloodPressure": 70,
  "BMI": 28.5,
  "Age": 45
}
```

**Response**

```json
{ "diabetic": true }
```

Invalid input (for example, text instead of a number) is rejected with a `422` error before it reaches the model.

---

## Tests

```bash
pytest
```

Three automated tests check that the API is reachable, that `/predict` returns a valid boolean prediction, and that invalid input is rejected. They run on every push as part of the pipeline.

---

## CI/CD pipeline

Defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

| Job | Runs on | Steps |
|---|---|---|
| `test` | every push and pull request | install dependencies → train model → run tests |
| `build-and-push` | pushes to `main`, only if `test` passes | build Docker image → push to Docker Hub |

Each image is published with two tags: `latest` and the Git commit SHA, so every deployed version can be traced back to the exact code that produced it. Docker Hub credentials are stored as encrypted GitHub secrets.

---

## Project structure

```
├── .github/workflows/ci.yml   # CI/CD pipeline
├── Dockerfile                 # Container definition
├── .dockerignore
├── requirements.txt           # Pinned dependencies
├── train.py                   # Training + evaluation
├── main.py                    # FastAPI app
└── test_main.py               # API tests
```

---

## Limitations

- **Small dataset** (768 patients from a single population), so results may not generalise.
- **Not for medical use.** This is a learning and portfolio project, not a clinical tool.
- **No range validation yet.** The API checks data types but accepts unrealistic values (e.g. negative glucose).
- **Limited recall** for diabetic patients, as described above.

## Next steps

- Improve recall with `class_weight="balanced"` or a tuned decision threshold
- Add realistic value ranges to input validation (Pydantic `Field` constraints)
- Track experiments and compare models with MLflow
- Deploy the container to a cloud service for a public URL

---

## Credits

Based on [Syedlukhman/MLops-diabetes-predicting-model](https://github.com/Syedlukhman/MLops-diabetes-predicting-model), rebuilt and extended with:

- Model evaluation against a baseline (accuracy, precision, recall)
- Automated API tests
- Pinned dependencies for reproducible builds
- Model trained inside the Docker image instead of committing the binary
- A working GitHub Actions CI/CD pipeline publishing to Docker Hub

## Author

**Iqra** · [GitHub](https://github.com/iqranishath)
