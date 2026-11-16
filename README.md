# Credit Card Fraud Detection

Predict whether a card transaction is fraudulent (`is_fraud`) and serve the model as a web service.

## Problem

A payment processor needs to decide, per transaction and in real time, which ones look fraudulent.
The service returns a fraud probability and a yes/no decision. Transactions flagged as fraud can be
blocked or sent for manual review or step-up verification.

Fraud is rare, so accuracy is meaningless. The project uses:

- **PR-AUC** as the main metric and **ROC-AUC** as a second reference
- **Stratified 5-fold CV**, the same folds for every experiment
- A decision threshold chosen on out-of-fold predictions: the highest precision that still reaches a target recall of 0.80 (missed fraud costs more than a false alarm)

## Dataset

[Kaggle Playground Series S6E11, Predicting Credit Card Fraud](https://www.kaggle.com/competitions/playground-series-s6e11/data)
(synthetic data, CC BY 4.0). `train.csv` has the target `is_fraud`. The data is not committed to this repository.

```bash
# needs a Kaggle API token in ~/.kaggle/kaggle.json and accepted competition rules
bash download_data.sh        # puts train.csv and test.csv in ./data
```

## Approach

1. **EDA**: ranges, missing values, duplicates, target balance, distributions by class, categorical fraud rates, correlations, feature importance (mutual information and LightGBM gain), train vs test shift.
2. **Models**: logistic regression, LightGBM and XGBoost, each with and without imbalance handling (`class_weight` / `scale_pos_weight`).
3. **Feature engineering**: categorical pair combinations and frequency encodings, kept only if the CV gain exceeds the fold standard deviation.
4. **Tuning**: Optuna (TPE) for LightGBM and XGBoost, small grid for logistic regression, all scored by PR-AUC on the same 5 folds.
5. **Final model**: best configuration refit on 100% of the data, threshold stored with the model.

Everything is in `notebook.ipynb`. `train.py` reproduces the final model from the configuration the notebook saves in `params/best_params.json`.

## Results

Numbers come from the notebook (selection table) and `models/metrics.json` (written by `train.py`).

| Model | PR-AUC (CV mean) | ROC-AUC (CV mean) |
|---|---|---|
| Selected model | _fill in after running_ | _fill in after running_ |

At the chosen threshold: precision _fill in_, recall _fill in_ (out-of-fold).

## Project structure

```
README.md              this file
notebook.ipynb         EDA, model selection, tuning
train.py               trains the final model, saves models/model.joblib
predict.py             Flask web service (loads the model and serves /predict)
fraud_pipeline.py      shared preprocessing, models, CV and metrics
params/                best_params.json written by the notebook
requirements.txt       full environment (notebook, training, deployment)
requirements-serve.txt minimal environment for the service and the Docker image
Dockerfile             container for the service
download_data.sh       downloads the data from Kaggle
deploy_hf.py           deploys the container to a Hugging Face Space
FUTURE_IMPROVEMENTS.md planned experiments
```

## Run it

### 1. Environment

Tested with Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data, notebook and training

```bash
bash download_data.sh
jupyter lab notebook.ipynb         # optional, writes params/best_params.json
python train.py                    # writes models/model.joblib, metrics.json, example_request.json
```

The notebook has `N_TRIALS` and `TUNE_FRAC` at the top to shorten the Optuna search.
`train.py` runs with defaults if `params/best_params.json` is missing.

### 3. Web service

```bash
gunicorn --bind 0.0.0.0:9696 predict:app
```

```bash
curl http://localhost:9696/health

curl -X POST -H "Content-Type: application/json" \
     -d @models/example_request.json \
     http://localhost:9696/predict
```

`models/example_request.json` is a real training row written by `train.py`. Response:

```json
{"fraud_probability": 0.0123, "is_fraud": false, "threshold": 0.4321}
```

Missing features are imputed and listed in a `missing_features` field.

### 4. Docker

Train first, because the image copies `models/model.joblib`.

```bash
docker build -t fraud-service .
docker run -it --rm -p 9696:9696 fraud-service
```

Then use the same `curl` commands as above.

### 5. Deployment (Hugging Face Space)

The same container runs as a Docker Space.

```bash
export HF_TOKEN=hf_xxx                        # token with write access
python deploy_hf.py --space <username>/credit-card-fraud
```

Live service: _add the Space URL here_
Screenshot or video of a prediction: _add the link here_
