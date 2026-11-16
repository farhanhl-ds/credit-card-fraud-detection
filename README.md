# Credit Card Fraud Detection

ML Zoomcamp midterm project. A binary classifier that scores a card transaction with the probability of fraud, served as a Flask API in Docker.

## Problem

Fraud is rare (1.67%, about 1 in 60 transactions), so accuracy is meaningless. The model ranks transactions by fraud risk and a decision threshold is chosen for a target recall of 80%, because a missed fraud costs more than a false alarm.

- Main metric: PR-AUC. Secondary: ROC-AUC.
- Validation: 5-fold stratified CV, out-of-fold predictions, same folds everywhere.

## Data

699,960 labelled transactions, 25 features (numeric, boolean and 5 categorical), target `is_fraud`. No missing values, no duplicates. The data is not stored in this repository.

Source: Kaggle Playground Series S6E11 "Predicting Credit Card Fraud". Accept the competition rules on Kaggle, set up the Kaggle API token, then:

```
pip install kaggle
kaggle competitions download -c playground-series-s6e11 -p data
```

Unzip into `data/` so that `data/train.csv` exists.

## Project structure

| File | Purpose |
|---|---|
| `notebook.ipynb` | EDA, baselines, feature engineering, tuning, model selection, threshold |
| `fraud_pipeline.py` | shared code (loading, features, CV, metrics, threshold) |
| `train.py` | trains the selected model with 5-fold CV and saves it |
| `predict.py` | Flask service (`/health`, `/predict`) |
| `params/best_params.json` | tuned hyperparameters chosen in the notebook |
| `models/model.joblib` | saved model used by the service |
| `Dockerfile`, `requirements.txt` | container and dependencies |

## EDA findings

- Strongest signals: `cvv_retry_count`, `merchant_risk_score`, `velocity_score`, `auth_method` (No Authentication has about 6.5% fraud), high-risk merchant categories (Gift Cards, Crypto Exchange, Gaming).
- Numeric features are almost uncorrelated with each other. No train-test shift (adversarial AUC 0.501).

## Models and results

Logistic regression, LightGBM and XGBoost, with and without class weighting; Optuna (40 trials, 30% sample) for the boosted models and a small grid for logistic regression. Out-of-fold scores on all rows:

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| **LightGBM tuned (selected)** | **0.5129** | **0.9472** |
| XGBoost tuned | 0.5116 | 0.9470 |
| XGBoost baseline | 0.5014 | 0.9455 |
| LightGBM baseline | 0.4980 | 0.9453 |
| Logistic regression | 0.4895 | 0.9419 |

Notes: class weighting and engineered features gave no gain over the fold std (about 0.01), so they were dropped. Tuning gave about +0.013 PR-AUC.

Threshold 0.193 (highest precision with recall at least 0.80): precision 0.161, recall 0.800, F1 0.268. The model catches 9,352 of 11,689 frauds and raises about 5 false alarms per caught fraud.

## Run locally

```
pip install -r requirements.txt
python train.py
python predict.py
```

Check the service:

```
curl http://localhost:9696/health
```

`POST /predict` takes one transaction as JSON with the 24 feature fields (without `id`) and returns the fraud probability and the label at the chosen threshold.

## Docker

```
docker build -t fraud-detection .
docker run -p 9696:9696 fraud-detection
```

## Cloud deployment

Deployed as a Docker Space on Hugging Face: `<add Space URL here>`

## Future improvements

Raise the tree cap (tuned models used 1,794 to 1,943 of 2,000 trees), ensemble LightGBM and XGBoost (hill climbing or stacking), pseudo-labelling, seed averaging.