# Future Improvements

Ideas parked for after the midterm, mostly from the NVIDIA article
[The Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques for Tabular Data](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/).

## What the current code already prepares

- `fraud_pipeline.make_folds` always returns the same stratified 5 folds
- the notebook saves out-of-fold probabilities of every model in `artifacts/oof/*.npy`, the raw material for hill climbing and stacking
- all models share one preprocessing and CV code path, so new models are one factory entry away

## Experiments, in the order worth trying

1. **Hill climbing** on out-of-fold predictions: start from the best single model, add the others with weights, keep a change only if CV PR-AUC improves.
2. **Stacking**: a level-2 model (for example logistic regression) trained on the out-of-fold predictions of the level-1 models. Variants: OOF predictions as features, or a second stage trained on residuals.
3. **Pseudo-labeling**: soft labels for `test.csv` from the best model, added to training, then fine-tune on the original labels at the end. With k-fold, compute k sets of pseudo-labels so a validation fold never sees labels from a model trained on itself.
4. **Seed averaging**: average several seeds of the final model, plus retrain on 100% of the data (already done). The cost is N models in the artifact and N times slower inference.
5. **Grid refinement**: a narrow grid around the best Optuna parameters.
6. **More feature engineering**: target-independent aggregates, more category combinations, validated with the rule used in the notebook (gain larger than the fold standard deviation).
7. **GPU acceleration**: cuDF, cuML and GPU backends of XGBoost and LightGBM make deep stacks practical.

Keep these experiments in separate notebooks. The deployed model should stay a single simple pipeline unless an ensemble clearly wins.

## Product and deployment

- choose the threshold from a cost model (cost of a missed fraud vs cost of a review) instead of a fixed recall target
- probability calibration check
- request logging and drift monitoring on the input features
- input validation with a schema (pydantic) in `predict.py`

## Kaggle submission

The competition metric is ROC-AUC on predicted probabilities. The out-of-fold ROC-AUC printed by `train.py` and the notebook is the local score. The submission file needs `id,is_fraud` with probabilities, not labels:

```python
import joblib
import pandas as pd

artifact = joblib.load("models/model.joblib")
test = pd.read_csv("data/test.csv")
X = test[artifact["features"]]
proba = artifact["model"].predict_proba(artifact["preprocessor"].transform(X))[:, 1]
pd.DataFrame({"id": test["id"], "is_fraud": proba}).to_csv("submission.csv", index=False)
```

Notes: the model in the artifact is refit on 100% of the training data. Competition-specific work (public notebooks, leaderboard tricks) should be kept out of the course submission.
