"""Train the final fraud model and save it for the web service.

Steps:
  1. 5-fold stratified CV with the chosen configuration (reports PR-AUC and ROC-AUC)
  2. Pick the decision threshold on out-of-fold probabilities (target recall)
  3. Refit on 100% of the data and save the artifact

The configuration comes from params/best_params.json, written by the
notebook after tuning. Without that file, sensible defaults are used.

Usage:
    python train.py
    python train.py --data data/train.csv --out models --target-recall 0.8
"""
import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd

import fraud_pipeline as fp

log = logging.getLogger("train")

DEFAULT_CONFIG = {
    "model": "lightgbm",
    "use_fe": False,
    "params": {},
}


def load_config(path):
    path = Path(path)
    if path.exists():
        log.info("Using configuration from %s", path)
        return json.loads(path.read_text())
    log.warning("%s not found, using defaults with a balanced class weight", path)
    return dict(DEFAULT_CONFIG)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--params", default="params/best_params.json")
    parser.add_argument("--out", default="models")
    parser.add_argument("--target-recall", type=float, default=0.8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    X, y = fp.load_train(args.data)
    log.info("Loaded %d rows, %d features, fraud rate %.4f", len(X), X.shape[1], y.mean())

    config = load_config(args.params)
    name, use_fe = config["model"], config["use_fe"]
    params = dict(config["params"])
    if not config["params"] and name != "logreg":
        params["scale_pos_weight"] = fp.pos_weight(y)

    # 1. Cross-validation
    log.info("Running %d-fold CV for %s (feature engineering: %s)", fp.N_FOLDS, name, use_fe)
    cv = fp.cv_evaluate(name, X, y, params=params, use_fe=use_fe)
    oof = cv.pop("oof")
    log.info("CV PR-AUC  %.4f +/- %.4f", cv["pr_auc_mean"], cv["pr_auc_std"])
    log.info("CV ROC-AUC %.4f +/- %.4f", cv["roc_auc_mean"], cv["roc_auc_std"])

    # 2. Threshold from out-of-fold predictions
    threshold, _, _ = fp.pick_threshold(y, oof, args.target_recall)
    report = fp.threshold_report(y, oof, threshold)
    log.info(
        "Threshold %.4f gives precision %.3f and recall %.3f (out-of-fold)",
        threshold, report["precision"], report["recall"],
    )

    # 3. Final fit on all data
    final_params = dict(params)
    if cv["best_iter"] is not None:
        final_params["n_estimators"] = cv["best_iter"]
    pre = fp.build_preprocessor(name, use_fe)
    X_all = pre.fit_transform(X)
    model = fp.make_model(name, final_params, early_stopping=False)
    fp.fit_model(name, model, X_all, y)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": name,
        "preprocessor": pre,
        "model": model,
        "threshold": threshold,
        "features": list(X.columns),
        "numeric_features": [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])],
    }
    joblib.dump(artifact, out / "model.joblib")
    (out / "metrics.json").write_text(
        json.dumps({"model": name, "use_fe": use_fe, "cv": cv, "out_of_fold": report}, indent=2)
    )
    example = {k: (None if v != v else v) for k, v in X.iloc[0].to_dict().items()}
    (out / "example_request.json").write_text(json.dumps(example, default=str, indent=2))
    log.info("Saved model, metrics and an example request to %s/", out)


if __name__ == "__main__":
    main()
