"""Shared building blocks used by the notebook, train.py and predict.py.

Everything that must be identical between experiments and the deployed
service lives here: preprocessing, model factories, cross-validation,
metrics and threshold selection.
"""
from __future__ import annotations

import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

SEED = 42
N_FOLDS = 5
TARGET = "is_fraud"
ID_COL = "id"
MODELS = ("logreg", "lightgbm", "xgboost")
EARLY_STOPPING_ROUNDS = 50


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_train(path):
    """Read the training csv and return (features DataFrame, target array)."""
    df = pd.read_csv(path)
    y = df[TARGET].astype(int).to_numpy()
    X = df.drop(columns=[c for c in (TARGET, ID_COL) if c in df.columns])
    return coerce_types(X), y


def coerce_types(X):
    """Booleans become ints so every column is either numeric or categorical."""
    X = X.copy()
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    return X


def pos_weight(y):
    """Negative to positive ratio, the classic scale_pos_weight value."""
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
class FeatureAdder(BaseEstimator, TransformerMixin):
    """Pairwise categorical combinations and frequency encodings.

    Everything is learned on the training fold only, so there is no leakage.
    """

    def __init__(self, max_cat_cardinality=30, max_pairs=15):
        self.max_cat_cardinality = max_cat_cardinality
        self.max_pairs = max_pairs

    def fit(self, X, y=None):
        cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        low = sorted(
            (c for c in cats if X[c].nunique() <= self.max_cat_cardinality),
            key=lambda c: X[c].nunique(),
        )
        self.pairs_ = list(itertools.combinations(low, 2))[: self.max_pairs]
        self.freq_ = {c: X[c].value_counts(normalize=True).to_dict() for c in cats}
        return self

    def transform(self, X):
        out = X.copy()
        for a, b in self.pairs_:
            left = X[a].astype("string").fillna("missing")
            right = X[b].astype("string").fillna("missing")
            out[f"{a}__{b}"] = (left + "_" + right).astype(object)
        for col, mapping in self.freq_.items():
            out[f"{col}__freq"] = X[col].map(mapping).astype(float).fillna(0.0)
        return out


def build_preprocessor(model_name, use_fe=False):
    """Preprocessing pipeline that depends on the model family.

    logreg: median impute + scale for numerics, one-hot for categoricals.
    trees:  numerics untouched (NaN is handled natively), categoricals are
            ordinal encoded.
    """
    cat_sel = make_column_selector(dtype_exclude="number")
    num_sel = make_column_selector(dtype_include="number")
    cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")

    if model_name == "logreg":
        num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
        cat = Pipeline(
            [
                ("imp", cat_imputer),
                ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.001)),
            ]
        )
        ct = ColumnTransformer([("num", num, num_sel), ("cat", cat, cat_sel)])
    else:
        cat = Pipeline(
            [
                ("imp", cat_imputer),
                ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )
        ct = ColumnTransformer(
            [("num", "passthrough", num_sel), ("cat", cat, cat_sel)],
            verbose_feature_names_out=False,
        )
        ct.set_output(transform="pandas")

    return Pipeline([("fe", FeatureAdder() if use_fe else "passthrough"), ("ct", ct)])


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def make_model(name, params=None, early_stopping=True):
    """Build a classifier. `params` override the defaults below.

    Imbalance is handled through `scale_pos_weight` (GBDT) or `class_weight`
    (logistic regression), both passed in `params`.
    """
    params = dict(params or {})

    if name == "logreg":
        cfg = dict(C=1.0, l1_ratio=0.0, max_iter=1000, class_weight=None)
        cfg.update(params)
        solver = "saga" if cfg["l1_ratio"] > 0 else "lbfgs"
        return LogisticRegression(solver=solver, random_state=SEED, **cfg)

    if name == "lightgbm":
        cfg = dict(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            metric="average_precision",
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
        cfg.update(params)
        return lgb.LGBMClassifier(**cfg)

    if name == "xgboost":
        cfg = dict(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="aucpr",
            random_state=SEED,
            n_jobs=-1,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS if early_stopping else None,
        )
        cfg.update(params)
        return xgb.XGBClassifier(**cfg)

    raise ValueError(f"unknown model: {name}")


def fit_model(name, model, X_train, y_train, X_valid=None, y_valid=None):
    """Fit one model. GBDTs use early stopping when a validation set is given.

    Returns the best number of trees for GBDTs, otherwise None.
    """
    if name == "lightgbm" and X_valid is not None:
        # Only average precision is monitored. The default binary_logloss would
        # stop training immediately when scale_pos_weight is used.
        
        # model.fit(
        #     X_train,
        #     y_train,
        #     eval_X=X_valid,
        #     eval_y=y_valid,
        #     callbacks=[
        #         lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False)
        #     ],
        # )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False)],
        )
        
        return int(model.best_iteration_)
    if name == "xgboost" and X_valid is not None:
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        return int(model.best_iteration) + 1
    model.fit(X_train, y_train)
    return None


# --------------------------------------------------------------------------
# Cross-validation and metrics
# --------------------------------------------------------------------------
def make_folds(y, n_folds=N_FOLDS, seed=SEED):
    """The same stratified folds are used for every model and experiment."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def cv_evaluate(name, X, y, params=None, use_fe=False, n_folds=N_FOLDS, seed=SEED):
    """Stratified k-fold CV. Returns mean/std of PR-AUC and ROC-AUC, the
    average best iteration (GBDT) and out-of-fold probabilities."""
    y = np.asarray(y)
    oof = np.zeros(len(y))
    pr_scores, roc_scores, iters = [], [], []
    for train_idx, valid_idx in make_folds(y, n_folds, seed):
        pre = build_preprocessor(name, use_fe)
        X_tr = pre.fit_transform(X.iloc[train_idx])
        X_va = pre.transform(X.iloc[valid_idx])
        model = make_model(name, params)
        iters.append(fit_model(name, model, X_tr, y[train_idx], X_va, y[valid_idx]))
        proba = model.predict_proba(X_va)[:, 1]
        oof[valid_idx] = proba
        pr_scores.append(average_precision_score(y[valid_idx], proba))
        roc_scores.append(roc_auc_score(y[valid_idx], proba))
    return {
        "pr_auc_mean": float(np.mean(pr_scores)),
        "pr_auc_std": float(np.std(pr_scores)),
        "roc_auc_mean": float(np.mean(roc_scores)),
        "roc_auc_std": float(np.std(roc_scores)),
        "best_iter": None if iters[0] is None else int(round(np.mean(iters))),
        "oof": oof,
    }


def pick_threshold(y, proba, target_recall=0.8):
    """Highest-precision threshold that still reaches the target recall.

    Computed on out-of-fold probabilities. Returns (threshold, precision, recall).
    """
    precision, recall, thresholds = precision_recall_curve(y, proba)
    ok = np.where(recall[:-1] >= target_recall)[0]
    best = ok[np.argmax(precision[:-1][ok])]
    return float(thresholds[best]), float(precision[best]), float(recall[best])


def threshold_report(y, proba, threshold):
    """Metrics at a fixed threshold plus the threshold-free PR-AUC and ROC-AUC."""
    y = np.asarray(y)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pr_auc": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }
