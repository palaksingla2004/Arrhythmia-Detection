from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class ClassicalModelArtifact:
    name: str
    path: Path
    validation_f1: float


def _maybe_xgboost(random_state: int = 42):
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=4,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        return None


def _maybe_lightgbm(random_state: int = 42):
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            num_class=4,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.8,
            random_state=random_state,
        )
    except Exception:
        return None


def build_classical_model_zoo(random_state: int = 42) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, n_jobs=-1))]
        ),
        "svm_rbf": Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(C=5.0, gamma="scale", probability=True))]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "knn": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=11))]),
        "bagging": BaggingClassifier(
            estimator=RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
            n_estimators=8,
            random_state=random_state,
            n_jobs=-1,
        ),
        "adaboost": AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=random_state),
    }

    stack_estimators = [
        ("rf", RandomForestClassifier(n_estimators=250, random_state=random_state, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(random_state=random_state)),
        ("lr", LogisticRegression(max_iter=3000)),
    ]
    models["stacking"] = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=LogisticRegression(max_iter=3000),
        passthrough=True,
        n_jobs=-1,
    )

    xgb = _maybe_xgboost(random_state=random_state)
    if xgb is not None:
        models["xgboost"] = xgb

    lgbm = _maybe_lightgbm(random_state=random_state)
    if lgbm is not None:
        models["lightgbm"] = lgbm

    return models


def train_classical_suite(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    random_state: int = 42,
) -> list[ClassicalModelArtifact]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    artifacts: list[ClassicalModelArtifact] = []
    report: dict[str, dict[str, float]] = {}

    models = build_classical_model_zoo(random_state=random_state)
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        f1 = float(f1_score(y_valid, pred, average="weighted"))
        model_path = output_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        artifacts.append(ClassicalModelArtifact(name=name, path=model_path, validation_f1=f1))
        report[name] = {"validation_f1_weighted": f1}

    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return sorted(artifacts, key=lambda a: a.validation_f1, reverse=True)


def load_classical_suite(model_dir: Path) -> list[tuple[str, Any, float]]:
    metric_path = model_dir / "metrics.json"
    metrics = {}
    if metric_path.exists():
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))

    loaded: list[tuple[str, Any, float]] = []
    for model_file in sorted(model_dir.glob("*.joblib")):
        name = model_file.stem
        model = joblib.load(model_file)
        f1 = float(metrics.get(name, {}).get("validation_f1_weighted", 0.0))
        loaded.append((name, model, f1))
    loaded.sort(key=lambda x: x[2], reverse=True)
    return loaded


def classical_ensemble_predict_proba(
    models: list[tuple[str, Any, float]],
    X: np.ndarray,
) -> np.ndarray:
    if not models:
        raise ValueError("No classical models loaded for prediction.")

    prob_list = []
    weights = []
    for _, model, weight in models:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
        else:
            pred = model.predict(X)
            classes = np.unique(pred)
            proba = np.zeros((len(pred), int(classes.max()) + 1), dtype=np.float32)
            proba[np.arange(len(pred)), pred.astype(int)] = 1.0
        prob_list.append(proba)
        weights.append(max(weight, 1e-3))

    weights_arr = np.asarray(weights, dtype=np.float32)
    weights_arr = weights_arr / weights_arr.sum()
    stacked = np.stack(prob_list, axis=0)
    return np.tensordot(weights_arr, stacked, axes=(0, 0))
