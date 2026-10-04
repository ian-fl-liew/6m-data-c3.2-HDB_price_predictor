"""Train, compare and save the HDB resale price model.

Run with:  python model.py

Steps
1. Load the HDB resale data and engineer features (see hdb_features.py).
2. Split into 80% training / 20% test data.
3. Train ten models: Linear Regression, Random Forest, Gradient Boosting,
   Hist Gradient Boosting (one-hot and native categories), each on raw and
   log price.
4. Compare them with 5-fold cross-validation on the TRAINING data only.
5. Pick the model with the lowest cross-validated RMSE.
6. Report every model's score on the untouched TEST data.
7. Save the chosen model and a model card (metrics + app metadata).
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from hdb_features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    add_features,
    load_data,
)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "hdb_price_model.joblib"
CARD_PATH = MODEL_DIR / "model_card.json"
RANDOM_STATE = 42
TRAIN_MONTHS = 24  # train on recent sales only; None = use every month


def build_pipeline(regressor, dense=False):
    """One-hot encode the text columns, pass numbers through, then fit.

    dense=True is needed for HistGradientBoosting, which can't take sparse input.
    """
    preprocess = ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense),
          CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    return Pipeline([("preprocess", preprocess), ("model", regressor)])


def build_native_pipeline():
    """HistGradientBoosting with its own categorical handling instead of one-hot."""
    preprocess = ColumnTransformer(
        [("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
          CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    model = HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.1,
        early_stopping=True,
        categorical_features=list(range(len(CATEGORICAL_FEATURES))),  # first columns
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def log_price(pipeline):
    """Train on log(price), predict back in S$."""
    return TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp)


def build_models():
    # Each builder returns a fresh object, so a model and its log version never share state.
    def lr():
        return build_pipeline(LinearRegression())

    def rf():
        return build_pipeline(
            RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        )

    # def gb():
        # return build_pipeline(
        #     GradientBoostingRegressor(
        #         n_estimators=3000,
        #         learning_rate=0.05,
        #         max_depth=5,
        #         n_iter_no_change=20,
        #         validation_fraction=0.1,
        #         random_state=RANDOM_STATE,
        #     )
        # )
    def gb():
        return build_pipeline(
            GradientBoostingRegressor(
                n_estimators=3000,
                learning_rate=0.04,
                max_depth=7,
                min_samples_leaf=3,
                n_iter_no_change=20,
                validation_fraction=0.1,
                random_state=RANDOM_STATE,
            )
        )

    def hgb():
        return build_pipeline(
            HistGradientBoostingRegressor(
                max_iter=1000,
                learning_rate=0.1,
                early_stopping=True,
                random_state=RANDOM_STATE,
            ),
            dense=True,
        )

    return {
        "Linear Regression": lr(),
        "Linear Regression (log price)": log_price(lr()),
        "Random Forest": rf(),
        "Random Forest (log price)": log_price(rf()),
        "Gradient Boosting": gb(),
        "Gradient Boosting (log price)": log_price(gb()),
        "Hist Gradient Boosting": hgb(),
        "Hist Gradient Boosting (log price)": log_price(hgb()),
        "Hist GB (native categories)": build_native_pipeline(),
        "Hist GB (native categories) (log price)": log_price(build_native_pipeline()),
    }


def evaluate(models, X_train, X_test, y_train, y_test):
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, pipeline in models.items():
        print(f"Training {name} ...")
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
            n_jobs=1,  # running 5 folds in parallel runs out of RAM
        )

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        y_pred = pipeline.predict(X_test)

        # Trees actually kept by early stopping (None for models without trees)
        reg = pipeline.regressor_ if hasattr(pipeline, "regressor_") else pipeline  # unwrap log-price models
        n_trees = getattr(reg[-1], "n_estimators_", getattr(reg[-1], "n_iter_", None))

        results[name] = {
            "cv_rmse": float(-scores["test_neg_root_mean_squared_error"].mean()),
            "cv_mae": float(-scores["test_neg_mean_absolute_error"].mean()),
            "cv_r2": float(scores["test_r2"].mean()),
            "test_rmse": float(root_mean_squared_error(y_test, y_pred)),
            "test_mae": float(mean_absolute_error(y_test, y_pred)),
            "test_r2": float(r2_score(y_test, y_pred)),
            "fit_seconds": round(fit_seconds, 2),
            "n_trees": n_trees,
        }
    return results


def build_model_card(best_name, results, df, X_train):
    """Everything the Streamlit app needs to know, without loading the CSV."""
    return {
        "best_model": best_name,
        "selection_rule": "lowest 5-fold cross-validated RMSE on the training set",
        "results": results,
        "features": FEATURES,
        "categories": {
            col: sorted(X_train[col].unique().tolist()) for col in CATEGORICAL_FEATURES
        },
        "numeric_ranges": {
            col: {
                "min": float(X_train[col].min()),
                "max": float(X_train[col].max()),
                "median": float(X_train[col].median()),
            }
            for col in NUMERIC_FEATURES
        },
        "data": {
            "rows": len(df),
            "first_month": df["month"].min(),
            "last_month": df["month"].max(),
        },
        "sklearn_version": sklearn.__version__,
    }


def main():
    df = add_features(load_data(last_n_months=TRAIN_MONTHS))
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}\n")

    models = build_models()
    results = evaluate(models, X_train, X_test, y_train, y_test)

    best_name = min(results, key=lambda name: results[name]["cv_rmse"])
    for name in results:
        results[name]["chosen"] = name == best_name

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(models[best_name], MODEL_PATH, compress=3) # Save the best model to disk
    card = build_model_card(best_name, results, df, X_train)
    CARD_PATH.write_text(json.dumps(card, indent=2) + "\n")

    table = pd.DataFrame(results).T.drop(columns="chosen")
    with pd.option_context("display.float_format", "{:,.3f}".format):
        print("\nModel comparison (errors in S$):")
        print(table.to_string())

    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"\nChosen model: {best_name} (lowest cross-validated RMSE)")
    print(f"Saved models/{MODEL_PATH.name} ({size_mb:.1f} MB) and models/{CARD_PATH.name}")


if __name__ == "__main__":
    main()
