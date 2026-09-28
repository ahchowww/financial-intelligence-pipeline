from typing import Sequence

import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling.linear import (
    DEFAULT_FEATURE_COLUMNS,
)

from src.modeling.metrics import (
    calculate_forecast_metrics,
)


DEFAULT_ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


def select_ridge_alpha(
    train: pd.DataFrame,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    n_splits: int = 5,
) -> tuple[float, pd.DataFrame]:
    """
    Select Ridge regularization strength using
    time-series cross-validation on training data only.

    Lower mean validation MAE is better.
    """

    x = train[list(feature_columns)]

    y = train["target_revenue"]

    splitter = TimeSeriesSplit(
        n_splits=n_splits
    )

    results = []

    for alpha in alphas:
        fold_maes = []

        for (train_indices, validation_indices) in splitter.split(x):

            # Gets the earlier training part
            x_fold_train = x.iloc[train_indices]

            y_fold_train = y.iloc[train_indices]

            # Gets the later validation fold  
            x_fold_validation = x.iloc[validation_indices]

            y_fold_validation = y.iloc[validation_indices]

            model = Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "ridge",
                        Ridge(
                            alpha=alpha
                        ),
                    ),
                ]
            )

            model.fit(
                x_fold_train,
                y_fold_train,
            )

            predictions = model.predict(
                x_fold_validation
            )

            metrics = calculate_forecast_metrics(
                    actual=y_fold_validation,
                    predicted=predictions,
            )

            fold_maes.append(metrics["MAE"])

        results.append(
            {
                "alpha": alpha,
                "mean_cv_mae": sum(fold_maes) / len(fold_maes),
            }
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "mean_cv_mae"
        )
        .reset_index(drop=True)
    )

    best_alpha = float(
        results_df.iloc[0][
            "alpha"
        ]
    )

    return best_alpha, results_df


def train_ridge_revenue_model(
    train: pd.DataFrame,
    alpha: float,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
) -> Pipeline:
    """
    Train Ridge Regression on the complete training period.
    """

    x_train = train[list(feature_columns)]

    y_train = train["target_revenue"]

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=alpha
                ),
            ),
        ]
    )

    model.fit(
        x_train,
        y_train,
    )

    return model


def predict_ridge_revenue(
    model: Pipeline,
    df: pd.DataFrame,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
) -> pd.Series:

    x = df[list(feature_columns)]

    predictions = model.predict(x)

    return pd.Series(
        predictions,
        index=df.index,
        name="predicted_revenue",
    )