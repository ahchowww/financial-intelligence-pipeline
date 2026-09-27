from typing import Sequence

import pandas as pd
from sklearn.linear_model import LinearRegression


DEFAULT_FEATURE_COLUMNS = [
    "revenue",
    "revenue_lag1",
    "revenue_lag2",
    "revenue_lag3",
    "revenue_lag4",
]


def train_linear_revenue_model(
        train: pd.DataFrame,
        feature_columns : Sequence[str] = DEFAULT_FEATURE_COLUMNS,
) -> LinearRegression:
    """
    Train a linear regression model for next-quarter revenue forecasting.
    """

    required_columns = (
        set(feature_columns) | {"target_revenue"}
    )

    missing_columns = (
        required_columns - set(train.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if train[list(feature_columns)].isna().any().any():
        raise ValueError(
            "Training features contain missing values."
        )

    if train["target_revenue"].isna().any():
        raise ValueError(
            "Training target contains missing values."
        )

    x_train = train[list(feature_columns)]
    y_train = train["target_revenue"]

    model = LinearRegression()

    model.fit(
        x_train,
        y_train,
    )

    return model


def predict_linear_revenue(
        model: LinearRegression,
        df: pd.DataFrame,
        feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
) -> pd.Series:
    """
    Generate next-quarter revenue predictions.
    """

    missing_columns = (
        set(feature_columns) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    x = df[list(feature_columns)]

    if x.isna().any().any():
        raise ValueError(
            "Prediction features contain missing values."
        )

    predictions = model.predict(x)

    return pd.Series(
        predictions,
        index=df.index,
        name="predicted_revenue",
    )