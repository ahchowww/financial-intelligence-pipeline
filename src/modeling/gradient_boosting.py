from typing import Sequence

import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from src.processing.features import(
    COMPACT_FINANCIAL_FEATURE_COLUMNS,
    GROWTH_FEATURE_COLUMNS,
)


def train_gradient_boosting_revenue_model(
        train: pd.DataFrame,
        feature_columns: Sequence[str] = COMPACT_FINANCIAL_FEATURE_COLUMNS,
) -> GradientBoostingRegressor:
    """
    Train a Gradient Boosting model for next-quarter revenue forecasting.
    """

    feature_columns = list(feature_columns)

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

    x_train = train[feature_columns]
    y_train = train["target_revenue"]

    if x_train.isna().any().any():
        raise ValueError(
            "Training features contain missing values."
        )

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    )

    model.fit(x_train, y_train)

    return model


def predict_gradient_boosting_revenue(
        model: GradientBoostingRegressor,
        df: pd.DataFrame,
        feature_columns: Sequence[str] = COMPACT_FINANCIAL_FEATURE_COLUMNS,
) -> pd.Series:

    x = df[list(feature_columns)]

    predictions = model.predict(x)

    return pd.Series(
        predictions,
        index=df.index,
        name="predicted_revenue",
    )


def train_gradient_boosting_growth_model(
        train: pd.DataFrame,
) -> GradientBoostingRegressor:
    """
    Train Gradient Boosting to predict next-quarter revenue growth.
    """
    required_columns = (
        set(GROWTH_FEATURE_COLUMNS) | {"target_revenue_growth"}
    )

    missing_columns = (
        required_columns - set(train.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    x_train = train[GROWTH_FEATURE_COLUMNS]
    y_train = train["target_revenue_growth"]

    if x_train.isna().any().any():
        raise ValueError(
            "Training features contain missing values."
        )

    if y_train.isna().any():
        raise ValueError(
            "Training target contains missing values."
        )

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    )

    model.fit(x_train, y_train)

    return model


def predict_gradient_boosting_growth(
        model: GradientBoostingRegressor,
        df: pd.DataFrame,
) -> pd.Series:
    """
    Predict next-quarter revenue growth.
    """

    x = df[GROWTH_FEATURE_COLUMNS]

    if x.isna().any().any():
        raise ValueError(
            "Prediction features contain missing values."
        )

    predictions = model.predict(x)

    return pd.Series(
        predictions,
        index=df.index,
        name="predicted_revenue_growth",
    )


def train_gradient_boosting_log_growth_model(
        train: pd.DataFrame,
) -> GradientBoostingRegressor:
    """
    Train Gradient Boosting to predict next-quarter log revenue growth.
    """

    required_columns = (
        set(GROWTH_FEATURE_COLUMNS) | {"target_log_revenue_growth"}
    )

    missing_columns = (
        required_columns - set(train.columns)
    ) 

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(required_columns)}"
        )

    x_train = train[GROWTH_FEATURE_COLUMNS]
    y_train = train["target_log_revenue_growth"]

    if x_train.isna().any().any():
        raise ValueError(
            "Training features contain missing values."
        )

    if y_train.isna().any():
        raise ValueError(
            "Training target contain missing values."
        )

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    )

    model.fit(x_train, y_train)

    return model


def predict_gradient_boosting_log_growth(
        model: GradientBoostingRegressor,
        df: pd.DataFrame,
) -> pd.Series:
    """
    Predict next-quarter log revenue growth
    """

    x = df[GROWTH_FEATURE_COLUMNS]

    if x.isna().any().any():
        raise ValueError(
            "Prediction features contain missing values."
        )

    predictions = model.predict(x)

    return pd.Series(
        predictions,
        index=df.index,
        name="predicted_log_revenue_growth",
    )
    