import pandas as pd

def naive_revenue_forecast(
        df: pd.DataFrame,
) -> pd.Series:
    """
    Naive one-step-ahead revenue forecast.

    Prediction:
        revenue(t+1) = revenue(t)
    """

    if "revenue" not in df.columns:
        raise ValueError(
            "Dataset must contain a revenue column."
        )

    return df["revenue"].copy()


def seasonal_naive_revenue_forecast(
        df: pd.DataFrame,
) -> pd.Series:
    """
    Seasonal naive one-step-ahead revenue forecast.

    For quarterly data:
        predicted revenue for quarter t+1 = revenue from the same quarter one year earlier

    ## Each row represent quarter t and predicts quarter t+1,
       corresponds to revenue_lag3.
    """

    required_column = "revenue_lag3"

    if required_column not in df.columns:
        raise ValueError(
            "Dataset must contain revenue_lag3."
        )

    if df[required_column].isna().any():
        raise ValueError(
            "revenue_lag3 contains missing values."
        )

    return df[required_column].copy()


def moving_average_revenue_forecast(
        df: pd.DataFrame,
) -> pd.Series:
    """
    Four-quarter moving-average forecast.

    Predict next-quarter revenue using the mean of current quarter
    and previous 3 quarters.
    """

    required_columns = [
        "revenue",
        "revenue_lag1",
        "revenue_lag2",
        "revenue_lag3",
    ]

    missing_columns = (
        set(required_columns) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Moving-average input contains missing values."
        )

    return df[required_columns].mean(axis=1)

