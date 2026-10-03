import pandas as pd

from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.arima.model import ARIMA


def train_holt_model(
        revenue_series: pd.Series,
):
    """
    Train Holt's linear trend model.

    Holt models:
    - level
    - trend

    It does not model seasonality yet.
    """

    if revenue_series.isna().any():
        raise ValueError(
            "Revenue series contains missing values."
        )

    if len(revenue_series) < 2:
        raise ValueError(
            "Holt requires at least 2 observations."
        )

    model = Holt(
        revenue_series.astype(float),
        initialization_method="estimated",
    )

    fitted_model = model.fit(
        optimized=True,
    )

    return fitted_model


def predict_holt_next_quarter(
        model,
) -> float:
    """
    Forecast one quarter ahead.
    """

    forecast = model.forecast(
        steps=1,
    )

    return float(
        forecast.iloc[0]
    )


def train_arima_model(
        revenue_series: pd.Series,
        order: tuple[int, int, int] = (1, 1, 1),
):
    """
    Train an ARIMA model on historical revenue.

    Default:
    ARIMA(1, 1, 1)

    p = autoregressive order
    d = differencing order
    q = moving average error order
    """

    if revenue_series.isna().any():
        raise ValueError(
            "Revenue series contains missing values."
        )

    if len(revenue_series) < 5:
        raise ValueError(
            "ARIMA requires at least 5 observations."
        )

    model = ARIMA(
        revenue_series.astype(float),
        order=order,
    )

    fitted_model = model.fit()

    return fitted_model


def predict_arima_next_quarter(
        model,
) -> float:
    """
    Forecast one quarter ahead using a fitted ARIMA model.
    """

    forecast = model.forecast(
        steps=1
    )

    return float(
        forecast.iloc[0]
    )