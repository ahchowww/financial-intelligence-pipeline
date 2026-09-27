import numpy as np
import pandas as pd

def calculate_forecast_metrics(
        actual: pd.DataFrame,
        predicted: pd.Series,
) -> dict[str, float]:
    """
    Calculate standard forecasting error metrics.

    Metrics:
        MAE - Mean Absolute Error
        RSME - Root Mean Squared Error
        MAPE - Mean Absolute Percentage Error
        SMAPE - Symmetric Mean Absolute Percentage Error
    """

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted must have the same length."
        )

    if len(actual) == 0:
        raise ValueError(
            "Cannot evaluate empty arrays."
        )

    errors = actual - predicted

    # 1. MAE
    mae = np.mean(
        np.abs(errors)
    )

    # 2. RSME
    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    # 3. MAPE
    # Exclude actual values equal to zero because
    # percentage error is undefined there.
    nonzero_actual = (actual != 0)

    if nonzero_actual.any():
        mape = np.mean(
            np.abs(
                errors[nonzero_actual] / actual[nonzero_actual]
            ) * 100
        )
    else:
        mape = np.nan    

    # 4. SMAPE
    denominator = (
        np.abs(actual) + np.abs(predicted)
    )

    valid_smape = (denominator != 0)

    if valid_smape.any():
        smape = np.mean(
            2 * np.abs(errors[valid_smape]) / denominator[valid_smape]
        ) * 100
    else:
        smape = np.nan

    
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "SMAPE": float(smape),
    }