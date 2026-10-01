import numpy as np
import pandas as pd

from src.processing.features import(
    build_financial_training_dataset,
)

from src.modeling.split import(
    chronological_train_val_test_split,
)

from src.modeling.baselines import(
    naive_revenue_forecast,
)

from src.modeling.gradient_boosting import(
    train_gradient_boosting_log_growth_model,
    predict_gradient_boosting_log_growth,
)

from src.modeling.metrics import(
    calculate_forecast_metrics,
)


DATA_PATH = "data/processed/tsla_quarterly_modeling.csv"


GB_WEIGHT = 0.25
NAIVE_WEIGHT = 0.75


def main() -> None:
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "quarter_end",
            "available_date",
        ],
    )

    training = build_financial_training_dataset(df)

    train, validation, test = chronological_train_val_test_split(training)

    gb_model = train_gradient_boosting_log_growth_model(train)

    # Naive validation predictions
    naive_predictions = naive_revenue_forecast(
        validation
    )

    # 4. GB Log Growth validation predictions
    predicted_log_growth = predict_gradient_boosting_log_growth(
        gb_model,
        validation,
    )

    gb_predictions = (
        validation["revenue"] * np.exp(predicted_log_growth)
    )


    # Blend
    # Weight was selected using TRAINING backtest only.
    # Do not tune it using validation.
    blended_predictions = (
        NAIVE_WEIGHT * naive_predictions + GB_WEIGHT * gb_predictions
    )


    # Evaluate
    metrics = calculate_forecast_metrics(
        actual=validation["target_revenue"],
        predicted=blended_predictions,
    )


    print("\n" + "=" * 60)
    print("BLENDED MODEL - VALIDATION")
    print("=" * 60)

    print(f"Naive weight: {NAIVE_WEIGHT:.2f}")
    print(f"GB weight: {GB_WEIGHT:.2f}")

    print(
        f"MAE: ${metrics['MAE']:,.2f}",
        f"\nRMSE: ${metrics['RMSE']:,.2f}",
        f"\nMAPE: {metrics['MAPE']:.2f}%",
        f"\nSMAPE: {metrics['SMAPE']:.2f}%",
    )


    # Individual forecasts
    results = pd.DataFrame(
        {
            "feature_quarter": validation["year"].astype(str) + validation["quarter"],
            "actual_revenue": validation["target_revenue"],
            "naive_prediction": naive_predictions,
            "gb_prediction": gb_predictions,
            "blended_prediction": blended_predictions,
        }
    )

    print("\nForecasts:")

    print(
        results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()