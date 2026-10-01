import numpy as np
import pandas as pd

from src.processing.features import(
    build_financial_training_dataset,
)

from src.modeling.split import(
    chronological_train_val_test_split,
)

from src.modeling.gradient_boosting import(
    train_gradient_boosting_log_growth_model,
    predict_gradient_boosting_log_growth,
)

from src.modeling.metrics import(
    calculate_forecast_metrics,
)

DATA_PATH = "data/processed/tsla_quarterly_modeling.csv"


def main() -> None:
    # 1. Load dataset
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "quarter_end",
            "available_date",
        ]
    )

    # 2. Build model-ready dataset
    training = build_financial_training_dataset(df)

    # 3. Chronological split
    train, validation, test = chronological_train_val_test_split(training)

    # 4.Train Log-growth model
    model = train_gradient_boosting_log_growth_model(train)

    # 5. Predict Log growth
    predicted_log_growth = predict_gradient_boosting_log_growth(
        model, 
        validation,
    )

    # 6. Convert back to revenue
    # Log growth = log(next_revenue / current_revenue)
    # next revenue = current_revenue * exp(Log_growth)
    predicted_revenue = (
        validation["revenue"] * np.exp(predicted_log_growth)
    )

    # 7. Evaluate dollar forecast
    metrics = calculate_forecast_metrics(
        actual=validation["target_revenue"],
        predicted=predicted_revenue,
    )


    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING LOG GROWTH - VALIDATION")
    print("=" * 60)

    print(f"Train rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")

    print(
        f"MAE: ${metrics['MAE']:,.2f}",
        f"\nRMSE: ${metrics['RMSE']:,.2f}",
        f"\nMAPE: {metrics['MAPE']:.2f}%",
        f"\nSMAPE: {metrics['SMAPE']:.2f}%",
    )

    # 8. Individual forecasts
    results = pd.DataFrame(
        {
            "feature_quarter": (validation["year"].astype(str) + validation["quarter"]),
            "actual_log_growth": validation["target_log_revenue_growth"],
            "prediction_log_growth": predicted_log_growth,
            "actual_revenue": validation["target_revenue"],
            "predicted_revenue": predicted_revenue,
        }
    )


    print("\nForecasts: ")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

