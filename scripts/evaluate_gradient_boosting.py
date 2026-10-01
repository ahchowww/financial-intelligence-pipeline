import pandas as pd

from src.processing.features import(
    build_financial_training_dataset,
)

from src.modeling.split import(
    chronological_train_val_test_split,
)

from src.modeling.gradient_boosting import(
    train_gradient_boosting_revenue_model,
    predict_gradient_boosting_revenue,
)

from src.modeling.metrics import(
    calculate_forecast_metrics,
)

DATA_PATH = "data/processed/tsla_quarterly_modeling.csv"


def main() -> None:
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "quarter_end",
            "available_date",
        ]
    )

    training = build_financial_training_dataset(df)

    train, validation, test = chronological_train_val_test_split(training)

    model = train_gradient_boosting_revenue_model(train)

    predictions = predict_gradient_boosting_revenue(
        model,
        validation,
    )

    metrics = calculate_forecast_metrics(
        actual=validation["target_revenue"],
        predicted=predictions,
    )

    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING - VALIDATION")
    print("=" * 60)

    print(f"Train rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")

    print(
        f"MAE: ${metrics['MAE']:,.2f}"
        f"\nRMSE: ${metrics['RMSE']:,.2f}"
        f"\nMAPE: {metrics['MAPE']:.2f}%"
        f"\nSMAPE: {metrics['SMAPE']:.2f}%"
    )

    results = pd.DataFrame(
        {
            "feature_quarter": (validation["year"].astype(str) + validation["quarter"]),
            "target_quarter_end": validation["target_quarter_end"],
            "actual_revenue": validation["target_revenue"],
            "predicted_revenue": predictions,
        }
    )


    print("\nForecasts: ")

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()