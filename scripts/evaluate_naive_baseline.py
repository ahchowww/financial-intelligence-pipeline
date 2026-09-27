import pandas as pd

from src.processing.features import(
    build_revenue_training_dataset,
)

from src.modeling.split import(
    chronological_train_val_test_split,
)

from src.modeling.baselines import(
    naive_revenue_forecast,
)

from src.modeling.metrics import(
    calculate_forecast_metrics,
)

DATA_PATH = "data/processed/tsla_quarterly_modeling.csv"


def evaluate_split(
        df: pd.DataFrame,
        split_name: str,
) -> None:
    """
    Evaluate naive forecast on one dataset split.
    """

    predictions = naive_revenue_forecast(df)

    metrics = calculate_forecast_metrics(
        actual=df["target_revenue"],
        predicted=predictions,
    )

    print("\n" + "=" * 60)
    print(split_name.upper())
    print("=" * 60)

    print(f"Rows: {len(df)}")
    print(f"MAE: ${metrics['MAE']:,.2f}")
    print(f"RMSE: ${metrics['RMSE']:,.2f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"SMAPE: {metrics['SMAPE']:.2f}%")


    # Show individual forecasts
    results = pd.DataFrame(
        {
            "feature_quarter": df["year"].astype(str) + df["quarter"],
            "target_quarter_end": df["target_quarter_end"],
            "actual_revenue": df["target_revenue"],
            "predicted_revenue": predictions,
        }
    )

    print("\nForecasts: ")
    print(
        results.to_string(index=False)
    )


def main() -> None:
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "quarter_end",
            "available_date",
        ],
    )

    training = build_revenue_training_dataset(df)

    train, validation, test = chronological_train_val_test_split(training)


    # primarily inspect validation first
    evaluate_split(
        validation, 
        "Validation",
    )

    evaluate_split(
        test,
        "Test",
    )


if __name__ == "__main__":
    main()