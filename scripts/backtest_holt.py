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

from src.modeling.statistical import(
    train_holt_model,
    predict_holt_next_quarter,
)

from src.modeling.metrics import(
    calculate_forecast_metrics,
)

DATA_PATH = "data/processed/tsla_quarterly_modeling.csv"

MIN_TRAIN_SIZE = 20


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

    # only use original training data for backtesting
    print("\n" + "=" * 60)
    print("HOLT WALK-FORWARD BACKTEST")
    print("=" * 60)

    print(f"Training rows available: {len(train)}")
    print(f"Initial training rows: {MIN_TRAIN_SIZE}")

    print(
        f"Forecast origins: "
        f"{len(train) - MIN_TRAIN_SIZE}"
    )

    actual_values = []
    naive_predictions = []
    holt_predictions = []

    forecast_rows = []


    # Walk-forward
    for index in range(MIN_TRAIN_SIZE, len(train)):
        current = train.iloc[[index]].copy()

        actual = float(
            current["target_revenue"].iloc[0]
        )


        # Naive forecast
        naive_prediction = float(
            naive_revenue_forecast(
                current
            ).iloc[0]
        )

        # Holt forecast
        # At this forecast origin, current-quarter revenue is already known.
        # So, Holt is trained using all revenue observations up to and 
        # including current quarter.
        revenue_history = train.iloc[:index+1]["revenue"].copy()

        holt_model = train_holt_model(
            revenue_history
        )

        holt_prediction = predict_holt_next_quarter(
            holt_model
        )


        # Save result
        actual_values.append(actual)
        naive_predictions.append(naive_prediction)
        holt_predictions.append(holt_prediction)

        forecast_rows.append(
            {
                "feature_quarter": (
                    str(current["year"].iloc[0]) + current["quarter"].iloc[0]
                ),
                "target_quarter_end": current["target_quarter_end"].iloc[0],
                "actual_revenue": actual,
                "naive_prediction": naive_prediction,
                "holt_prediction": holt_prediction,
                "history_rows": len(revenue_history),
            }
        )

    # Evaluate
    prediction_sets = {
        "Naive": naive_predictions,
        "Holt Linear Trend": holt_predictions,
    }

    summary_rows = []

    for model_name, predictions in prediction_sets.items():
        metrics = calculate_forecast_metrics(
            actual=actual_values,
            predicted=predictions,
        )

        summary_rows.append(
            {
                "model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
                "SMAPE": metrics["SMAPE"],
            }
        )

    summary = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            "MAE"
        )
        .reset_index(
            drop=True
        )
    )

    display = summary.copy()

    display["MAE"] = display["MAE"].map(
        lambda value: f"${value:,.2f}"
    )

    display["RMSE"] = display["RMSE"].map(
        lambda value: f"${value:,.2f}"
    )

    display["MAPE"] = display["MAPE"].map(
        lambda value: f"{value:.2f}%"
    )

    display["SMAPE"] = display["SMAPE"].map(
        lambda value: f"{value:.2f}%"
    )

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(
        display.to_string(index=False)
    )

    # individual forecasts
    print("\n" + "=" * 60)
    print("INDIVIDUAL FORECASTS")
    print("=" * 60)

    print(
        pd.DataFrame(
            forecast_rows
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
