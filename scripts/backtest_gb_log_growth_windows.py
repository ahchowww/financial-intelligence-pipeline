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
    predict_gradient_boosting_log_growth
)

from src.modeling.metrics import(
    calculate_forecast_metrics,
)

DATA_PATH = "data/processed/tsla_quarterly_modeling.csv"

INITIAL_TRAIN_SIZE = 20
ROLLING_WINDOW_SIZE = 20


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


    print("\n" + "=" * 60)
    print("GB LOG GROWTH - EXPANDING VS ROLLING")
    print("=" * 60)

    print(f"Training rows available: {len(train)}")
    print(f"Initial training size: {INITIAL_TRAIN_SIZE}")
    print(f"Rolling window size: {ROLLING_WINDOW_SIZE}")

    print(
        f"Forecast origins: "
        f"{len(train) - INITIAL_TRAIN_SIZE}"
    )


    # Prediction storage
    actual_values = []
    naive_predictions = []
    expanding_predictions = []
    rolling_predictions = []
    forecast_rows = []

    # Walk-forward forecasting
    for index in range(INITIAL_TRAIN_SIZE, len(train)):
        current = train.iloc[[index]].copy()

        actual = float(
            current["target_revenue"].iloc[0]
        )

        current_revenue = float(
            current["revenue"].iloc[0]
        )

        # Naive
        naive_prediction = float(
            naive_revenue_forecast(current).iloc[0]
        )


        # Expanding-window Gradient Boosting
        expanding_history = train.iloc[:index].copy()

        expanding_model = train_gradient_boosting_log_growth_model(expanding_history)

        expanding_log_growth = float(
            predict_gradient_boosting_log_growth(
                expanding_model,
                current,
            ).iloc[0]
        )

        expanding_prediction = (
            current_revenue * np.exp(expanding_log_growth)
        )


        # Rolling-window Gradient Boosting
        rolling_start = max(
            0,
            index - ROLLING_WINDOW_SIZE,
        )

        rolling_history = train.iloc[rolling_start: index].copy()

        rolling_model = train_gradient_boosting_log_growth_model(
            rolling_history
        )

        rolling_log_growth = float(
            predict_gradient_boosting_log_growth(
                rolling_model,
                current,
            ).iloc[0]
        )

        rolling_prediction = (
            current_revenue * np.exp(rolling_log_growth)
        )


        # Save
        actual_values.append(actual)
        naive_predictions.append(naive_prediction)
        expanding_predictions.append(expanding_prediction)
        rolling_predictions.append(rolling_prediction)

        forecast_rows.append(
            {
                "feature_quarter": (
                    str(current["year"].iloc[0]) + current["quarter"].iloc[0]
                ),
                "target_quarter_end": current["target_quarter_end"].iloc[0],
                "actual_revenue": actual,
                "naive": naive_prediction,
                "expanding_gb_log": expanding_prediction,
                "expanding_log_growth": expanding_log_growth,
                "rolling_gb_log": rolling_prediction,
                "rolling_log_growth": rolling_log_growth,
                "rolling_train_rows": len(rolling_history),
                "rolling_start_quarter": (
                    str(rolling_history["year"].iloc[0]) + rolling_history["quarter"].iloc[0]
                ),
                "rolling_end_quarter": (
                    str(rolling_history["year"].iloc[-1]) + rolling_history["quarter"].iloc[-1]
                ),
            }
        )

    # Evaluate models
    prediction_sets = {
        "Naive": naive_predictions,
        "GB Log Growth - Expanding": expanding_predictions,
        "GB Log Growth - Rolling 20Q": rolling_predictions,
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
        ).sort_values(
            "MAE"
        )
        .reset_index(
            drop=True
        )
    )


    # Print summary
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

    print(display.to_string(
        index=False
    ))


    # Individual forecast
    forecast_df = pd.DataFrame(forecast_rows)

    print("\n" + "=" * 60)
    print("INDIVIDUAL FORECASTS")
    print("=" * 60)

    print(forecast_df.to_string(
        index=False,
    ))


if __name__ == "__main__":
    main()