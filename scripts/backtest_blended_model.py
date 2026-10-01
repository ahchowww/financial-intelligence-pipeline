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

MIN_TRAIN_SIZE = 20

BLEND_WEIGHTS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]


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

    actual_values = []
    naive_predictions = []
    gb_predictions = []

    # Walk-forward forecasts inside training only
    for index in range(MIN_TRAIN_SIZE, len(train)):
        history = train.iloc[:index].copy()

        current = train.iloc[[index]].copy()

        actual = float(
            current["target_revenue"].iloc[0]
        )

        # Naive
        naive_prediction = float(
            naive_revenue_forecast(
                current
            ).iloc[0]
        )

        # GB Log Growth
        gb_model = train_gradient_boosting_log_growth_model(
            history
        )

        predicted_log_growth = float(
            predict_gradient_boosting_log_growth(
                gb_model,
                current,
            ).iloc[0]
        )

        gb_prediction = float(
            current["revenue"].iloc[0] * np.exp(predicted_log_growth)
        )


        actual_values.append(actual)
        naive_predictions.append(naive_prediction)
        gb_predictions.append(gb_prediction)


    # Try blended weights
    # weight = 0 -> pure Naive
    # weight = 1 -> pure GB
    results = []

    naive_array = np.asarray(
        naive_predictions
    )

    gb_array = np.asarray(
        gb_predictions
    )

    for weight in BLEND_WEIGHTS:
        blended = (
            (1 - weight) * naive_array + weight * gb_array
        )

        metrics = calculate_forecast_metrics(
            actual=actual_values,
            predicted=blended,
        )

        results.append(
            {
                "gb_weight": weight,
                "naive_weight": 1 - weight,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
                "SMAPE": metrics["SMAPE"],
            }
        )

    results_df = (
        pd.DataFrame(
            results
        )
        .sort_values(
            "MAE"
        )
        .reset_index(
            drop=True
        )
    )


    print("\n" + "=" * 60)
    print("NAIVE + GB LOG GROWTH BLEND")
    print("=" * 60)

    display = results_df.copy()

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

    print(display.to_string(index=False))

    best_weight = float(results_df.iloc[0]["gb_weight"])

    print("\nSelected GB weight:", best_weight)
    print("\nSelected Naive weight:", 1 - best_weight)


if __name__ == "__main__":
    main()