import pandas as pd

from src.processing.features import(
    build_financial_training_dataset
)

from src.modeling.split import(
    chronological_train_val_test_split,
)

from src.modeling.statistical import(
    train_arima_model,
    predict_arima_next_quarter,
)

from src.modeling.metrics import(
    calculate_forecast_metrics,
)

DATA_PATH = "data/processed/tsla_quarterly_modeling.csv"

MIN_TRAIN_SIZE = 20

ARIMA_ORDERS = [
    (0, 1, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 0),
    (0, 1, 2),
    (2, 1, 1),
    (1, 1, 2),
]


def evaluate_arima_order(
        train: pd.DataFrame,
        order: tuple[int, int, int],
) -> dict:

    actual_values = []
    predictions = []

    # Walk-forward inside original training set
    for index in range(MIN_TRAIN_SIZE, len(train)):
        current = train.iloc[[index]].copy()

        actual = float(
            current["target_revenue"].iloc[0]
        )

        # Revenue available up to the current quarter
        revenue_history = (
            train.iloc[:index+1]["revenue"].copy()
        )

        model = train_arima_model(
            revenue_series=revenue_history,
            order=order,
        )

        prediction = predict_arima_next_quarter(
            model
        )

        actual_values.append(actual)
        predictions.append(prediction)

    metrics = calculate_forecast_metrics(
        actual=actual_values,
        predicted=predictions,
    )

    return {
        "order": str(order),
        "p": order[0],
        "d": order[1],
        "q": order[2],
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"],
        "SMAPE": metrics["SMAPE"],
    }


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
    print("ARIMA ORDER SELECTION")
    print("=" * 60)

    print(f"Training rows available: {len(train)}")
    print(f"Initial training rows: {MIN_TRAIN_SIZE}")

    print(
        f"Forecast origins per order: "
        f"{len(train) - MIN_TRAIN_SIZE}"
    )

    print(
        f"Candidate orders: "
        f"{len(ARIMA_ORDERS)}"
    )

    # Evaluate candidate orders
    results = []

    for order in ARIMA_ORDERS:
        print(f"Evaluating ARIMA{order}...")

        try:
            result = evaluate_arima_order(
                train=train,
                order=order,
            )

            results.append(result)

        except Exception as error:
            print(f"ARIMA{order} failed: ")
            print(error)

    # Rank candidates by MAE
    if not results:
        raise RuntimeError(
            "All ARIMA configurations failed."
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

    results_df.insert(
        0,
        "rank",
        range(1, len(results_df) + 1,),
    )

    # Display results
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

    print("\n" + "=" * 60)
    print("ARIMA ORDER RANKING")
    print("=" * 60)

    print(
        display[
            [
                "rank",
                "order",
                "MAE",
                "RMSE",
                "MAPE",
                "SMAPE",
            ]
        ].to_string(index=False)
    )

    # Best order
    best = results_df.iloc[0]

    best_order = (
        int(best["p"]),
        int(best["d"]),
        int(best["q"]),
    )

    print("\n" + "=" * 60)
    print("SELECTED ARIMA ORDER")
    print("=" * 60)

    print(f"Best order: ARIMA{best_order}")

    print(f"Backtest MAE: ${best['MAE']:,.2f}")
    print(f"Backtest RMSE: ${best['RMSE']:,.2f}")
    print(f"Backtest MAPE: {best['MAPE']:.2f}%")
    print(f"Backtest SMAPE: {best['SMAPE']:.2f}%")


if __name__ == "__main__":
    main()