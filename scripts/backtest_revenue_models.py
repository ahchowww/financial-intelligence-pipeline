import pandas as pd

from src.processing.features import(
    build_financial_training_dataset,
    COMPACT_FINANCIAL_FEATURE_COLUMNS,
)

from src.modeling.split import(
    chronological_train_val_test_split,
)

from src.modeling.baselines import(
    naive_revenue_forecast,
    seasonal_naive_revenue_forecast,
    moving_average_revenue_forecast,
)

from src.modeling.linear import(
    train_linear_revenue_model,
    predict_linear_revenue,
)

from src.modeling.ridge import(
    select_ridge_alpha,
    train_ridge_revenue_model,
    predict_ridge_revenue,
)

from src.modeling.metrics import(
    calculate_forecast_metrics,
)

DATA_PATH = "data/processed/tsla_quarterly_modeling.csv"

# 20 training observation (around 5 years of historical training examples)
MIN_TRAIN_SIZE = 20


def main() -> None:
    """
    Walk-forward backtest
    -> Simulate how each model would have performed historically 
       if you had repeatedly trained only on info available up to that point, 
       forecast the next quarter, and then compared the forecast with actual values.
    """
    
    # 1. Load dataset
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "quarter_end",
            "available_date",
        ],
    )

    training = build_financial_training_dataset(df)

    train, validation, test = chronological_train_val_test_split(training)


    ## Backtest only inside the original training set.
    ## Validation and test are not used here.
    print("\n" + "=" * 60)
    print("WALK-FORWARD BACKTEST")
    print("=" * 60)

    print(f"Training rows available: {len(train)}")
    print(f"Initial training window: {MIN_TRAIN_SIZE}")

    print(
        f"Forecast origins: {len(train) - MIN_TRAIN_SIZE}"
    )

    # 2. Store forecasts
    actual_values = []
    naive_predictions = []
    seasonal_predictions = []
    moving_average_predictions = []
    linear_predictions = []
    ridge_predictions = []

    forecast_rows = []
    multivariate_ridge_predictions = []

    # 3. Expanding-window backtest
    for index in range(MIN_TRAIN_SIZE, len(train)):
        # history - used to train Linear/Ridge
        # current - used to generate the next-quarter forecast

        # Everything before this row is historical
        history = train.iloc[:index].copy()

        # This is the current forecast origin
        current = train.iloc[
            [index]
        ].copy()

        actual = float(current["target_revenue"].iloc[0])


        naive_prediction = float(naive_revenue_forecast(current).iloc[0])
        seasonal_prediction = float(seasonal_naive_revenue_forecast(current).iloc[0])
        moving_average_prediction = float(moving_average_revenue_forecast(current).iloc[0])

        # Linear regression
        linear_model = train_linear_revenue_model(history)
        linear_prediction = float(
            predict_linear_revenue(
                linear_model,
                current,
            ).iloc[0]
        )

        # Ridge
        # Alpha selection uses only history available at this forecast origin
        best_alpha, _ = (
            select_ridge_alpha(
                history,
                n_splits=4,
            )
        )

        ridge_model = (
            train_ridge_revenue_model(
                train=history,
                alpha=best_alpha,
            )
        )

        ridge_prediction = float(
            predict_ridge_revenue(
                ridge_model,
                current,
            ).iloc[0]
        )

        # Multivariate Ridge
        multivariate_best_alpha, _ = select_ridge_alpha(
            history,
            feature_columns=COMPACT_FINANCIAL_FEATURE_COLUMNS,
            n_splits=4,
        )

        multivariate_ridge_model = train_ridge_revenue_model(
            train=history,
            alpha=multivariate_best_alpha,
            feature_columns=COMPACT_FINANCIAL_FEATURE_COLUMNS,
        )

        multivariate_ridge_prediction = float(
            predict_ridge_revenue(
                model=multivariate_ridge_model,
                df=current,
                feature_columns=COMPACT_FINANCIAL_FEATURE_COLUMNS,
            ).iloc[0]
        )


        # Save
        actual_values.append(actual)
        naive_predictions.append(naive_prediction)
        seasonal_predictions.append(seasonal_prediction)
        moving_average_predictions.append(moving_average_prediction)
        linear_predictions.append(linear_prediction)
        ridge_predictions.append(ridge_prediction)
        multivariate_ridge_predictions.append(multivariate_ridge_prediction)

        forecast_rows.append(
            {
                "feature_quarter": (
                    str(current["year"].iloc[0]) + current["quarter"].iloc[0]
                ),
                "target_quarter_end": (
                    current["target_quarter_end"].iloc[0]
                ),
                "actual_revenue": actual,
                "naive": naive_prediction,
                "seasonal_naive": seasonal_prediction,
                "moving_average": moving_average_prediction,
                "linear": linear_prediction,
                "ridge": ridge_prediction,
                "ridge_alpha": best_alpha,
                "multivariate_ridge": multivariate_ridge_prediction,
                "multivariate_best_alpha": multivariate_best_alpha,
            }
        )

    # 4. Evaluate each model
    prediction_sets = {
        "Naive": naive_predictions,
        "Seasonal Naive": seasonal_predictions,
        "Moving Average": moving_average_predictions,
        "Linear Regression": linear_predictions,
        "Ridge Regression": ridge_predictions,
        "Multivariate Ridge": multivariate_ridge_predictions,
    }

    summary_rows = []

    for (model_name, predictions) in prediction_sets.items():
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
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    # 5. Print summary
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    display = summary.copy()

    display["MAE"] = (
        display["MAE"]
        .map(lambda value: f"${value:,.2f}")
    )

    display["RMSE"] = (
        display["RMSE"]
        .map(lambda value: f"${value:,.2f}")
    )

    display["MAPE"] = (
        display["MAPE"]
        .map(lambda value: f"{value:.2f}%")
    )

    display["SMAPE"] = (
        display["SMAPE"]
        .map(lambda value: f"{value:.2f}%")
    )

    print(
        display.to_string(index=False)
    )

    # 6. Print individual forecast
    forecast_df = pd.DataFrame(forecast_rows)

    print("\n" + "=" * 60)
    print("INDIVIDUAL FORECAST")
    print("=" * 60)

    print(
        forecast_df.to_string(index=False)
    )


if __name__ == "__main__":
    main()
