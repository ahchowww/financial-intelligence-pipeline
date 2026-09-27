import pandas as pd

from src.processing.features import (
    build_revenue_training_dataset,
)

from src.modeling.split import (
    chronological_train_val_test_split,
)

from src.modeling.linear import (
    DEFAULT_FEATURE_COLUMNS,
    predict_linear_revenue,
    train_linear_revenue_model,
)

from src.modeling.metrics import (
    calculate_forecast_metrics,
)

DATA_PATH = "data/processed/tsla_quarterly_modeling.csv"


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


    # Train only on training period
    model = (
        train_linear_revenue_model(train)
    )

    # Evaluate on validation period
    predictions = predict_linear_revenue(
            model,
            validation,
    )

    metrics = calculate_forecast_metrics(
        actual=validation[
            "target_revenue"
        ],
        predicted=predictions,
    )

    print("\n" + "=" * 60)
    print("LINEAR REGRESSION - VALIDATION")
    print("=" * 60)

    print(
        f"Train rows: {len(train)}"
    )

    print(
        f"Validation rows: {len(validation)}"
    )

    print()

    print(
        f"MAE:   ${metrics['MAE']:,.2f}",
        f"\nRMSE:  ${metrics['RMSE']:,.2f}",
        f"\nMAPE:  {metrics['MAPE']:.2f}%",
        f"\nSMAPE:  {metrics['SMAPE']:.2f}%",
    )


    # Model coefficients
    print("\nCoefficients:")

    for feature, coefficient in zip(
        DEFAULT_FEATURE_COLUMNS,
        model.coef_,
    ):

        print(
            f"{feature:15s}: "
            f"{coefficient:.4f}"
        )

    print(
        f"intercept: "
        f"${model.intercept_:,.2f}"
    )


    # Individual validation forecasts
    results = pd.DataFrame(
        {
            "feature_quarter": (
                validation["year"]
                .astype(str)
                + validation["quarter"]
            ),
            "target_quarter_end": (
                validation[
                    "target_quarter_end"
                ]
            ),
            "actual_revenue": (
                validation[
                    "target_revenue"
                ]
            ),
            "predicted_revenue": (
                predictions
            ),
        }
    )

    print("\nForecasts:")

    print(
        results.to_string(index=False)
    )


if __name__ == "__main__":
    main()