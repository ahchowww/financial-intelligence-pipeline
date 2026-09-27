import pandas as pd

from src.processing.features import (
    build_revenue_training_dataset,
)

from src.modeling.split import (
    chronological_train_val_test_split,
)

from src.modeling.ridge import (
    predict_ridge_revenue,
    select_ridge_alpha,
    train_ridge_revenue_model,
)

from src.modeling.metrics import (
    calculate_forecast_metrics,
)


DATA_PATH = "data/processed/tsla_quarterly_modeling.csv"


def main() -> None:
    # 1. Load modeling dataset
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "quarter_end",
            "available_date",
        ],
    )

    # 2. Build supervised forecasting dataset
    training = build_revenue_training_dataset(df
    )

    # 3. Chronological split
    train, validation, test = chronological_train_val_test_split(training)


    # 4. Choose Ridge alpha using training data only
    best_alpha, cv_results = select_ridge_alpha(train)

    print("\n" + "=" * 60)
    print("RIDGE REGRESSION - ALPHA SELECTION")
    print("=" * 60)

    print(
        f"Training rows: {len(train)}"
    )

    print("\nTime-series cross-validation results:")

    display_results = (
        cv_results.copy()
    )

    display_results["mean_cv_mae"] = (
        display_results["mean_cv_mae"]
        .map(
            lambda value: f"${value:,.2f}"
        )
    )

    print(
        display_results.to_string(index=False)
    )

    print(
        f"\nSelected alpha: {best_alpha}"
    )


    # 5. Train final Ridge model on all training rows
    model = train_ridge_revenue_model(
        train=train,
        alpha=best_alpha,
    )

    # 6. Evaluate once on validation set
    predictions = predict_ridge_revenue(
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
    print("RIDGE REGRESSION - VALIDATION")

    print("=" * 60)

    print(
        f"Validation rows: {len(validation)}"
    )

    print()

    print(
        f"MAE:   ${metrics['MAE']:,.2f}",
        f"\nRMSE:  ${metrics['RMSE']:,.2f}",
        f"\nMAPE:  {metrics['MAPE']:.2f}%",
        f"\nSMAPE: {metrics['SMAPE']:.2f}%"
    )

    # 7. Individual validation forecasts
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