import pandas as pd

from src.processing.features import(
    build_financial_training_dataset,
    COMPACT_FINANCIAL_FEATURE_COLUMNS,
)

from src.modeling.split import(
    chronological_train_val_test_split,
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


def main() -> None:
    # 1. Load data
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "quarter_end",
            "available_date",
        ],
    )

    # 2. Build multivariate training dataset
    training = build_financial_training_dataset(df)

    # 3. Chronological split
    train, validation, test = chronological_train_val_test_split(training)

    # 4. Select alpha using training data only
    best_alpha, cv_results = select_ridge_alpha(
        train=train,
        feature_columns=COMPACT_FINANCIAL_FEATURE_COLUMNS,
    )

    print("\n" + "=" * 60)
    print("MULTIVARIATE RIDGE -ALPHA SELECTION")
    print("=" * 60)

    print(f"Training rows: {len(train)}")
    print(f"Features: {len(COMPACT_FINANCIAL_FEATURE_COLUMNS)}")

    print("\nFeature columns: ")
    for feature in COMPACT_FINANCIAL_FEATURE_COLUMNS:
        print(f"  - {feature}")

    display_cv = cv_results.copy()

    display_cv["mean_cv_mae"] = (
        display_cv["mean_cv_mae"]
        .map(
            lambda value: f"${value:,.2f}"
        )
    )

    print("\nTime-series CV: ")
    print(display_cv.to_string(index=False))

    print(f"Selected alpha: {best_alpha}")

    # 5. Fit final model on training period
    model = train_ridge_revenue_model(
        train=train,
        alpha=best_alpha,
        feature_columns=COMPACT_FINANCIAL_FEATURE_COLUMNS,
    )

    # 6. Validation predictions
    predictions = predict_ridge_revenue(
        model=model,
        df=validation,
        feature_columns=COMPACT_FINANCIAL_FEATURE_COLUMNS,
    )

    metrics = calculate_forecast_metrics(
        actual=validation["target_revenue"],
        predicted=predictions,
    )


    # 7. Result
    print("\n" + "=" * 60)
    print("MULTIVARIATE RIDGE - VALIDATION")
    print("=" * 60)

    print(
        f"MAE: {metrics['MAE']:,.2f}",
        f"\nRMSE: {metrics['RMSE']:,.2f}",
        f"\nMAPE: {metrics['MAPE']:.2f}%",
        f"\nSMAPE: {metrics['SMAPE']:.2f}%"
    )

    results = pd.DataFrame(
        {
            "feature_quarter": validation["year"].astype(str) + validation["quarter"],
            "target_quarter_end": validation["target_quarter_end"],
            "actual_revenue": validation["target_revenue"],
            "predicted_revenue": predictions,
        }
    )

    print("\nForecasts: ")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()