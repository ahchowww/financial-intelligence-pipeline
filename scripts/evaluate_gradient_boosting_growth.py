import pandas as pd

from src.processing.features import (
    build_financial_training_dataset,
)

from src.modeling.split import (
    chronological_train_val_test_split,
)

from src.modeling.gradient_boosting import (
    train_gradient_boosting_growth_model,
    predict_gradient_boosting_growth,
)

from src.modeling.metrics import (
    calculate_forecast_metrics,
)


DATA_PATH = "data/processed/tsla_quarterly_modeling.csv"


def main() -> None:

    # 1. Load dataset
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
    
    # 4. Train growth model
    model = train_gradient_boosting_growth_model(train)
    
    # 5. Predict next-quarter growth
    predicted_growth = predict_gradient_boosting_growth(
        model,
        validation,
    )
    
    # 6. Convert growth forecast back to revenue
    # predicted revenue = current revenue * (1 + predicted growth)
    predicted_revenue = (validation["revenue"] * (1 + predicted_growth))    

    # 7. Evaluate dollar revenue forecast
    metric = calculate_forecast_metrics(
        actual=validation["target_revenue"],
        predicted=predicted_revenue,
    )

    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING GROWTH - VALIDATION")
    print("=" * 60)

    print(f"Train rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")

    print(
        f"MAE: ${metric['MAE']:,.2f}",
        f"\nRMSE: ${metric['RMSE']:,.2f}",
        f"\nMAPE: {metric['MAPE']:.2f}%",
        f"\nSAPE: {metric['SMAPE']:.2f}%",
    )
    
    # 8. Individual forecasts
    results = pd.DataFrame(
        {
            "feature_quarter": (validation["year"].astype(str) + validation["quarter"]),
            "actual_growth": validation["target_revenue_growth"],
            "predicted_growth": predicted_growth,
            "actual_revenue": validation["target_revenue"],
            "predicted_revenue": predicted_revenue,
        }
    )

    print("\nForecasts:")

    print(
        results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()