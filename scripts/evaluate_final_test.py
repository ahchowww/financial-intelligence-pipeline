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

NAIVE_WEIGHT = 0.75
GB_WEIGHT = 0.25


def main() -> None:
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "quarter_end",
            "available_date",
        ],
    )

    training = build_financial_training_dataset(df)

    train, validation, test = chronological_train_val_test_split(training)


    # Initial history = train + validation
    # Model selection is already finished.
    # Test data has not been used for tuning.
    history = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    actual_values = []
    naive_predictions = []
    gb_predictions = []
    blended_predictions = []

    rows = []


    # Final walk-forward test
    for index in range(len(test)):
        current = (
            test.iloc[
                [index]
            ]
            .copy()
        )

        actual = float(
            current[
                "target_revenue"
            ].iloc[0]
        )

        current_revenue = float(
            current[
                "revenue"
            ].iloc[0]
        )
        
        # Naive        
        naive_prediction = float(
            naive_revenue_forecast(
                current
            ).iloc[0]
        )

        
        # GB Log Growth
        gb_model = train_gradient_boosting_log_growth_model(history)

        predicted_log_growth = float(
            predict_gradient_boosting_log_growth(
                gb_model,
                current,
            ).iloc[0]
        )

        gb_prediction = (
            current_revenue* np.exp(predicted_log_growth)
        )

        
        # Frozen blend
        blended_prediction = (
            NAIVE_WEIGHT * naive_prediction + GB_WEIGHT * gb_prediction
        )

        actual_values.append(actual)
        naive_predictions.append(naive_prediction)
        gb_predictions.append(gb_prediction)
        blended_predictions.append(blended_prediction)

        rows.append(
            {
                "feature_quarter": (
                    str(current["year"].iloc[0]) + current["quarter"].iloc[0]
                ),
                "target_quarter_end": current["target_quarter_end"].iloc[0],

                "actual_revenue": actual,
                "naive_prediction": naive_prediction,
                "gb_log_growth_prediction": gb_prediction,
                "blended_prediction": blended_prediction,
                "predicted_log_growth": predicted_log_growth, 
            }
        )

    
        # After this outcome becomes historical,
        # make the row available for the next forecast.
        history = pd.concat(
            [
                history,
                current,
            ],
            ignore_index=True,
        )

    # Final test metrics
    prediction_sets = {
        "Naive": naive_predictions,
        "GB Log Growth": gb_predictions,
        "75% Naive + 25% GB": blended_predictions,
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

    summary = pd.DataFrame(
        summary_rows
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
    print("FINAL UNTOUCHED TEST EVALUATION")
    print("=" * 60)

    print(
        f"Initial history rows: "
        f"{len(train) + len(validation)}"
    )

    print(
        f"Test forecast origins: "
        f"{len(test)}"
    )

    print(
        display.to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)
    print("INDIVIDUAL TEST FORECASTS")
    print("=" * 60)

    print(
        pd.DataFrame(
            rows
        ).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()