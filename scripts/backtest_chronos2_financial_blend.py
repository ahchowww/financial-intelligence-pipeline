import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline

from src.processing.features import(
    build_financial_training_dataset,
)

from src.modeling.split import(
    chronological_train_val_test_split,
)

from src.modeling.metrics import(
    calculate_forecast_metrics,
)

DATA_PATH = "data/processed/tsla_quarterly_modeling.csv"

MODEL_ID = "autogluon/chronos-2-small"

MIN_TRAIN_SIZE = 20

FINANCIAL_COVARIATE_COLUMNS = [
    "gross_profit",
    "operating_income",
    "net_income",
    "cash",
    "inventory",
    "total_assets",
    "total_liabilities",
]

CHRONOS_WEIGHTS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]


def extract_scalar(
        value,
) -> float:
    """
    Convert Chronos output to one float.
    """

    if isinstance(value, torch.Tensor):
        value = (
            value.detach()
            .cpu()
            .numpy()
        )

    array = np.asarray(
        value,
        dtype=float,
    )

    return float(
        array.reshape(-1)[0]
    )


def main() -> None:
    # 1. Load dataset
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "quarter_end",
            "available_date",
        ]
    )

    df = (
        df.sort_values(
            "quarter_end"
        )
        .reset_index(
            drop=True
        )
    )

    training = build_financial_training_dataset(df)

    train, validation, test = chronological_train_val_test_split(training)


    print("\n" + "=" * 60)
    print("NAIVE + CHRONOS-2 FINANCIAL BLEND")
    print("=" * 60)

    print(f"Model: {MODEL_ID}")
    
    print(f"Training rows available: {len(train)}")
    print(f"Forecast origins: {len(train) - MIN_TRAIN_SIZE}")


    # 2. Load Chronos ONCE
    print("\nLoading Chronos-2...")

    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    print("Chronos-2 loaded.")

    actual_values = []
    naive_predictions = []
    chronos_predictions = []

    forecast_rows = []

    # 3. Generate forecasts once
    for index in range(MIN_TRAIN_SIZE, len(train)):
        current = train.iloc[[index]].copy()
        current_quarter_end = current["quarter_end"].iloc[0]

        actual = float(
            current["target_revenue"].iloc[0]
        )

        current_revenue = float(
            current["revenue"].iloc[0]
        )

        feature_quarter = str(current["year"].iloc[0]) + current["quarter"].iloc[0]

        # Naive
        naive_prediction = current_revenue

        # Chronos history
        # use all revenue observations that were known
        # up to and including the current quarter
        financial_history = (
            df.loc[
                df["quarter_end"] <= current_quarter_end,
                [
                    "quarter_end",
                    "revenue",
                    *FINANCIAL_COVARIATE_COLUMNS,
                ],
            ]
            .dropna()
            .sort_values(
                "quarter_end"
            )
            .copy()
        )


        context_df = financial_history[
            [
                "quarter_end",
                "revenue",
                *FINANCIAL_COVARIATE_COLUMNS,
            ]
        ].copy()

        context_df.insert(
            0,
            "item_id",
            "TSLA_REVENUE",
        )

        context_df = context_df.rename(
            columns={
                "quarter_end": "timestamp",
            }
        )

        # Zero-shot forecast
        prediction_df = pipeline.predict_df(
            context_df,
            prediction_length=1,
            quantile_levels=[
                0.1,
                0.5,
                0.9,
            ],
            id_column="item_id",
            timestamp_column="timestamp",
            target="revenue",
        )

        chronos_prediction = extract_scalar(
            prediction_df["predictions"].iloc[0]
        )

        # Save
        actual_values.append(actual)
        naive_predictions.append(naive_prediction)
        chronos_predictions.append(chronos_prediction)

        forecast_rows.append(
            {
                "feature_quarter": feature_quarter,
                "actual_revenue": actual,
                "naive_prediction": naive_prediction,
                "chronos_prediction": chronos_prediction,
            }
        )

        print(
            "{}: actual=${:,.0f}, "
            "Naive=${:,.0f}, "
            "Chronos=${:,.0f}".format(
                feature_quarter,
                actual,
                naive_prediction,
                chronos_prediction,
            )
        )

    # 4. Convert predictions to arrays
    actual_array = np.asarray(
        actual_values,
        dtype=float,
    )

    naive_array = np.asarray(
        naive_predictions,
        dtype=float,
    )

    chronos_array = np.asarray(
        chronos_predictions,
        dtype=float,
    )

    # 5. Try blend weights
    # Chronos weight = 0 -> pure Naive
    # Chronos weight = 1 -> pure Chronos 
    summary_rows = []

    blend_predictions = {}

    for chronos_weight in CHRONOS_WEIGHTS:
        naive_weight = 1.0 - chronos_weight
        
        blended = (
            naive_weight * naive_array + chronos_weight * chronos_array
        )

        metrics = calculate_forecast_metrics(
            actual=actual_array,
            predicted=blended,
        )

        summary_rows.append(
            {
                "chronos_weight": chronos_weight,
                "naive_weight": naive_weight,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
                "SMAPE": metrics["SMAPE"],
            }
        )

        blend_predictions[chronos_weight] = blended

    # 6. Rank by MAE
    results_df = (
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

    # 7. Results
    print("\n" + "=" * 60)
    print("BLEND RESULTS")
    print("=" * 60)
    print(display.to_string(index=False))


    # 8. Selected weight
    best = results_df.iloc[0]

    best_chronos_weight = float(
        best["chronos_weight"]
    )

    best_naive_weight = float(
        best["naive_weight"]
    )


    print("\n" + "=" * 60)
    print("SELECTED BLEND")
    print("=" * 60)
    print(
        f"Naive weight: {best_naive_weight:.2f}"
    )

    print(
        f"Chronos weight: {best_chronos_weight:.2f}"
    )

    print(
        f"Backtest MAE: ${best['MAE']:,.2f}"
    )

    print(
        f"Backtest RMSE: ${best['RMSE']:,.2f}"
    )

    print(
        f"Backtest MAPE: {best['MAPE']:.2f}%"
    )

    print(
        f"Backtest SMAPE: {best['SMAPE']:.2f}%"
    )

    # 9. Individual forecasts for selected blend
    selected_predictions = blend_predictions[best_chronos_weight]

    forecast_df = pd.DataFrame(
        forecast_rows
    )

    forecast_df["blended_prediction"] = selected_predictions

    print("\n" + "=" * 60)
    print("SELECTED BLEND FORECASTS")
    print("=" * 60)

    print(
        forecast_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()