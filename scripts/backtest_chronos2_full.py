import pandas as pd
import numpy as np
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

MODEL_ID = "amazon/chronos-2"

MIN_TRAIN_SIZE = 20


def extract_scalar(value) -> float:
    """
    Convert Chronos output to one float.

    Handles:
    - scalar
    - list
    - nested list
    - numpy array
    - torch tensor
    """

    if isinstance(value, torch.Tensor):
        value = (
            value
            .detach()
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

    # 2. Build supervised dataset only to obtain 
    # exactly the same historical forecast origins
    # used by our other models.
    training = build_financial_training_dataset(df)
    train, validation, test = chronological_train_val_test_split(training)


    print("\n" + "=" * 60)
    print("CHRONOS-2 FULL ZERO-SHOT WALK-FORWARD BACKTEST")
    print("=" * 60)

    print(f"Model: {MODEL_ID}")
    print("Device: CPU")
    
    print(f"Training forecast origins available: {len(train)}")
    print(f"Initial forecast origin: {MIN_TRAIN_SIZE}")
    print(f"Forecasts: {len(train) - MIN_TRAIN_SIZE}")


    # 3. Load Chronos ONCE
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

    # 4. Walk-forward backtest
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
        # use all revenue observations that were knownup to and including the current quarter
        revenue_history = (
            df.loc[
                df["quarter_end"] <= current_quarter_end,
                [
                    "quarter_end",
                    "revenue",
                ],
            ]
            .dropna()
            .sort_values(
                "quarter_end"
            )
            .copy()
        )


        context_df = pd.DataFrame(
            {
                "item_id": "TSLA_REVENUE",
                "timestamp": revenue_history["quarter_end"],
                "target": revenue_history["revenue"].astype(float),
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
            target="target",
        )

        chronos_prediction = extract_scalar(
            prediction_df["predictions"].iloc[0]
        )

        q10 = extract_scalar(
            prediction_df["0.1"].iloc[0]
        )

        q50 = extract_scalar(
            prediction_df["0.5"].iloc[0]
        )

        q90 = extract_scalar(
            prediction_df["0.9"].iloc[0]
        )

        # Save
        actual_values.append(actual)
        naive_predictions.append(naive_prediction)
        chronos_predictions.append(chronos_prediction)

        forecast_rows.append(
            {
                "feature_quarter": feature_quarter,
                "target_quarter_end": current["target_quarter_end"].iloc[0],
                "actual_revenue": actual,
                "naive_prediction": naive_prediction,
                "chronos_prediction": chronos_prediction,
                "q10": q10,
                "q50": q50,
                "q90": q90,
                "context_rows": len(revenue_history),
            }
        )

        feature_quarter_text = str(
            feature_quarter
        )

        print(
            "{}: actual=${:,.0f}, Chronos=${:,.0f}".format(
                feature_quarter_text,
                float(actual),
                float(chronos_prediction),
            )
        )
        
    # 5. Evaluate
    prediction_sets = {
        "Naive": naive_predictions,
        "Chronos-2 Full zero-shot": chronos_predictions,
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
        )
        .sort_values(
            "MAE"
        )
        .reset_index(
            drop=True
        )
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

    # 6. Results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(display.to_string(index=False))

    print("\n" + "=" * 60)
    print("INDIVIDUAL FORECASTS")
    print("=" * 60)
    print(
        pd.DataFrame(
            forecast_rows
        ).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()