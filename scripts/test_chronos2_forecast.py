import pandas as pd
import torch

from chronos import Chronos2Pipeline

from src.processing.features import(
    build_financial_training_dataset,
)

from src.modeling.split import(
    chronological_train_val_test_split,
)

DATA_PATH = "data/processed/tsla_quarterly_modeling.csv"

MODEL_ID = "autogluon/chronos-2-small"

FORECAST_INDEX = 20


def main() -> None:
    # 1. Load dataset
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "quarter_end",
            "available_date",
        ]
    )

    training = build_financial_training_dataset(df)

    train, validation, test = chronological_train_val_test_split(training)


    # 2. Reproduce the FIRST historical backtest origin
    # use rows 0 - 20
    # Current quarter: 2017Q2
    # Forecast: 2017Q3
    current = train.iloc[[FORECAST_INDEX]].copy()
    history = train.iloc[:FORECAST_INDEX+1].copy()

    actual_revenue = float(
        current["target_revenue"].iloc[0]
    )

    current_revenue = float(
        current["revenue"].iloc[0]
    )

    feature_quarter = str(current["year"].iloc[0]) + current["quarter"].iloc[0]


    # 3. Convert to chronos Long-format dataframe
    # Chronos expects: item_id, timestamp, target
    context_df = pd.DataFrame(
        {
            "item_id": "TSLA_REVENUE",
            "timestamp": history["quarter_end"],
            "target": history["revenue"].astype(float),
        }
    )

    # 4. Load pretrained Chronos-2 model
    # Zero-shot: No Tesla-specific model training happens here.
    print("\nLoading Chronos-2...")
    
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    print("Chronos-2 loaded.")

    # 5. Forecast one quarter ahead
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

    # Chronos returns a point prediction
    chronos_prediction = float(
        prediction_df["predictions"].iloc[0]
    )

    # 6. Print result
    print("\n" + "=" * 60)
    print("CHRONOS-2 ZERO-SHOT TEST")
    print("=" * 60)

    print(f"Model: {MODEL_ID}")
    print("Device: CPU")
    
    print(f"History rows: {len(history)}")
    print(f"Feature quarter: {feature_quarter}")

    print(
        f"Current revenue: "
        f"${current_revenue:,.2f}"
    )

    print(
        f"Actual next revenue: "
        f"${actual_revenue:,.2f}"
    )

    print(
        f"Chronos prediction: "
        f"${chronos_prediction:,.2f}"
    )

    print("\nChronos output: ")

    print(prediction_df.to_string(index=False))


if __name__ == "__main__":
    main()
