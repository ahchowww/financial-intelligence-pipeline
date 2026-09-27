import pandas as pd

REQUIRED_COLUMNS = [
    "year",
    "quarter",
    "quarter_number",
    "quarter_end",
    "available_date",
    "revenue",
]


def build_revenue_forecasting_features(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build basic time-series features for forecasting next quarter revenue.
    -> take quarterly revenue history and create lag features (to predict next revenue)

    Each row = the information available at quarter t
    Target: revenue at quarter t+1
    """

    # 1. Validate required columns
    missing_columns = (
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # 2. Make a copy and ensure chronological order
    result = (
        df
        .copy()
        .sort_values(
            [
                "year",
                "quarter_number",
            ]
        )
        .reset_index(drop=True)
    )

    # 3. Historical revenue features
    result["revenue_lag1"] = result["revenue"].shift(1)
    result["revenue_lag2"] = result["revenue"].shift(2)
    result["revenue_lag3"] = result["revenue"].shift(3)
    result["revenue_lag4"] = result["revenue"].shift(4)

    # 4. Next-quarter prediction target
    result["target_revenue"] = result["revenue"].shift(-1)
    result["target_quarter_end"] = result["quarter_end"].shift(-1)

    return result


def build_revenue_training_dataset(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a model-ready dataset for next-quarter revenue forecasting.

    Forecast origin:
        quarter t, after that quarter's filing become available.

    Target:
        revenue at quarter t+1.

    # Rows without sufficient historical lag data or without 
    # a known next-quarter target are removed. 
    """

    result = build_revenue_forecasting_features(df)

    feature_columns = [
        "revenue",
        "revenue_lag1",
        "revenue_lag2",
        "revenue_lag3",
        "revenue_lag4",
    ]

    required_columns = (
        feature_columns 
        + 
        [
            "target_revenue",
            "target_quarter_end",
            "available_date",
        ]
    )

    # Keep only rows that can actually be used for supervised model training
    training = (
        result
        .dropna(
            subset=required_columns,
        )
        .copy()
        .reset_index(drop=True)
    )

    return training