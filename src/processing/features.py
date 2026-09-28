import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "year",
    "quarter",
    "quarter_number",
    "quarter_end",
    "available_date",
    "revenue",
]

FINANCIAL_FEATURE_COLUMNS = [
    "revenue",
    "revenue_lag1",
    "revenue_lag2",
    "revenue_lag3",
    "revenue_lag4",
    "revenue_qoq_growth",
    "revenue_yoy_growth",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "cash_to_assets",
    "inventory_to_assets",
    "liabilities_to_assets",
]

COMPACT_FINANCIAL_FEATURE_COLUMNS = [
    "revenue",
    "revenue_lag1",
    "revenue_lag4",
    "gross_margin",
    "operating_margin",
    "cash_to_assets",
    "liabilities_to_assets",
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


def build_financial_forecasting_features(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build multivariate financial features for next-quarter revenue forecasting.

    Each row represents info known for quarter t after all required financial metrics
    for that quarter have become available.


    Target:
        revenue at quarter t+1
    """

    required_columns = {
        "year",
        "quarter",
        "quarter_number",
        "quarter_end",
        "available_date",
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "cash",
        "inventory",
        "total_assets",
        "total_liabilities",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # 1. Sort chronologically
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

    # 2. Revenue lag features
    result["revenue_lag1"] = result["revenue"].shift(1)
    result["revenue_lag2"] = result["revenue"].shift(2)
    result["revenue_lag3"] = result["revenue"].shift(3)
    result["revenue_lag4"] = result["revenue"].shift(4)

    # 3. Revenue growth features
    result["revenue_qoq_growth"] = (
        result["revenue"] / result["revenue_lag1"] - 1
    )

    result["revenue_yoy_growth"] = (
        result["revenue"] / result["revenue_lag4"] - 1
    )

    # 4. Profitable ratios
    result["gross_margin"] = (
        result["gross_profit"] / result["revenue"]
    )

    result["operating_margin"] = (
        result["operating_income"] / result["revenue"]
    )

    result["net_margin"] = (
        result["net_income"] / result["revenue"]
    )

    # 5. Balance sheet ratios
    result["cash_to_assets"] = (
        result["cash"] / result["total_assets"]
    )

    result["inventory_to_assets"] = (
        result["inventory"] / result["total_assets"]
    )

    result["liabilities_to_assets"] = (
        result["total_liabilities"] / result["total_assets"]
    )

    # 6. Next-quarter target
    result["target_revenue"] = result["revenue"].shift(-1)

    result["target_quarter_end"] = result["quarter_end"].shift(-1)

    return result


def build_financial_training_dataset(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a model-ready multivariate dataset for next-quarter revenue forecasting.
    """

    result = build_financial_forecasting_features(df)

    result[FINANCIAL_FEATURE_COLUMNS] = (
        result[FINANCIAL_FEATURE_COLUMNS]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )
    
    required_columns = (
        FINANCIAL_FEATURE_COLUMNS 
        + 
        [
            "target_revenue",
            "target_quarter_end",
            "available_date",
        ]
    )

    training = (
        result
        .dropna(
            subset=required_columns,
        )
        .copy()
        .reset_index(drop=True)
    )

    return training

