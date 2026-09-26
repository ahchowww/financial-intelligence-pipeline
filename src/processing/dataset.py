from pathlib import Path
import pandas as pd

"""
Utilities for building a merged quarterly financial dataset 
from individually validated metric files.
"""

def load_quarterly_metric(
        path: Path,
        metric_name: str,
) -> pd.DataFrame:
    """
    Load one processed quarterly metric and convert it into a merge-ready table.
    e.g. 
        val -> revenue
        filed -> revenue_filed
        end -> quarter_end

    Only the columns needed for the master dataset are kept.
    """

    df = pd.read_csv(
        path,
        parse_dates=[
            "end",
            "filed",
        ],
    )

    required_columns = {
        "year",
        "quarter",
        "quarter_number",
        "end",
        "val",
        "filed",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ensure one row per economic quarter
    duplicates = df.duplicated(
        subset=[
            "year",
            "quarter",
        ], 
        keep=False,
    )

    if duplicates.any():
        raise ValueError(
            f"{path.name} contains duplicate year-quarter rows."
        )

    # keep only columns needed for merging
    result = df[
        [
            "year",
            "quarter",
            "quarter_number",
            "end",
            "val",
            "filed",
        ]
    ].copy()

    # standardize names
    result = result.rename(
        columns={
            "end": "quarter_end",
            "val": metric_name,
            "filed": f"{metric_name}_filed",
        }
    )

    return result

def merge_quarterly_metric_tables(
        left: pd.DataFrame,
        right: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge quarterly metric tables using economic year and quarter.
    -> The merge is outer so that quarters are not silently dropped 
       when one metric has shorter historical coverage.
    -> Quarter metadata must agree whenever both tables contain the same quarter. 
    """

    merged = left.merge(
        right,
        on=[
            "year",
            "quarter",
        ],
        how="outer",
        suffixes=(
            "_left",
            "_right",
        ),
    )

    # 1. Check quarter-number consistency
    both_quarter_numbers = (
        merged["quarter_number_left"].notna() & merged["quarter_number_right"].notna()
    )

    quarter_number_mismatch = (
        both_quarter_numbers 
        & 
        (merged["quarter_number_left"] != merged["quarter_number_right"])
    )

    if quarter_number_mismatch.any():
        raise ValueError(
            "Quarter-number mismatch found while merging metric tables."
        )

    # 2. Check quarter-end consistency
    both_quarter_ends = (
        merged["quarter_end_left"].notna() & merged["quarter_end_right"].notna()
    )

    quarter_end_mismatch = (
        both_quarter_ends
        &
        (merged["quarter_end_left"] != merged["quarter_end_right"])
    )

    if quarter_end_mismatch.any():
        raise ValueError(
            "Quarter-end mismatch found while merging metric tables."
        )

    # 3. Combine shared quarter metadata
    merged["quarter_number"] = (
        merged["quarter_number_left"]
        .combine_first(
            merged["quarter_number_right"]
        )
    )

    merged["quarter_end"] = (
        merged["quarter_end_left"]
        .combine_first(
            merged["quarter_end_right"]
        )
    )

    # 4. Remove temporary duplicate metadata columns
    merged = merged.drop(
        columns=[
            "quarter_number_left",
            "quarter_number_right",
            "quarter_end_left",
            "quarter_end_right",
        ]
    )

    # 5. Sort chronologically
    merged = (
        merged
        .sort_values(
            by=[
                "year",
                "quarter_number",
            ]
        )
        .reset_index(drop=True)
    )

    return merged


def build_quarterly_dataset(
        processed_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build:
    1. Master quarterly dataset
       - keeps all quarters from all metrics
       - preserves missing historical observations

    2. Modeling quarterly dataset
       - keeps only quarters where all required financial metrics are available
    
    The modeling dataset is therefore suitable for feature engineering and forecasting.
    """

    # 1. Define the financial metrics we want
    metric_files = {
        "revenue": "tsla_quarterly_revenue.csv",
        "gross_profit": "tsla_quarterly_gross_profit.csv",
        "operating_income": "tsla_quarterly_operating_income.csv",
        "net_income": "tsla_quarterly_net_income.csv",
        "cash": "tsla_quarterly_cash.csv",
        "inventory": "tsla_quarterly_inventory.csv",
        "total_assets": "tsla_quarterly_total_assets.csv",
        "total_liabilities": "tsla_quarterly_total_liabilities.csv",
    }

    # 2. Load each metric
    metric_tables = {}

    for metric_name, filename in metric_files.items():
        path = (processed_dir / filename)

        metric_tables[metric_name] = load_quarterly_metric(
            path=path,
            metric_name=metric_name,
        )

    # 3. Start with revenue
    master = metric_tables["revenue"].copy()

    # 4. Outer-merge remaining metrics
    remaining_metrics = [
        "gross_profit",
        "operating_income",
        "net_income",
        "cash",
        "inventory",
        "total_assets",
        "total_liabilities",
    ]

    for metric_name in remaining_metrics:
        master = (
            merge_quarterly_metric_tables(
                master,
                metric_tables[metric_name],
            )
        )

    # 5. Define financial value columns
    value_columns = [
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "cash",
        "inventory",
        "total_assets",
        "total_liabilities",
    ]

    filed_columns = [
        f"{metric}_filed"
        for metric in value_columns
    ]

    # 6. Identify complete rows
    master["is_complete"] = (
        master[value_columns]
        .notna()
        .all(axis=1)
    )

    # 7. Calculate full-row availability date
    # The row is only usable after the last required metric became available.
    master["available_date"] = (
        master[filed_columns]
        .max(
            axis=1,
            skipna=False,
        )
    )

    # 8. Sort master dataset
    master = (
        master.sort_values(
            by=[
                "year",
                "quarter_number",
            ]
        )
        .reset_index(drop=True)
    )

    # normalize final dataset dtypes
    master["year"] = (master["year"].astype("Int64"))
    master["quarter_number"] = (master["quarter_number"].astype("Int64"))    

    # --------------------------------------------------
    # 9. Create continuous modeling dataset
    #
    # A forecasting time series must not contain gaps.
    # Starting from the latest quarter, walk backwards
    # until we reach an incomplete quarter.
    # --------------------------------------------------
    
    complete_mask = master["is_complete"]

    incomplete_indices = master.index[
        ~complete_mask
    ]

    if len(incomplete_indices) == 0:
      modeling_start_index = 0
    else:
        last_incomplete_index = incomplete_indices.max()
        modeling_start_index = last_incomplete_index + 1

    modeling = (
        master.loc[
            modeling_start_index:
        ]
        .copy()
        .reset_index(drop=True)
    )

    return master, modeling

