from pathlib import Path

import pandas as pd

from src.processing.dataset import(
    load_quarterly_metric,
)

VALUE_COLUMNS = [
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "cash",
    "inventory",
    "total_assets",
    "total_liabilities",
]

FILED_COLUMNS = [
    f"{metric}_filed"
    for metric in VALUE_COLUMNS
]

METRIC_FILES = {
    "revenue": "tsla_quarterly_revenue.csv",
    "gross_profit": "tsla_quarterly_gross_profit.csv",
    "operating_income": "tsla_quarterly_operating_income.csv",
    "net_income": "tsla_quarterly_net_income.csv",
    "cash": "tsla_quarterly_cash.csv",
    "inventory": "tsla_quarterly_inventory.csv",
    "total_assets": "tsla_quarterly_total_assets.csv",
    "total_liabilities": "tsla_quarterly_total_liabilities.csv",
}

REQUIRED_COLUMNS = [
    "year",
    "quarter",
    "quarter_number",
    "quarter_end",
    *VALUE_COLUMNS,
    *FILED_COLUMNS,
    "is_complete",
    "available_date",
]

QUARTER_NUMBER_MAP = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "Q4": 4,
}


def validate_required_columns(
        df: pd.DataFrame,
        dataset_name: str,
) -> bool:
    """
    Ensure all required dataset columns exist.
    """

    missing = (
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    if missing:
        print(
            f"[FAIL] {dataset_name} is missing "
            f"required columns: {sorted(missing)}"
        )

        return False

    print(
        f"[PASS] {dataset_name} contains all required columns."
    )

    return True


def validate_no_duplicate_quarters(
        df: pd.DataFrame,
        dataset_name: str,
) -> bool:
    """
    Ensure each economic quarter appears only once.
    """

    duplicates = df.duplicated(
        subset=[
            "year",
            "quarter",
        ],
        keep=False,
    )

    if duplicates.any():
        print(
            f"[FAIL] Duplicate year-quarter rows found in {dataset_name}: "
        )

        print(
            df.loc[
                duplicates,
                [
                    "year",
                    "quarter",
                    "quarter_end",
                ],
            ].to_string(index=False)
        )

        return False

    print(f"[PASS] No duplicate year-quarter rows in {dataset_name}")

    return True


def validate_quarter_metadata(
        df: pd.DataFrame,
        dataset_name: str,
) -> bool:
    """
    Verify that:
        year
        quarter
        quarter_number
        quarter_end
    
    all describe the same calendar quarter.
    """

    expected_year = df["quarter_end"].dt.year

    expected_quarter_number = df["quarter_end"].dt.quarter

    expected_quarter = ("Q" + expected_quarter_number.astype(str))

    invalid = (
        (df["year"] != expected_year) | 
        (df["quarter_number"] != expected_quarter_number) |
        (df["quarter"] != expected_quarter)
    )

    if invalid.any():
        print(
            f"[FAIL] Quarter metadata mismatch in {dataset_name}: "
        )

        print(
            df.loc[
                invalid,
                [
                    "year",
                    "quarter",
                    "quarter_number",
                    "quarter_end",
                ],
            ].to_string(index=False)
        )

        return False

    print(
        f"[PASS] Quarter metadata is consistent in {dataset_name}."
    )

    return True

def validate_chronological_order(
        df: pd.DataFrame,
        dataset_name: str,
) -> bool:
    """
    Ensure rows are chronologically ordered.
    """

    quarter_index = (
        df["year"] * 4 + df["quarter_number"]
    )

    if not quarter_index.is_monotonic_increasing:
        print(
            f"[FAIL] {dataset_name} is not chronologically ordered."
        )

        return False
    
    print(f"[PASS] {dataset_name} is chronologically ordered.")

    return True


def validate_filing_dates(
        df: pd.DataFrame,
        dataset_name: str,
) -> bool:
    """
    Every available metric filing date must be on or after the economic quarter end.
    """

    problems = []

    for filed_column in FILED_COLUMNS:
        invalid = (
            df[filed_column].notna() & (df[filed_column] < df["quarter_end"])
        )

        if invalid.any():
            for _, row in df.loc[invalid].iterrows():
                problems.append(
                    {
                        "year": row["year"],
                        "quarter": row["quarter"],
                        "column": filed_column,
                        "quarter_end": row["quarter_end"],
                        "filed": row[filed_column],
                    }
                )

    if problems:
        print(
            f"[FAIL] Filing dates before quarter end found in {dataset_name}:"
        )

        print(
            pd.DataFrame(
                problems
            ).to_string(index=False)
        )

        return False

    print(f"[PASS] All filing dates are on or after quarter end in {dataset_name}.")

    return True


def validate_is_complete(
        master: pd.DataFrame,
) -> bool:
    """
    Verify that is_complete is True exactly 
    when all 8 financial metric values are populated.
    """

    expected = (
        master[VALUE_COLUMNS]
        .notna()
        .all(axis=1)
    )

    invalid = (
        master["is_complete"] != expected
    )

    if invalid.any():
        print("[FAIL] is_complete does not match actual metric completeness:")

        print(
            master.loc[
                invalid,
                [
                    "year",
                    "quarter",
                    "is_complete",
                ]
                + VALUE_COLUMNS,
            ].to_string(index=False)
        )

        return False

    print("[PASS] is_complete matches actual metric completeness.")

    return True


def validate_available_date(
        df: pd.DataFrame,
        dataset_name: str,
) -> bool:
    """
    available_date must equal the latest filing date among all required metrics.
    Incomplete rows should have available_date = NaT.
    """

    expected = (
        df[FILED_COLUMNS].max(
            axis=1,
            skipna=False,
        )
    )

    actual = df["available_date"]

    matches = (
        (actual == expected) | (actual.isna() & expected.isna())
    )

    invalid = ~matches

    if invalid.any():
        print(
            f"[FAIL] Incorrect available_date values in {dataset_name}:"
        )

        problem_rows = (
            df.loc[
                invalid,
                [
                    "year",
                    "quarter",
                    "available_date",
                ]
                + FILED_COLUMNS,
            ]
            .copy()
        )

        problem_rows["expected_available_date"] = (
            expected.loc[invalid]
        )

        print(
            problem_rows.to_string(index=False)
        )

        return False

    print(f"[PASS] available_date is correct in {dataset_name}.")

    return True


def validate_modeling_complete(
        modeling: pd.DataFrame,
) -> bool:
    """
    Modeling dataset must contain no missing values 
    in the 8 required financial metrics.
    """

    missing = (
        modeling[VALUE_COLUMNS]
        .isna()
        .any(axis=1)
    )

    if missing.any():
        print("[FAIL] Modeling dataset contains missing financial values:")

        print(
            modeling.loc[
                missing,
                [
                    "year",
                    "quarter",
                ]
                + VALUE_COLUMNS,
            ].to_string(index=False)
        )

        return False

    if not modeling["is_complete"].all():
        print("[FAIL] Some modeling rows are not marked complete.")

        return False

    print("[PASS] All modeling rows contain all 8 financial metrics.")

    return True


def validate_modeling_continuity(
        modeling: pd.DataFrame,
) -> bool:
    """
    Ensure every modeling row is exactly one quarter after the previous row.
    """

    quarter_index = (
        modeling["year"] * 4 + modeling["quarter_number"]
    )

    differences = quarter_index.diff()

    invalid = (differences.iloc[1:] != 1)

    if invalid.any():
        problem_positions = (
            invalid[invalid]
            .index
        )

        print("[FAIL] Gaps found in modeling quarterly history:")

        for index in problem_positions:
            previous = (
                modeling.loc[index - 1]
            )

            current = (
                modeling.loc[index]
            )

            print(
                f"  "
                f"{previous['year']} "
                f"{previous['quarter']}"
                " -> "
                f"{current['year']} "
                f"{current['quarter']}"
            )

        return False

    print("[PASS] Modeling dataset is a continuous quarterly series.")

    return True


def validate_modeling_period(
        master: pd.DataFrame,
        modeling: pd.DataFrame,
) -> bool:
    """
    Verify that the modeling dataset begins immediately 
    after the final incomplete master row and then uses
    every subsequent quarter.
    """

    incomplete_indices = (
        master.index[~master["is_complete"]]
    )

    if len(incomplete_indices) == 0:
        expected_start_index = 0
    else:
        expected_start_index = (incomplete_indices.max() + 1)

    expected = (
        master.loc[
            expected_start_index:,
            [
                "year",
                "quarter",
            ],
        ]
        .reset_index(drop=True)
    )

    actual = (
        modeling[
            [
                "year",
                "quarter",
            ]
        ]
        .reset_index(drop=True)
    )

    if not actual.equals(expected):
        print("[FAIL] Modeling dataset is not the expected continuous complete suffix of the master dataset.")

        return False

    print("[PASS] Modeling period begins after the final incomplete master quarter.")

    return True


def validate_source_metric_alignment(
    master: pd.DataFrame,
    processed_dir: Path,
) -> bool:
    """
    Verify that values and filing dates in the merged master
    dataset still match the original validated metric files.
    -> protects against accidental changes introduced during dataset merging.
    """

    all_valid = True

    for metric_name, filename in METRIC_FILES.items():
        # 1. Load original validated metric file
        source = load_quarterly_metric(
            path=(processed_dir / filename),
            metric_name=metric_name,
        )

        filed_column = (
            f"{metric_name}_filed"
        )

        # 2. Select corresponding columns from master
        master_metric = master[
            [
                "year",
                "quarter",
                "quarter_end",
                metric_name,
                filed_column,
            ]
        ].copy()

        # 3. Compare source and master
        comparison = source.merge(
            master_metric,
            on=[
                "year",
                "quarter",
            ],
            how="outer",
            suffixes=(
                "_source",
                "_master",
            ),
            indicator=True,
        )

        metric_valid = True

        # 4. Every source quarter must exist in master
        missing_from_master = (
            comparison["_merge"] == "left_only"
        )

        if missing_from_master.any():
            metric_valid = False
            all_valid = False

            print(
                f"[FAIL] {metric_name}: source quarters are missing from master dataset."
            )

            print(
                comparison.loc[
                    missing_from_master,
                    [
                        "year",
                        "quarter",
                    ],
                ].to_string(index=False)
            )

        # 5. Compare quarters that exist in both datasets
        both = (
            comparison["_merge"] == "both"
        )

        quarter_end_match = (
            comparison["quarter_end_source"] == comparison["quarter_end_master"]
        )

        value_match = (
            comparison[f"{metric_name}_source"] == comparison[f"{metric_name}_master"]
        )

        filed_match = (
            comparison[f"{filed_column}_source"] == comparison[f"{filed_column}_master"]
        )

        mismatch = (
            both & (~quarter_end_match | ~value_match | ~filed_match)
        )

        if mismatch.any():
            metric_valid = False
            all_valid = False

            print(
                f"[FAIL] {metric_name}: merged values differ from the original metric file."
            )

            print(
                comparison.loc[
                    mismatch,
                    [
                        "year",
                        "quarter",
                        "quarter_end_source",
                        "quarter_end_master",
                        f"{metric_name}_source",
                        f"{metric_name}_master",
                        f"{filed_column}_source",
                        f"{filed_column}_master",
                    ],
                ].to_string(index=False)
            )

        # 6. Quarters absent from source should remain
        #    missing for that metric in master

        master_only = (
            comparison["_merge"] == "right_only"
        )

        unexpected_values = (
            master_only &
            (comparison[f"{metric_name}_master"].notna() | 
             comparison[f"{filed_column}_master"].notna()
            )
        )

        if unexpected_values.any():
            metric_valid = False
            all_valid = False

            print(
                f"[FAIL] {metric_name}: master contains values that do not exist in the source file."
            )

            print(
                comparison.loc[
                    unexpected_values,
                    [
                        "year",
                        "quarter",
                        f"{metric_name}_master",
                        f"{filed_column}_master",
                    ],
                ].to_string(index=False)
            )

        # 7. Metric passed
        if metric_valid:
            print(
                f"[PASS] {metric_name} matches its source metric file."
            )

    return all_valid


# ============================================================
# Main validator
# ============================================================

def validate_quarterly_dataset(
    master: pd.DataFrame,
    modeling: pd.DataFrame,
    processed_dir: Path,
) -> bool:
    """
    Run all validation checks for the merged
    quarterly financial dataset.
    """

    checks = [
        validate_required_columns(
            master,
            "master dataset",
        ),

        validate_required_columns(
            modeling,
            "modeling dataset",
        ),

        validate_no_duplicate_quarters(
            master,
            "master dataset",
        ),

        validate_no_duplicate_quarters(
            modeling,
            "modeling dataset",
        ),

        validate_quarter_metadata(
            master,
            "master dataset",
        ),

        validate_quarter_metadata(
            modeling,
            "modeling dataset",
        ),

        validate_chronological_order(
            master,
            "master dataset",
        ),

        validate_chronological_order(
            modeling,
            "modeling dataset",
        ),

        validate_filing_dates(
            master,
            "master dataset",
        ),

        validate_filing_dates(
            modeling,
            "modeling dataset",
        ),

        validate_is_complete(
            master
        ),

        validate_available_date(
            master,
            "master dataset",
        ),

        validate_available_date(
            modeling,
            "modeling dataset",
        ),

        validate_modeling_complete(
            modeling
        ),

        validate_modeling_continuity(
            modeling
        ),

        validate_modeling_period(
            master,
            modeling,
        ),

        validate_source_metric_alignment(
            master,
            processed_dir,
        ),
    ]

    return all(checks)