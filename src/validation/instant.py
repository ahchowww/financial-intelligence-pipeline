import pandas as pd

REQUIRED_QUARTERS = {
    "Q1",
    "Q2",
    "Q3",
    "Q4",
}

QUARTER_ORDER = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "Q4": 4,
}


def validate_no_duplicate_quarters(
        df:pd.DataFrame,
) -> bool:
    """
    Ensure there is only one observation for each economic year-quarter.
    """

    duplicates = df.duplicated(
        subset=[
            "year",
            "quarter",
        ],
        keep=False,
    )

    if duplicates.any():
        print("[FAIL] Duplicate year-quarter rows found:")

        print(
            df.loc[
                duplicates,
                [
                    "year",
                    "quarter",
                    "end",
                    "val",
                    "filed",
                ],
            ].to_string(index=False)
        )

        return False

    print("[PASS] No duplicate year-quarter found.")

    return True


def validate_required_values(
        df: pd.DataFrame,
) -> bool:
    """
    Ensure important output fields are populated.
    """

    required_columns = [
        "year",
        "quarter",
        "end",
        "val",
        "filed",
        "form",
        "derived",
        "source_method",
    ]

    missing = df[
        required_columns
    ].isna()

    if missing.any().any():
        print("\n[FAIL] Missing required values found.")

        problem_rows = df[
            missing.any(axis=1)
        ]

        print(
            problem_rows[
                required_columns
            ].to_string(index=False)
        )

        return False

    print("[PASS] No missing required values.")

    return True


def validate_filing_dates(
        df: pd.DataFrame,
) -> bool:
    """
    A financial fact cannot become available before its economic balance date.
    """

    invalid = (
        df["filed"] < df["end"]
    )

    if invalid.any():
        print("[FAIL] Filing dates before balance dates found:")

        print(
            df.loc[
                invalid,
                [
                    "year",
                    "quarter",
                    "end",
                    "filed",
                ],
            ].to_string(index=False)
        )

        return False

    print("[PASS] All filing dates are on or after balance dates.")

    return True


def validate_quarter_labels(
        df: pd.DataFrame,
) -> bool:
    """
    Verify that year and quarter agree with the balance sheet end date.
    """

    expected_year = (
        df["end"].dt.year
    )

    expected_quarter = (
        "Q"
        + df["end"]
        .dt.quarter
        .astype(str)
    )

    invalid = (
        (df["year"] != expected_year) | (df["quarter"] != expected_quarter)
    )

    if invalid.any():
        print("[FAIL] Quarter labels do not match balance dates:")

        print(
            df.loc[
                invalid,
                [
                    "year",
                    "quarter",
                    "end",
                ],
            ].to_string(index=False)
        )

        return False

    print("[PASS] Quarter labels agree with balance dates.")

    return True


def validate_calendar_quarter_ends(
        df: pd.DataFrame,
) -> bool:
    """
    Ensure observations occur on calendar quarter-end dates.
    """

    valid_month_day = {
        (3, 31),
        (6, 30),
        (9, 30),
        (12, 31),
    }

    invalid = ~df["end"].apply(
        lambda date: (
            date.month,
            date.day,
        ) in valid_month_day
    )

    if invalid.any():
        print("[FAIL] Non-quarter-end balance dates found:")

        print(
            df.loc[
                invalid,
                [
                    "year",
                    "quarter",
                    "end",
                ],
            ].to_string(index=False)
        )

        return False

    print("[PASS] All observations are calendar quarter-end balances.")

    return True


def validate_chronological_order(
        df: pd.DataFrame,
) -> bool:
    """
    Ensure the quarterly dataset is chronologically ordered.
    """

    expected = (
        df
        .sort_values(
            by=[
                "year",
                "quarter_number",
            ]
        )
        .reset_index(drop=True)
    )

    actual = df.reset_index(drop=True)

    if not actual[
        [
            "year",
            "quarter",
        ]
    ].equals(
        expected[
            [
                "year",
                "quarter",
            ]
        ]
    ):
        print("[FAIL] Dataset is not in chronological order.")

        return False

    print("[PASS] Dataset is in chronological order.")

    return True


def validate_source_metadata(
        df: pd.DataFrame,
) -> bool:
    """
    Instance balance should be direct SEC observations.
    """

    invalid = (
        (df['derived'] != False) | (df["source_method"] != "direct")
    )

    if invalid.any():
        print("[FAIL] Invalid instant source metadata found:")

        print(
            df.loc[
                invalid,
                [
                    "year",
                    "quarter",
                    "derived",
                    "source_method",
                ],
            ].to_string(index=False)
        )

        return False

    print("[PASS] All instant observations are direct SEC facts.")

    return True


def validate_complete_history(
        df: pd.DataFrame,
) -> bool:
    """
    Find the first year containing Q1-Q4.
    -> From the year onward, require complete historical years.
    -> The latest year may be partial.
    """

    quarters_by_year = (
        df.groupby("year")["quarter"]
        .apply(set)
    )

    complete_years = [
        int(year)
        for year, quarters in quarters_by_year.items()
        if REQUIRED_QUARTERS.issubset(
            quarters
        )
    ]

    if not complete_years:
        print("[FAIL] No complete Q1-Q4 year was found.")

        return False

    first_complete_year = min(
        complete_years
    )

    latest_year = int(
        df["year"].max()
    )

    latest_quarters = (
        quarters_by_year.loc[latest_year]
    )

    # If latest year does not yet have Q4,
    # treat it as an in-progress year
    if REQUIRED_QUARTERS.issubset(latest_quarters):
        last_required_year = latest_year
    else:
        last_required_year = (latest_year - 1)

    problems = []

    for year in range(first_complete_year, last_required_year + 1):
        actual = quarters_by_year.get(
            year,
            set(),
        )

        missing = (
            REQUIRED_QUARTERS - actual
        )

        if missing:
            problems.append(
                (
                    year,
                    sorted(missing),
                )
            )

    if problems:
        print("[FAIL] Missing quarters after the first complete year.")

        for year, missing in problems:
            print(f" {year}: missing {missing}")

            return False

    print(
        "[PASS] Quarterly history is complete "
        f"from {first_complete_year} through "
        f"{last_required_year}."
    )

    return True


def validate_earliest_filing_selection(
        quarterly_df: pd.DataFrame,
        source_facts: pd.DataFrame,
) -> bool:
    """
    Verify the processed dataset uses the earliest available SEC filing 
    for each balance date.
    """

    instant_source = source_facts[
        source_facts["fact_type"] == "instant"
    ].copy()

    earliest = (
        instant_source
        .groupby("end")["filed"]
        .min()
    )

    problems = []

    for _, row in quarterly_df.iterrows():
        end = row["end"]

        if end not in earliest.index:
            continue

        expected_filed = (
            earliest.loc[end]
        )

        if row["filed"] != expected_filed:
            problems.append(
                {
                    "end": end,
                    "selected_filed": row["filed"],
                    "earliest_filed": expected_filed,
                }
            )

    if problems:
        print("[FAIL] Some balances do not use the earliest filing:")

        print(
            pd.DataFrame(problems)
            .to_string(index=False)
        )

        return False

    print("[PASS] Earliest filing selected for every balance date.")

    return True


def validate_nonnegative(
        df: pd.DataFrame,
        metric_name: str,
) -> bool:
    """
    Validate metrics that are expected to be non-negative.
    """

    invalid = (
        df["val"] < 0
    )

    if invalid.any():
        print(f"[FAIL] Negative {metric_name} values found:")

        print(
            df.loc[
                invalid,
                [
                    "year",
                    "quarter",
                    "val",
                ],
            ].to_string(index=False)
        )

        return False

    print(f"[PASS] All {metric_name} values are non-negative.")

    return True

def validate_quarterly_instant_metric(
        quarterly_df: pd.DataFrame,
        source_facts: pd.DataFrame,
        metric_name: str,
        require_nonnegative: bool = False,
) -> bool:
    """
    Run validation checks for a quarterly instant metric.
    """

    checks = [
        validate_no_duplicate_quarters(
            quarterly_df
        ),

        validate_required_values(
            quarterly_df
        ),

        validate_filing_dates(
            quarterly_df
        ),

        validate_quarter_labels(
            quarterly_df
        ),

        validate_calendar_quarter_ends(
            quarterly_df
        ),

        validate_chronological_order(
            quarterly_df
        ),

        validate_source_metadata(
            quarterly_df
        ),

        validate_complete_history(
            quarterly_df
        ),

        validate_earliest_filing_selection(
            quarterly_df,
            source_facts,
        ),
    ]

    if require_nonnegative:
        checks.append(
            validate_nonnegative(
                quarterly_df,
                metric_name,
            )
        )

    return all(checks)
