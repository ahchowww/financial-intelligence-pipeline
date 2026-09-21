import pandas as pd

def validate_no_duplicate_quarters(
        df: pd.DataFrame,
) -> bool:
    """
    Ensure each year-quarter appears only once.
    """

    duplicates = df.duplicated(
        subset=["year", "quarter"],
        keep=False,
    )

    if duplicates.any():
        print("\n[FAIL] Duplicate quarters found: ")

        print(
            df.loc[
                duplicates,
                [
                    "year",
                    "quarter",
                    "val",
                    "filed",
                ],
            ].to_string(index=False)
        )

        return False

    print("[PASS] No duplicate year-quarter rows.")

    return True


def validate_no_missing_values(
        df: pd.DataFrame,
) -> bool:
    """
    Ensure required columns do not contain missing values.
    """

    required_columns = [
        "year",
        "quarter",
        "start",
        "end",
        "val",
        "filed",
        "derived",
        "source_method",
    ]

    missing = (
        df[required_columns]
        .isna()
        .sum()
    )

    problems = missing[
        missing > 0
    ]

    if not problems.empty:
        print("\n[FAIL] Missing required values: ")

        print(problems)

        return False

    print("[PASS] No missing required values.")

    return True


def validate_revenue_nonnegative(
        df: pd.DataFrame,
) -> bool:
    """
    Revenue should not be negative.
    """

    invalid = df[
        df["val"] < 0
    ]

    if not invalid.empty:
        print("[FAIL] Negative revenue values found:")

        print(
            invalid[
                [
                    "year",
                    "quarter",
                    "val",
                ]
            ].to_string(index=False)
        )

        return False

    print("[PASS] ALL revenue values are non-negative.")

    return True


def validate_filing_after_period_end(
        df: pd.DataFrame,
) -> bool:
    """
    Filing date should not occur before the financial end period.
    """

    invalid = df[
        df["filed"] < df["end"]
    ]

    if not invalid.empty:
        print("\n[FAIL] Filing date before period end: ")

        print(
            invalid[
                [
                    "year",
                    "quarter",
                    "end",
                    "filed",
                ]
            ].to_string(index=False)
        )

        return False

    print("[PASS] All filing dates are on or after period end.")

    return True


def validate_complete_years(
        df:pd.DataFrame,
) -> bool:
    """
    Ensure all completed historical years contain Q1-Q4.
    The latest year is allowed to be incomplete.
    """

    latest_year = df["year"].max()

    historical = df[
        df["year"] < latest_year
    ]

    expected_quarters = {
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    }

    problems = []

    for year, group in historical.groupby("year"):
        actual_quarters = set(
            group["quarter"]
        )

        if actual_quarters != expected_quarters:

            missing =(
                expected_quarters - actual_quarters
            )

            problems.append(
                {
                    "year": year,
                    "missing": sorted(missing),
                }
            )

    if problems:
        print("\n[FAIL] Incomplete historical years: ")

        for problem in problems:
            print(
                f"{problem['year']}: "
                f"missing {problem['missing']}"
            )

        return False

    print("[PASS] All completed historical years contain Q1-Q4.")

    return True


def validate_chronological_order(
        df: pd.DataFrame,
) -> bool:
    """
    Ensure the dataset is ordered by year and quarter.
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

    current = df.reset_index(drop=True)

    if not current[
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
    Check consistency between derived and source_method.
    """

    direct_invalid = df[
        (~df["derived"]) & (df["source_method"] != "direct")
    ]

    derived_invalid = df[
        (df["derived"] & df["source_method"] == "direct")
    ]

    if (not direct_invalid.empty or not derived_invalid.empty):
        print("[FAIL] Inconsistent source metadata.")

        return False

    print("[PASS] Consistent source metadata.")

    return True


def validate_against_annual_facts(
        quarterly_df: pd.DataFrame,
        revenue_facts: pd.DataFrame,
        tolerance: float = 1_000_000,
) -> bool:
    """
    Check whether Q1 + Q2 + Q3 + Q4 approximately equals to SEC annual
    """

    annual = revenue_facts[
        revenue_facts["duration_type"] == "annual"
    ].copy()

    annual["year"] = (
        annual["end"].dt.year
    )

    # earliest publicly available annual fact
    annual = (
        annual
        .sort_values(
            by=[
                "year",
                "filed",
            ]
        )
        .drop_duplicates(
            subset=["year"],
            keep="first",
        )
    )

    quarterly_totals =(
        quarterly_df
        .groupby("year")["val"]
        .sum()
        .reset_index(
            name="quarter_sum"
        )
    )

    comparison = annual[
        [
            "year",
            "val",
        ]
    ].rename(
        columns={
            "val": "annual_value"
        }
    )

    comparison = comparison.merge(
        quarterly_totals,
        on="year",
        how="inner",
    )

    comparison["difference"] = (
        comparison["quarter_sum"] - comparison["annual_value"]
    )

    comparison["abs_difference"] = comparison["difference"].abs()

    invalid = comparison[
        comparison["abs_difference"] > tolerance
    ]

    if not invalid.empty:
        print("\n[FAIL] Quarterly totals do not reconcile with annual SEC facts:")

        print(
            invalid[
                [
                    "year",
                    "quarter_sum",
                    "annual_value",
                    "difference",
                ]
            ].to_string(index=False)
        )

        return False

    print("[PASS] Quarterly totals reconcile with annual SEC facts.")

    return True


def validate_against_half_year_facts(
    quarterly_df: pd.DataFrame,
    revenue_facts: pd.DataFrame,
    tolerance: float = 1_000_000,
) -> bool:
    """
    Check Q1 + Q2 ≈ reported H1 revenue.
    """

    half_year = revenue_facts[
        revenue_facts["duration_type"] == "half_year"
    ].copy()

    # Economic year comes from period end.
    half_year["year"] = (
        half_year["end"].dt.year
    )

    half_year = (
        half_year
        .sort_values(
            by=[
                "year",
                "filed",
            ]
        )
        .drop_duplicates(
            subset=["year"],
            keep="first",
        )
    )

    q1_q2 = quarterly_df[
        quarterly_df["quarter"].isin(
            [
                "Q1",
                "Q2",
            ]
        )
    ].copy()

    totals = (
        q1_q2
        .groupby("year")["val"]
        .sum()
        .reset_index(
            name="quarter_sum"
        )
    )

    reported = (
        half_year[
            [
                "year",
                "val",
            ]
        ]
        .rename(
            columns={
                "val": "reported_value",
            }
        )
    )

    comparison = reported.merge(
        totals,
        on="year",
        how="inner",
    )

    comparison["difference"] = (
        comparison["quarter_sum"] - comparison["reported_value"]
    )

    comparison["abs_difference"] = (
        comparison["difference"].abs()
    )

    invalid = comparison[
        comparison["abs_difference"] > tolerance
    ]

    if not invalid.empty:

        print(
            "\n[FAIL] Q1 + Q2 does not reconcile with H1: "
        )

        print(
            invalid[
                [
                    "year",
                    "quarter_sum",
                    "reported_value",
                    "difference",
                ]
            ].to_string(index=False)
        )

        return False

    print(
        "[PASS] Q1 + Q2 reconciles with H1."
    )

    return True


def validate_against_nine_month_facts(
    quarterly_df: pd.DataFrame,
    revenue_facts: pd.DataFrame,
    tolerance: float = 1_000_000,
) -> bool:
    """
    Check whether:

        Q1 + Q2 + Q3 ≈ reported 9M revenue.
    """

    nine_month = revenue_facts[
        revenue_facts["duration_type"] == "nine_month"
    ].copy()

    nine_month["year"] = (
        nine_month["end"].dt.year
    )

    nine_month = (
        nine_month
        .sort_values(
            by=[
                "year",
                "filed",
            ]
        )
        .drop_duplicates(
            subset=["year"],
            keep="first",
        )
    )

    q1_q3 = quarterly_df[
        quarterly_df["quarter"].isin(
            [
                "Q1",
                "Q2",
                "Q3",
            ]
        )
    ].copy()

    totals = (
        q1_q3
        .groupby("year")["val"]
        .sum()
        .reset_index(
            name="quarter_sum"
        )
    )

    reported = (
        nine_month[
            [
                "year",
                "val",
            ]
        ]
        .rename(
            columns={
                "val": "reported_value",
            }
        )
    )

    comparison = reported.merge(
        totals,
        on="year",
        how="inner",
    )

    comparison["difference"] = (
        comparison["quarter_sum"] - comparison["reported_value"]
    )

    comparison["abs_difference"] = (
        comparison["difference"].abs()
    )

    invalid = comparison[
        comparison["abs_difference"] > tolerance
    ]

    if not invalid.empty:

        print(
            "\n[FAIL] Q1 + Q2 + Q3 does not reconcile with 9M: "
        )

        print(
            invalid[
                [
                    "year",
                    "quarter_sum",
                    "reported_value",
                    "difference",
                ]
            ].to_string(index=False)
        )

        return False

    print(
        "[PASS] Q1 + Q2 + Q3 reconciles with 9M."
    )

    return True


def validate_quarterly_revenue(
        quarterly_df: pd.DataFrame,
        revenue_facts: pd.DataFrame,
) -> bool:
    """
    Run all quarterly revnue validation checks.
    """

    print("\n" + "=" * 60)
    print("QUARTERLY REVENUE VALIDATION")
    print("=" * 60)

    results = [
        validate_no_duplicate_quarters(
            quarterly_df
        ),

        validate_no_missing_values(
            quarterly_df
        ),

        validate_revenue_nonnegative(
            quarterly_df
        ),

        validate_filing_after_period_end(
            quarterly_df
        ),

        validate_complete_years(
            quarterly_df
        ),

        validate_chronological_order(
            quarterly_df
        ),

        validate_source_metadata(
            quarterly_df
        ),

        validate_against_half_year_facts(
            quarterly_df,
            revenue_facts,
        ),

        validate_against_nine_month_facts(
            quarterly_df,
            revenue_facts,
        ),

        validate_against_annual_facts(
            quarterly_df,
            revenue_facts,
        ),
    ]

    print("\n" + "=" * 60)

    if all(results):
        print("VALIDATION RESULT: PASS")
        print("Dataset passed all checked.")

        return True

    print("VALIDATION RESULT: FAIL")
    print("One or more validation checks failed.")

    return False
