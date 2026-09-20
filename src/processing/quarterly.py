import pandas as pd

"""
Take SEC duration facts and turn them into one clean revenue value for each quarter. 
"""

def select_direct_quarter_facts(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select direct single-quarter facts and keep the earliest filing for each period. 
    """
    quarterly = df[
        df["duration_type"] == "quarter"
    ].copy()

    quarterly = quarterly.sort_values(
        by=[
            "start",
            "end", 
            "filed",
        ]
    )

    quarterly = quarterly.drop_duplicates(
        subset=[
            "start", 
            "end",
        ],
        keep="first",
    )

    return quarterly.reset_index(drop=True)


def add_calendar_quarter(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add calendar year and quarter from period end date.
    # Tesla's fiscal year follows the calendar year
    """

    result = df.copy()

    result["year"] = (
        result["end"].dt.year
    )

    result["quarter"] = (
        "Q" 
        + result["end"]
        .dt.quarter
        .astype(str)
    )

    return result

def derive_q4_from_annual(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive Q4 using:
        Q4 = full-year value - nine-month cumulative value

    The Q4 filing date is the annual 10-K filing date.
    """

    # --------------------------------------------------
    # 1. Get annual and 9-month observations
    # --------------------------------------------------
    annual = df[
        df["duration_type"] == "annual"
    ].copy()

    nine_month = df[
        df["duration_type"] == "nine_month"
    ].copy()

    # --------------------------------------------------
    # 2. Add economic year
    # --------------------------------------------------
    annual["year"] = annual["end"].dt.year

    nine_month["year"] = (
        nine_month["end"].dt.year
    )

    # --------------------------------------------------
    # 3. Keep earliest available fact for each year
    # --------------------------------------------------
    annual = (
        annual
        .sort_values(
            by=["year", "filed"]
        )
        .drop_duplicates(
            subset=["year"],
            keep="first",
        )
    )

    nine_month = (
        nine_month
        .sort_values(
            by=["year", "filed"]
        )
        .drop_duplicates(
            subset=["year"],
            keep="first",
        )
    )

    # --------------------------------------------------
    # 4. Match FY and 9M by year
    # --------------------------------------------------
    merged = annual.merge(
        nine_month,
        on="year",
        how="inner",
        suffixes=("_fy", "_9m"),
    )

    print("\n=== DEBUG: FY / 9M matches ===")

    if merged.empty:
        print("No matching FY and 9M years.")
        return pd.DataFrame()

    print(
        merged[
            [
                "year",
                "val_fy",
                "val_9m",
                "filed_fy",
                "filed_9m",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # 5. Calculate Q4
    # --------------------------------------------------
    merged["val"] = (
        merged["val_fy"] - merged["val_9m"]
    )

    merged["quarter"] = "Q4"

    merged["start"] = (
        merged["end_9m"] + pd.Timedelta(days=1)
    )

    merged["end"] = merged["end_fy"]

    # Q4 becomes known when the 10-K is filed.
    merged["filed"] = merged["filed_fy"]

    merged["form"] = merged["form_fy"]

    merged["derived"] = True

    merged["source_method"] = (
        "fy_minus_9m"
    )

    # --------------------------------------------------
    # 6. Return clean Q4 rows
    # --------------------------------------------------
    result = merged[
        [
            "year",
            "quarter",
            "start",
            "end",
            "val",
            "filed",
            "form",
            "derived",
            "source_method",
        ]
    ].copy()

    return result


def build_quarterly_series(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a complete quarterly series from SEC duration facts.

    Q1-Q3:
        Prefer directly reported single-quarter facts
    
    Q4:
        Derive from annual - nine_month cummulative value 
    """

    # 1. Direct quarterly facts
    direct = select_direct_quarter_facts(df)

    direct = add_calendar_quarter(direct)

    direct["derived"] = False
    direct["source_method"] = "direct"

    direct = direct[
        [
            "year",
            "quarter",
            "start",
            "end",
            "val",
            "filed",
            "form",
            "derived",
            "source_method",
        ]
    ]

    # 2. Potential derived Q4 facts
    derived_q4 = derive_q4_from_annual(df)

    print("\n=== DEBUG: Derived Q4 ===")

    if derived_q4.empty:
        print("No derived Q4 rows were created.")
    else:
        print(
            derived_q4[
                [
                    "year",
                    "quarter",
                    "val",
                    "filed",
                    "derived",
                    "source_method",
                ]
            ].to_string(index=False)
        )

    # 3. Only use derived Q4 when direct Q4 does not already exist
    direct_q4_years = set(
        direct.loc[
            direct["quarter"] == "Q4",
            "year",
        ]
    )

    if not derived_q4.empty:
        derived_q4 = derived_q4[
            ~derived_q4["year"].isin(
                direct_q4_years
            )
        ]

    # 4. Combine 
    result = pd.concat(
        [
            direct,
            derived_q4,
        ],
        ignore_index=True,
    )

    quarter_order ={
        "Q1": 1,
        "Q2": 2,
        "Q3": 3,
        "Q4": 4,
    }

    result["quarter_number"] = (
        result["quarter"]
        .map(quarter_order)
    )

    result = (
        result
        .sort_values(
            by=[
                "year", 
                "quarter_number",
            ]
        )
        .reset_index(drop=True)
    )

    return result