import pandas as pd

"""
Flow:
1. Instant SEC facts.
2. Keep fact_type == "instant".
3. Derive year from end date.
4. Derive Q1/Q2/Q3/Q4 from end date.
5. Resolve repeated filings.
6. Keep earliest filing.
7. Quarter-end balance dataset.
"""

QUARTER_ORDER = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "Q4": 4,
}

INSTANT_COLUMNS = [
    "year",
    "quarter",
    "end", 
    "val", 
    "filed", 
    "form", 
    "derived", 
    "source_method",  
]


def select_instant_facts(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only instant XBRL observations.
    """

    return df[
        df["fact_type"] == "instant"
    ].copy()


def add_calendar_quarter(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive economic year and quarter from balance date.
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


def select_earliest_balance(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    When the same balance date appears in multiple filings, keep the earliest filing.
    """

    result = (
        df
        .sort_values(
            by=[
                "end",
                "filed",
            ]
        )
        .drop_duplicates(
            subset=[
                "end",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return result


def build_quarterly_instant_series(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one quarterly balance for each available instant reporting date.
    """

    # 1. Keep instant observations
    instant = select_instant_facts(df)

    # 2. Keep earliest filing per economic balance date
    instant = select_earliest_balance(instant)

    # 3. Add year and quarter
    instant = add_calendar_quarter(instant)

    # 4. Instant facts are always direct observations
    instant["derived"] = False
    instant["source_method"] = "direct"

    # 5. Add numeric quarter order
    instant["quarter_number"] = (
        instant["quarter"]
        .map(QUARTER_ORDER)
    )

    # 6. Keep clean output columns
    result = instant[
        INSTANT_COLUMNS + ["quarter_number"]
    ].copy()

    # 7. Chronological ordering
    result = (
        result
        .sort_values(
            by=[
                "year",
                "quarter_number"
            ]
        )
        .reset_index(drop=True)
    )

    return result

