import pandas as pd


"""
Utilities for converting SEC duration facts into a clean quarterly series.

Designed for duration-based financial metrics such as:

- Revenue
- Gross Profit
- Operating Income
- Net Income

Direct quarterly facts are preferred whenever available.

If a direct quarter is missing, cumulative SEC facts can be used
to reconstruct it:

Q1 = H1 - Q2
Q2 = H1 - Q1
Q3 = 9M - H1
Q4 = FY - 9M

Note:
The current year/quarter assignment assumes Tesla's fiscal year
matches the calendar year.
"""


# ============================================================
# Constants
# ============================================================

QUARTER_ORDER = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "Q4": 4,
}


QUARTERLY_COLUMNS = [
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


# ============================================================
# Direct quarter selection
# ============================================================

def select_direct_quarter_facts(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select direct single-quarter SEC facts.

    If the same economic quarter appears in multiple filings,
    keep the earliest filing.
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

    return quarterly.reset_index(
        drop=True
    )


# ============================================================
# Calendar quarter assignment
# ============================================================

def add_calendar_quarter(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add economic year and calendar quarter based on period end.

    Tesla currently uses a calendar fiscal year, so:
        March 31     -> Q1
        June 30      -> Q2
        September 30 -> Q3
        December 31  -> Q4
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


# ============================================================
# Cumulative fact selection
# ============================================================

def select_earliest_cumulative_fact(
    df: pd.DataFrame,
    duration_type: str,
) -> pd.DataFrame:
    """
    Select the earliest available cumulative fact for each economic year.

    Examples of duration_type:
        half_year
        nine_month
        annual
    """

    result = df[
        df["duration_type"] == duration_type
    ].copy()

    if result.empty:
        return result

    result["year"] = (
        result["end"].dt.year
    )

    result = (
        result
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
        .reset_index(drop=True)
    )

    return result


# ============================================================
# Missing-quarter reconstruction
# ============================================================

def derive_missing_quarters(
    df: pd.DataFrame,
    direct: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct missing quarterly duration facts.

    Fallback rules:
        Q1 = H1 - Q2
        Q2 = H1 - Q1
        Q3 = 9M - H1
        Q4 = FY - 9M

    Direct quarterly facts always take priority.

    The filing date of a derived quarter is the latest filing
    date among the facts required to calculate that quarter.
    This reflects when the derived value first becomes knowable.
    """

    # --------------------------------------------------------
    # 1. Select cumulative facts
    # --------------------------------------------------------

    half_year = select_earliest_cumulative_fact(
        df,
        "half_year",
    )

    nine_month = select_earliest_cumulative_fact(
        df,
        "nine_month",
    )

    annual = select_earliest_cumulative_fact(
        df,
        "annual",
    )

    # --------------------------------------------------------
    # 2. Build year -> cumulative fact lookup tables
    # --------------------------------------------------------

    h1_by_year = {
        int(row["year"]): row
        for _, row in half_year.iterrows()
    }

    nine_month_by_year = {
        int(row["year"]): row
        for _, row in nine_month.iterrows()
    }

    annual_by_year = {
        int(row["year"]): row
        for _, row in annual.iterrows()
    }

    # --------------------------------------------------------
    # 3. Build (year, quarter) -> direct fact lookup
    # --------------------------------------------------------

    direct_by_quarter = {}

    for _, row in direct.iterrows():

        key = (
            int(row["year"]),
            row["quarter"],
        )

        direct_by_quarter[key] = row

    # --------------------------------------------------------
    # 4. Find every economic year contained in source facts
    # --------------------------------------------------------

    years = sorted(
        df["end"]
        .dropna()
        .dt.year
        .astype(int)
        .unique()
    )

    derived_rows = []

    # --------------------------------------------------------
    # 5. Reconstruct missing quarters year by year
    # --------------------------------------------------------

    for year in years:

        # Direct quarterly observations
        q1 = direct_by_quarter.get(
            (year, "Q1")
        )

        q2 = direct_by_quarter.get(
            (year, "Q2")
        )

        q3 = direct_by_quarter.get(
            (year, "Q3")
        )

        q4 = direct_by_quarter.get(
            (year, "Q4")
        )

        # Cumulative observations
        h1 = h1_by_year.get(
            year
        )

        nine_month_fact = (
            nine_month_by_year.get(
                year
            )
        )

        annual_fact = (
            annual_by_year.get(
                year
            )
        )

        # ====================================================
        # Q1 fallback
        #
        # Q1 = H1 - Q2
        # ====================================================

        if (
            q1 is None
            and q2 is not None
            and h1 is not None
        ):

            filed = max(
                h1["filed"],
                q2["filed"],
            )

            derived_rows.append(
                {
                    "year": year,
                    "quarter": "Q1",
                    "start": h1["start"],
                    "end": (
                        q2["start"] - pd.Timedelta(days=1)
                    ),
                    "val": (
                        h1["val"] - q2["val"]
                    ),
                    "filed": filed,
                    "form": h1["form"],
                    "derived": True,
                    "source_method": ("h1_minus_q2"),
                }
            )

        # ====================================================
        # Q2 fallback
        #
        # Q2 = H1 - Q1
        # ====================================================

        if (
            q2 is None
            and q1 is not None
            and h1 is not None
        ):

            filed = max(
                h1["filed"],
                q1["filed"],
            )

            derived_rows.append(
                {
                    "year": year,
                    "quarter": "Q2",
                    "start": (
                        q1["end"] + pd.Timedelta(days=1)
                    ),
                    "end": h1["end"],
                    "val": (
                        h1["val"] - q1["val"]
                    ),
                    "filed": filed,
                    "form": h1["form"],
                    "derived": True,
                    "source_method": ("h1_minus_q1"),
                }
            )

        # ====================================================
        # Q3 fallback
        #
        # Q3 = 9M - H1
        # ====================================================

        if (
            q3 is None
            and h1 is not None
            and nine_month_fact is not None
        ):

            filed = max(
                h1["filed"],
                nine_month_fact["filed"],
            )

            derived_rows.append(
                {
                    "year": year,
                    "quarter": "Q3",
                    "start": (
                        h1["end"] + pd.Timedelta(days=1)
                    ),
                    "end": (nine_month_fact["end"]),
                    "val": (
                        nine_month_fact["val"] - h1["val"]
                    ),
                    "filed": filed,
                    "form": (
                        nine_month_fact["form"]
                    ),
                    "derived": True,
                    "source_method": ("nine_month_minus_h1"),
                }
            )

        # ====================================================
        # Q4 fallback
        #
        # Q4 = FY - 9M
        # ====================================================

        if (
            q4 is None
            and nine_month_fact is not None
            and annual_fact is not None
        ):

            filed = max(
                nine_month_fact["filed"],
                annual_fact["filed"],
            )

            derived_rows.append(
                {
                    "year": year,
                    "quarter": "Q4",
                    "start": (
                        nine_month_fact["end"] + pd.Timedelta(days=1)
                    ),
                    "end": (
                        annual_fact["end"]
                    ),
                    "val": (
                        annual_fact["val"] - nine_month_fact["val"]
                    ),
                    "filed": filed,
                    "form": (
                        annual_fact["form"]
                    ),
                    "derived": True,
                    "source_method": ("fy_minus_9m"),
                }
            )

    # --------------------------------------------------------
    # IMPORTANT:
    # Return only AFTER every year has been processed.
    # --------------------------------------------------------

    if not derived_rows:

        return pd.DataFrame(
            columns=QUARTERLY_COLUMNS
        )

    return pd.DataFrame(
        derived_rows,
        columns=QUARTERLY_COLUMNS,
    )


# ============================================================
# Main quarterly builder
# ============================================================

def build_quarterly_series(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the best available quarterly series from SEC duration facts.

    Priority
        1. Direct quarterly facts
        2. Derived quarterly facts when direct facts
           are unavailable

    Direct SEC observations always take priority over reconstructed observations.
    """

    # --------------------------------------------------------
    # 1. Select direct quarterly facts
    # --------------------------------------------------------

    direct = select_direct_quarter_facts(
        df
    )

    direct = add_calendar_quarter(
        direct
    )

    direct["derived"] = False

    direct["source_method"] = (
        "direct"
    )

    direct = direct[
        QUARTERLY_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # 2. Reconstruct missing quarters
    # --------------------------------------------------------

    derived = derive_missing_quarters(
        df=df,
        direct=direct,
    )

    # --------------------------------------------------------
    # 3. Combine direct and derived observations
    # --------------------------------------------------------

    result = pd.concat(
        [
            direct,
            derived,
        ],
        ignore_index=True,
    )

    # Nothing to process.
    if result.empty:
        result["quarter_number"] = (
            pd.Series(dtype="Int64")
        )

        return result

    # --------------------------------------------------------
    # 4. Add quarter number
    # --------------------------------------------------------

    result["quarter_number"] = (
        result["quarter"]
        .map(QUARTER_ORDER)
    )

    # --------------------------------------------------------
    # 5. Direct observations receive higher priority
    # --------------------------------------------------------

    result["source_priority"] = (
        result["derived"]
        .map(
            {
                False: 0,
                True: 1,
            }
        )
    )

    # --------------------------------------------------------
    # 6. Remove duplicate year-quarter observations
    #
    # Direct facts win over derived facts.
    # --------------------------------------------------------

    result = (
        result
        .sort_values(
            by=[
                "year",
                "quarter_number",
                "source_priority",
            ]
        )
        .drop_duplicates(
            subset=[
                "year",
                "quarter",
            ],
            keep="first",
        )
    )

    # --------------------------------------------------------
    # 7. Final chronological ordering
    # --------------------------------------------------------

    result = (
        result
        .sort_values(
            by=[
                "year",
                "quarter_number",
            ]
        )
        .drop(
            columns=[
                "source_priority",
            ]
        )
        .reset_index(drop=True)
    )

    return result