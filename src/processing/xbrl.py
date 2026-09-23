from typing import Any

import pandas as pd


def extract_concept(
        company_facts: dict[str, Any],
        concept: str,
        unit: str = "USD",
) -> pd.DataFrame:
    """
    1. Find the requested concept and unit.
    2. Converts SEC observations into DataFrame.
    3. Convert date columns to datetime.
    4. Classifies each observation as duration or instant.
    5. Calculates reporting-period length for duration facts.
    6. Classifies duration facts as quarter / H1 / 9M / annual.
    """

    # 1. Extract US-GAAP concept 
    us_gaap = company_facts["facts"]["us-gaap"]

    if concept not in us_gaap:
        raise ValueError(
            f"Concept '{concept}' was not found."
        )

    concept_data = us_gaap[concept]

    units = concept_data["units"]

    if unit not in units:
        raise ValueError(
            f"Unit '{unit}' was not found for '{concept}'."
        )

    # 2. Converts SEC observations into DataFrame
    observations = units[unit]

    df = pd.DataFrame(observations)

    # 3. Convert date columns
    if "start" in df.columns:
        df["start"] = pd.to_datetime(df["start"])

    if "end" in df.columns:
        df["end"] = pd.to_datetime(df["end"])

    if "filed" in df.columns:
        df["filed"] = pd.to_datetime(df["filed"])

    # 4. classify XBRL fact type
    #    -> Duration: start + end
    #    -> Instant: end only
    df["fact_type"] = df.apply(
            classify_fact_type,
            axis=1,
        )

    # 5.Create duration-related columns for all rows
    df["period_days"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64",
    )

    df["duration_type"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="object",
    )

    # 6. Calculate period length only for duration facts
    duration_mask = (
        df["fact_type"] == "duration"
    )

    if duration_mask.any():
        period_days = (
            df.loc[
                duration_mask,
                "end",
            ]
            -
            df.loc[
                duration_mask,
                "start",
            ]
        ).dt.days + 1

        df.loc[
            duration_mask,
            "period_days",
        ] = period_days

        df.loc[
            duration_mask,
            "duration_type",
        ] = period_days.apply(
            classify_duration
        )

    return df


def classify_duration(
        period_days: int,
) -> str:
    """
    Classify a duration fact by approximate reporting period.
    """

    if 75 <= period_days <= 105:
        return "quarter"

    if 150 <= period_days <= 200:
        return "half_year"

    if 240 <= period_days <= 300:
        return "nine_month"

    if 330 <= period_days <= 380:
        return "annual"

    return "other"

    
def filter_financial_filings(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep common SEC periodic financial filings.
    """

    allowed_forms = [
        "10-Q",
        "10-K",
        "10-Q/A",
        "10-K/A",
    ]

    return (
        df[df["form"].isin(allowed_forms)]
        .copy()
        .sort_values(
            by=["end", "filed"]
        )
        .reset_index(drop=True)
    )


def classify_fact_type(
        row: pd.Series,
) -> str:
    """
    Classify an XBRL fact as duration or instant.
    -> Duration facts have both start and end dates.
    -> Instant facts have only an end date.
    """

    has_start = (
        "start" in row.index and pd.notna(row.get("start"))
    )

    has_end = (
        "end" in row.index and pd.notna(row.get("end"))
    )

    if has_start and has_end:
        return "duration"

    if has_end:
        return "instant"

    return "unknown"