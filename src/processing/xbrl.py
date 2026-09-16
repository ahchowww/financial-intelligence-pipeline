from typing import Any

import pandas as pd


def extract_concept(
        company_facts: dict[str, Any],
        concept: str,
        unit: str = "USD",
) -> pd.DataFrame:
    """
    Extract observations for one US-GAAP concept
    from SEC Company Facts data
    """

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

    observations = units[unit]

    df = pd.DataFrame(observations)

    if "start" in df.columns:
        df["start"] = pd.to_datetime(df["start"])

    if "end" in df.columns:
        df["end"] = pd.to_datetime(df["end"])

    if "filed" in df.columns:
        df["filed"] = pd.to_datetime(df["filed"])
    
    if "start" in df.columns and "end" in df.columns:
        df["period_days"] =(
            df["end"] - df["start"]
        ).dt.days + 1
    
    return df


def filter_financial_filings(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep common SEC periodic financial filings
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