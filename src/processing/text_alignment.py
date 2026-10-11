import pandas as pd

"""
Quarterly financial data
        +
Clean MD&A metadata
        ↓
match by reporting quarter
        ↓
Quarter + financial data + MD&A
        ↓
calculate when BOTH were available
"""

def build_quarterly_text_alignment(
        quarterly: pd.DataFrame,
        mda_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align each Tesla financial quarter with the MD&A filing that 
    discusses that same quarter.

    Matching rule:
        quarterly.quarter_end == mda_metadata.report_date
    
    Point-in-time rule:
        multimodal_available_date = max(financial available_date, MD&A filed date)
    
    If the financial row is incomplete and therefore has no available_date,
    multimodal_available_date remains NAT.
    """

    quarterly = quarterly.copy()
    mda_metadata = mda_metadata.copy()

    # 1. Parse dates
    quarterly["quarter_end"] = pd.to_datetime(
        quarterly["quarter_end"]
    )

    quarterly["available_date"] = pd.to_datetime(
        quarterly["available_date"]
    )

    mda_metadata["report_date"] = pd.to_datetime(
        mda_metadata["report_date"]
    )

    mda_metadata["filed"] = pd.to_datetime(
        mda_metadata["filed"]
    )

    # 2. Check uniqueness before merging
    if quarterly["quarter_end"].duplicated().any():
        raise ValueError(
            "Duplicated quarter_end values found in quarterly dataset."
        )

    if mda_metadata["report_date"].duplicated().any():
        raise ValueError(
            "Duplicated report_date values found in MD&A metadata."
        )

    # 3. Keep MD&A columns needed downstream
    mda = (
        mda_metadata[
            [
                "accession_number",
                "form",
                "report_date",
                "filed",
                "clean_mda_path",
                "clean_mda_chars",
                "clean_mda_lines",
            ]
        ]
        .rename(
            columns={
                "filed": "mda_filed",
                "clean_mda_path": "mda_path",
                "clean_mda_chars": "mda_chars",
                "clean_mda_lines": "mda_lines",
            }
        )
    )

    # 4. Align financial quarter with MD&A
    # Left merge preserves the whole quarterly master.
    # The earliest master rows may have no MD&A.
    result = quarterly.merge(
        mda,
        how="left",
        left_on="quarter_end",
        right_on="report_date",
        validate="one_to_one",
    )

    # 5. Text availability
    result["text_available_date"] = result["mda_filed"]

    # 6. Multimodal availability
    # Both the numerical data and MD&A must exist.
    # Do not allow skipna=True here because an incomplete financial row
    # is not a valid multimodal observation.
    result["multimodal_available_date"] = pd.NaT

    valid_mask = (
        result["available_date"].notna()
        & 
        result["text_available_date"].notna()
    )

    result.loc[
        valid_mask, 
        "multimodal_available_date"
    ] = (
        result.loc[
            valid_mask,
            [
                "available_date",
                "text_available_date",
            ],
        ]
        .max(
            axis=1
        )
    )

    # 7. Whether this quarter has usable MD&A
    result["has_mda"] = result["mda_path"].notna()

    return result


def build_multimodal_modeling_dataset(
        aligned: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the continuous point-in-time multimodal modeling dataset.

    A valid row must have:
        1. Complete financial data
        2. An MD&A data
        3. A valid multimodal availability date
    
    We keep the continuous suffix after the final invalid row so that
    isolated early observations are not used in time-series modeling.
    """

    result = aligned.copy()

    # 1. Parse important dates
    result["quarter_end"] = pd.to_datetime(
        result["quarter_end"]
    )

    result["multimodal_available_date"] = pd.to_datetime(
        result["multimodal_available_date"]
    )

    # 2. Define usable multimodal observations
    valid_mask = (
        result["is_complete"].fillna(False)
        &
        result["has_mda"].fillna(False)
        & 
        result["multimodal_available_date"].notna()
    )

    # 3. Start after the final invalid row
    # example:
    # 2010 Q4 -> valid
    # 2011 Q1 -> invalid
    # 2011 Q2 -> valid
    # ...

    # We therefore start from 2011 Q2
    invalid_indices = (
        result.index[~valid_mask]
    )

    if len(invalid_indices) == 0:
        modeling = result.copy()
    else:
        start_index = invalid_indices.max() + 1
        modeling = result.loc[start_index:].copy()


    # 4. Safety check
    # Every row in the final suffix must be valid
    final_valid_mask = (
        modeling["is_complete"].fillna(False)
        &
        modeling["has_mda"].fillna(False)
        &
        modeling["multimodal_available_date"].notna()
    )

    if not final_valid_mask.all():
        raise ValueError(
            "Final multimodal modeling dataset still contains invalid rows."
        )

    # 5. Check calendar-quarter continuity
    quarter_index = (
        modeling["year"].astype(int) * 4 + modeling["quarter_number"].astype(int)
    )

    quarter_differences = quarter_index.diff()

    if not (quarter_differences.dropna() == 1).all():
        raise ValueError(
            "Multimodal modeling dataset is not quarterly continuous."
        )

    # 6. Sort and reset index
    modeling = (
        modeling
        .sort_values(
            "quarter_end"
        )
        .reset_index(
            drop=True
        )
    )

    return modeling
