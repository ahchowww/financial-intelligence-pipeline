from pathlib import Path

import pandas as pd

ALIGNMENT_PATH = Path(
    "data/processed/"
    "tsla_quarterly_text_alignment.csv"
)

def main() -> None:
    # 1. Load datapath
    df = pd.read_csv(
        ALIGNMENT_PATH,
        parse_dates=[
            "quarter_end",
            "available_date",
            "report_date",
            "mda_filed",
            "text_available_date",
            "multimodal_available_date",
        ],
    )

    errors = []

    print("\n" + "=" * 60)
    print("VALIDATE QUARTERLY + MD&A ALIGNMENT")
    print("=" * 60)

    print(f"Rows: {len(df)}")

    # 2. Basic row checks
    if len(df) != 68:
        errors.append(
            f"Expected 68 rows, found {len(df)}"
        )

    if df["quarter_end"].duplicated().any():
        errors.append(
            "Duplicate quarter_end values found."
        )

    # 3. MD&A count
    has_mda_count = int(df["has_mda"].sum())

    if has_mda_count != 65:
        errors.append(
            f"Expected 65 rows with MD&A, found {has_mda_count}"
        )


    # 4. Validate report-date alignment
    with_mda = df[
        df["has_mda"]
    ].copy()

    bad_report_dates = (
        with_mda["report_date"] != with_mda["quarter_end"]
    )

    if bad_report_dates.any():
        errors.append(
            "Some MD&A report_date values do not match quarter_end."
        )

    # 5. Filing must occur on or after quarter end
    bad_filing_dates = (
        with_mda["mda_filed"] < with_mda["quarter_end"]
    )

    if bad_filing_dates.any():
        errors.append(
            "Some MD&A filings occur before their report quarter ends."
        )

    # 6. Form must match quarter
    # Tesla uses calendar fiscal quarters:
    # Q1-Q3 -> 10-Q
    # Q4 -> 10-K
    bad_10q = (
        (with_mda["quarter"] != "Q4")
        & 
        (with_mda["form"] != "10-Q")
    )

    if bad_10q.any():
        errors.append(
            "A Q1-Q3 quarter is not mapped to a 10-Q."
        )

    bad_10k = (
        (with_mda["quarter"] == "Q4")
        &
        (with_mda["form"] != "10-K")
    )

    if bad_10k.any():
        errors.append(
            "A Q4 quarter is not mapped to a 10-K."
        )

    # 7. Text availability should equal filing date
    bad_text_availability = (
        with_mda["text_available_date"] != with_mda["mda_filed"]
    )

    if bad_text_availability.any():
        errors.append(
            "text_available_date does not match mda_filed."
        )

    # 8. Validate multimodal availability date
    complete_multimodal = df[
        df["multimodal_available_date"].notna()
    ].copy()

    expected_available = (
        complete_multimodal[
            [
                "available_date",
                "text_available_date",
            ]
        ]
        .max(
            axis=1
        )
    )

    bad_multimodel_dates = (
        complete_multimodal["multimodal_available_date"] != expected_available
    )

    if bad_multimodel_dates.any():
        errors.append(
            "Incorrect multimodal_available_date found."
        )

    # 9. Incomplete financial rows must not receive multimodal availability dates
    invalid_incomplete = (
        df["available_date"].isna()
        &
        df["multimodal_available_date"].notna()
    )

    if invalid_incomplete.any():
        errors.append(
            "Incomplete financial rows have multimodal availability dates."
        )

    # 10. Clean text files must exist
    for _, row in with_mda.iterrows():
        path = Path(row["mda_path"])

        if not path.exists():
            errors.append(
                f"Missing MD&A file: {path}"
            )

    # 11. Summary
    print(
        f"Rows with MD&A: {has_mda_count}"
    )

    print(
        f"Complete multimodal rows: {len(complete_multimodal)}"
    )

    print(
        f"\nForm / Quarter mapping: "
    )

    print(
        with_mda.groupby(
            [
                "quarter",
                "form",
            ]
        )
        .size()
    )

    print("\nRows without MD&A:")
    print(
        df[
            ~df["has_mda"]
        ][
            [
                "year",
                "quarter",
                "quarter_end",
                "is_complete",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)

    if errors:
        print("VALIDATION RESULT: FAIL")
        print("=" * 60)

        for error in errors:
            print(
                f"- {error}"
            )

    else:
        print("VALIDATION RESULT: PASS")
        print("=" * 60)
        print("Quarterly financail data and MD&A are correctly aligned.")



if __name__ == "__main__":
    main()
    