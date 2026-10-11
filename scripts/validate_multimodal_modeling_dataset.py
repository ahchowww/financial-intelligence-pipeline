from pathlib import Path

import pandas as pd


MULTIMODAL_PATH = Path(
    "data/processed/"
    "tsla_multimodal_modeling.csv"
)

FINANCIAL_MODELING_PATH = Path(
    "data/processed/"
    "tsla_quarterly_modeling.csv"
)


def main() -> None:
    # 1. Load datasets
    multimodal = pd.read_csv(
        MULTIMODAL_PATH,
        parse_dates=[
            "quarter_end",
            "available_date",
            "mda_filed",
            "text_available_date",
            "multimodal_available_date",
        ],
    )

    financial = pd.read_csv(
        FINANCIAL_MODELING_PATH,
        parse_dates=[
            "quarter_end",
            "available_date",
        ],
    )

    errors = []

    print("\n" + "=" * 60)
    print("VALIDATE MULTIMODAL MODELING DATASET")
    print("=" * 60)

    # 2. Row count
    print(
        f"Multimodal rows: {len(multimodal)}"
    )

    print(
        f"Financial modeling rows: {len(financial)}"
    )

    if len(multimodal) != 61:
        errors.append(
            f"Expected 61 multimodal rows, found {len(multimodal)}."
        )

    # 3. Exact quarter alignment with numerical dataset
    multimodal_keys = (
        multimodal[
            [
                "year",
                "quarter",
                "quarter_end",
            ]
        ]
        .reset_index(
            drop=True
        )
    )

    financial_keys = (
        financial[
            [
                "year",
                "quarter",
                "quarter_end",
            ]
        ]
        .reset_index(
            drop=True
        )
    )

    if not multimodal_keys.equals(financial_keys):
        errors.append(
            "Multimodal quarters do not exactly match the financial modeling dataset."
        )

    
    # 4. No duplicate quarters
    if multimodal["quarter_end"].duplicated().any():
        errors.append(
            "Duplicate quarter_end values found."
        )


    # 5. Check quarterly continuity
    quarter_index = (
        multimodal["year"].astype(int) * 4 + multimodal["quarter_number"].astype(int)
    )

    differences = quarter_index.diff()

    if not (differences.dropna() == 1).all():
        errors.append(
            "Multimodal dataset is not quarterly continuous."
        )

    
    # 6. Every row must contain MD&A
    if multimodal["mda_path"].isna().any():
        errors.append(
            "Some multimodal rows have no MD&A path."
        )


    # 7. Every MD&A file must exist
    for _, row in multimodal.iterrows():
        path = Path(row["mda_path"])

        if not path.exists():
            errors.append(
                f"Missing MD&A file: {path}"
            )

    
    # 8. MD&A must not be suspiciously short
    if (multimodal["mda_chars"] < 10_000).any():
        errors.append(
            "At least one MD&A is suspiciously short."
        )

    
    # 9. Filing must occur after quarter end
    if (
        multimodal["mda_filed"] < multimodal["quarter_end"]
    ).any():
        errors.append(
            "At least one MD&A was filed before quarter end."
        )

    
    # 10. Multimodal availability must equal
    #     max(financial availability, text availability)
    expected_available = (
        multimodal[
            [
                "available_date",
                "text_available_date",
            ]
        ]
        .max(
            axis=1
        )
    )

    if not (
        multimodal["multimodal_available_date"] == expected_available
    ).all():
        errors.append(
            "Incorrect multimodal availability date found."
        )

    
    # 11. Availability must occur after quarter end
    if (
        multimodal["multimodal_available_date"] < multimodal["quarter_end"]
    ).any():
        errors.append(
            "A multimodal observation becomes available before quarter end."
        )

    
    # 12. Form / quarter relationship
    bad_10q = (
        (multimodal["quarter"] != "Q4")
        &
        (multimodal["form"] != "10-Q")
    )

    bad_10k = (
        (multimodal["quarter"] == "Q4")
        &
        (multimodal["form"] != "10-K")
    )

    if bad_10q.any():
        errors.append(
            "A Q1-Q3 observation is not mapped to a 10-Q."
        )

    if bad_10k.any():
        errors.append(
            "A Q4 observation is not mapped to a 10-K."
        )

    
    # 13. Summary
    print("\nCoverage:")

    print(
        f"{multimodal.iloc[0]['year']}"
        f"{multimodal.iloc[0]['quarter']}"
        f" -> "
        f"{multimodal.iloc[-1]['year']}"
        f"{multimodal.iloc[-1]['quarter']}"
    )

    print("\nForm counts:")

    print(
        multimodal["form"].value_counts()
    )

    print(
        "\nMD&A character statistics:"
    )

    print(
        multimodal["mda_chars"].describe()
    )

    print(
        "\nMultimodal availability lag "
        "(days after quarter end):"
    )

    lag_days = (
        multimodal["multimodal_available_date"] - multimodal["quarter_end"]
    ).dt.days

    print(
        lag_days.describe()
    )

    
    # 14. Final result
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

        print(
            "The multimodal modeling dataset is continuous, point-in-time safe, "
            "and exactly aligned with the financial modeling dataset."
        )


if __name__ == "__main__":
    main()