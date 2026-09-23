import argparse
import json
from pathlib import Path

from src.processing.xbrl import(
    extract_concept,
    filter_financial_filings,
)

RAW_DATA_PATH = Path("data/raw/tsla_company_facts.json")

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect instant XBRL facts for a financial metric."
        )
    )

    parser.add_argument(
        "concept",
        help=(
            "US-GAAP concept, e.g. CashAndCashEquivalentsAtCarryingValue"
        ),
    )

    parser.add_argument(
        "--unit",
        default="USD",
    )

    args = parser.parse_args()

    # Load SEC company facts
    with RAW_DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    # extract concept
    df = extract_concept(
        company_facts=data,
        concept=args.concept,
        unit=args.unit,
    )

    df = filter_financial_filings(df)

    # keep instant observations
    instant = df[
        df["fact_type"] == "instant"
    ].copy()

    print("\n=== FACT TYPES ===")

    print(
        df["fact_type"].value_counts(dropna=False)
    )

    if instant.empty:
        print("\nNo instant facts found.")

        return

    print("\n=== DATA COVERAGE ===")

    print(
        "Earliest:",
        instant["end"].min(),
    )

    print(
        "Latest:",
        instant["end"].max(),
    )

    print("\n=== INSTANT FACTS ===")

    columns = [
        "end",
        "val",
        "form",
        "filed",
    ]

    optional_columns = [
        "fy",
        "fp",
        "frame",
    ]

    for column in optional_columns:
        if column in instant.columns:
            columns.append(column)

    print(
        instant[columns]
        .sort_values(
            by=[
                "end",
                "filed",
            ]
        )
        .to_string(index=False)
    )

if __name__ == "__main__":
    main()