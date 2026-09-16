import json
from pathlib import Path

from src.processing.xbrl import (
    extract_concept,
    filter_financial_filings,
)

RAW_DATA_PATH = Path(
    "data/raw/tsla_company_facts.json"
)

OUTPUT_PATH = Path(
    "data/processed/tsla_revenue.json"
)

REVENUE_CONCEPT = (
    "RevenueFromContractWithCustomerExcludingAssessedTax"
)

def main() -> None:
    with RAW_DATA_PATH.open(
        "r",
        encoding="utf-8",

    ) as file:
        data = json.load(file)

    revenue = extract_concept(
        company_facts=data,
        concept=REVENUE_CONCEPT,
    )

    revenue = filter_financial_filings(revenue)

    print(revenue.head())

    print("\nColumns:")
    print(revenue.columns.tolist())

    print("\nNumber of rows:")
    print(len(revenue))

    print("\nForms:")
    print(revenue["form"].value_counts())


    # inspect useful SEC fields
    useful_columns = [
        "start",
        "end",
        "val",
        "form",
        "filed",
        "fy",
        "fp",
        "frame",
        "period_days",
    ]

    # Some SEC facts may not contain every column,
    # so only keep columns that actually exist.
    available_columns = [
        column
        for column in useful_columns
        if column in revenue.columns
    ]

    print("\n=== Latest 30 Revenue Observations ===")

    print(
        revenue[available_columns]
        .tail(30)
        .to_string(index=False)
    )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    revenue.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved revenue data to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()