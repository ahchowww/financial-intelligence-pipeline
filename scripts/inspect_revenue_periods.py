import json
from pathlib import Path

from src.processing.xbrl import(
    extract_concept,
    filter_financial_filings,
)

RAW_DATA_PATH = Path(
    "data/raw/tsla_company_facts.json"
)

REVENUE_CONCEPT = (
    "RevenueFromContractWithCustomerExcludingAssessedTax"
)

def main() -> None:
    with RAW_DATA_PATH.open(
        "r",
        encoding="UTF-8",
    ) as file:
        data = json.load(file)

    revenue = extract_concept(
        company_facts=data,
        concept=REVENUE_CONCEPT,
    )

    revenue = filter_financial_filings(revenue)

    columns = [
        "start",
        "end",
        "val",
        "form",
        "filed",
        "fy",
        "fp",
        "period_days",
        "duration_type",
    ]

    available_columns = [
        column
        for column in columns
        if column in revenue.columns
    ]

    print(
        revenue[available_columns]
        .sort_values(
            by=[
                "end",
                "period_days",
                "filed",
            ]
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
