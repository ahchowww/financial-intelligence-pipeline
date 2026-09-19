import json
from pathlib import Path

import pandas as pd

from src.processing.xbrl import (
    extract_concept,
    filter_financial_filings,
)


RAW_DATA_PATH = Path(
    "data/raw/tsla_company_facts.json"
)

CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
]


def prepare_concept(
    data: dict,
    concept: str,
) -> pd.DataFrame:

    df = extract_concept(
        company_facts=data,
        concept=concept,
    )

    df = filter_financial_filings(df)

    # Focus on recent years where both tags may overlap.
    df = df[
        df["end"].dt.year >= 2021
    ].copy()

    columns = [
        "start",
        "end",
        "val",
        "form",
        "filed",
        "fy",
        "fp",
        "period_days",
    ]

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    return df[available_columns]


def main() -> None:

    with RAW_DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    for concept in CONCEPTS:

        print("\n")
        print("=" * 80)
        print(f"Concept: {concept}")
        print("=" * 80)

        df = prepare_concept(
            data,
            concept,
        )

        print(
            df.tail(30)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()