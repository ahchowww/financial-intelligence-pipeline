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
            "Inspect reporting periods for an XBRL concept."
        )
    )

    parser.add_argument(
        "concept",
        help=(
            "US-GAAP concept name, e.g. GrossProfit"
        ),
    )

    parser.add_argument(
        "--unit",
        default="USD",
    )

    args = parser.parse_args()

    with RAW_DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    df = extract_concept(
        company_facts=data,
        concept=args.concept,
        unit=args.unit,
    )

    df = filter_financial_filings(df)

    print("\n=== DURATION TYPES ===")

    print(
        df["duration_type"].value_counts()
    )

    print(
        "\n=== DATE COVERAGE ==="
    )

    print(
        "Earliest end date:", df["end"].min(),
    )

    print(
        "Latest end date:", df["end"].max(),
    )

    columns = [
        "start",
        "end",
        "val",
        "form",
        "filed",
        "period_days",
        "duration_type",
    ]

    print(
        "\n=== SAMPLE FACTS ==="
    )

    print(
        df[
            columns
        ]
        .sort_values(
            [
                "end",
                "filed",
            ]
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()