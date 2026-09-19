import json
from pathlib import Path

from src.processing.xbrl import(
    extract_concept,
    filter_financial_filings,
)

from src.processing.quarterly import(
    build_quarterly_series,
)

from src.processing.metrics import NORMALIZED_METRICS


RAW_DATA_PATH = Path(
    "data/raw/tsla_company_facts.json"
)

OUTPUT_PATH = Path(
    "data/processed/tsla_quarterly_revenue.csv"
)

REVENUE_CONFIG = NORMALIZED_METRICS["revenue"]


def main() -> None:
    with RAW_DATA_PATH.open(
        "r", 
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    revenue = extract_concept(
        company_facts=data,
        concept=REVENUE_CONFIG["primary_concept"],
        unit=REVENUE_CONFIG["unit"],
    )

    revenue = filter_financial_filings(revenue)

    quarterly = build_quarterly_series(revenue)

    columns = [
        "year",
        "quarter",
        "start",
        "end",
        "val",
        "filed",
        "form",
        "derived",
        "source_method",
    ]

    print(
        quarterly[columns]
        .to_string(index=False)
    )

    # Save the result
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    quarterly.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved quarterly revenue to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()