import json
from pathlib import Path

from src.processing.metrics import(
    NORMALIZED_METRICS,
)

from src.processing.quarterly import(
    build_quarterly_series,
)

from src.processing.xbrl import(
    extract_concept,
    filter_financial_filings,
)

from src.validation.financial import(
    validate_quarterly_revenue,
)

RAW_DATA_PATH = Path("data/raw/tsla_company_facts.json")

REVENUE_CONFIG = (
    NORMALIZED_METRICS["revenue"]
)

def main() -> None:
    with RAW_DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    # extract revenue
    revenue = extract_concept(
        company_facts=data,
        concept=REVENUE_CONFIG["primary_concept"],
        unit= REVENUE_CONFIG["unit"],
    )

    revenue = filter_financial_filings(revenue)

    # build quarterly dataset
    quarterly = build_quarterly_series(revenue)

    # validate
    validate_quarterly_revenue(
        quarterly_df=quarterly,
        revenue_facts=revenue,
    )


if __name__ == "__main__":
    main()