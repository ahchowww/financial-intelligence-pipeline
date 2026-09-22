import argparse
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
    validate_quarterly_duration_metric,
)

RAW_DATA_PATH = Path("data/raw/tsla_company_facts.json")

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a quarterly financial metric."
        )
    )

    parser.add_argument(
        "metric",
        choices=sorted(
            NORMALIZED_METRICS.keys()
        ),
    )

    args = parser.parse_args()

    metric_name = args.metric

    config = NORMALIZED_METRICS[metric_name]

    if config["fact_type"] != "duration":
        raise ValueError(
            f"{metric_name} is not a duration metric."
        )

    with RAW_DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    facts = extract_concept(
        company_facts=data,
        concept=config["primary_concept"],
        unit=config["unit"],
    )

    facts = filter_financial_filings(facts)

    quarterly = build_quarterly_series(facts)

    validate_quarterly_duration_metric(
        quarterly_df=quarterly,
        source_facts=facts,
        metric_name=config["display_name"],
        require_nonnegative=config["nonnegative_expected"],
    )


if __name__ == "__main__":
    main()