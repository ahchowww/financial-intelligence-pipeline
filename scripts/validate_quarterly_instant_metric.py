import argparse
import json
from pathlib import Path

from src.processing.instant import(
    build_quarterly_instant_series,
)

from src.processing.metrics import(
    NORMALIZED_METRICS,
)

from src.processing.xbrl import(
    extract_concept,
    filter_financial_filings,
)

from src.validation.instant import(
    validate_quarterly_instant_metric,
)

RAW_PATH_DATA = Path("data/raw/tsla_company_facts.json")

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a quarterly instant financial metric."
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

    if config["fact_type"] != "instant":
        raise ValueError(
            f"{metric_name} is not an instant metric."
        )

    # load SEC company facts
    with RAW_PATH_DATA.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    # extract source facts
    facts = extract_concept(
        company_facts=data,
        concept=config["primary_concept"],
        unit=config["unit"],
    )

    facts = filter_financial_filings(facts)

    # build quarterly balances
    quarterly = (
        build_quarterly_instant_series(facts)
    )


    # validate
    print(
        "\n"
        + "=" * 60
    )

    print(
        f"QUARTERLY "
        f"{config['display_name'].upper()} "
        f"VALIDATION"
    )

    print("=" * 60)

    passed = (
        validate_quarterly_instant_metric(
            quarterly_df=quarterly,
            source_facts=facts,
            metric_name=metric_name,
            require_nonnegative=config[
                "nonnegative_expected"
            ],
        )
    )

    print("\n" + "=" * 60)

    if passed:
        print("VALIDATION RESULT: PASS")
        print("Dataset passed all checks.")
    else:
        print("VALIDATION RESULT: FAIL")
        print("Dataset failed one or more checks.")


if __name__ == "__main__":
    main()