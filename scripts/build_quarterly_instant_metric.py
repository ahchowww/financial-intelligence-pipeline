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

RAW_DATA_PATH = Path("data/raw/tsla_company_facts.json")

OUTPUT_DIR = Path("data/processed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a quarterly instant financial metric."
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

    # load SEC data
    with RAW_DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    # extract concept
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

    # save
    output_path = (
        OUTPUT_DIR / f"tsla_quarterly_{metric_name}.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    quarterly.to_csv(
        output_path,
        index=False,
    )

    print(f"\n=== {config['display_name']} ===")

    print(
        quarterly[
            [
                "year",
                "quarter",
                "end",
                "val",
                "filed",
                "form",
                "derived",
                "source_method",
            ]
        ]
        .to_string(index=False)
    )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()