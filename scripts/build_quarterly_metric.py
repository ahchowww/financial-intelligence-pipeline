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

RAW_DATA_PATH = Path("data/raw/tsla_company_facts.json")

OUTPUT_DIR = Path("data/processed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a normalized quarterly financial metric."
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

    # currently only support duration metric
    if config["fact_type"] != "duration":
        raise ValueError(
            f"{metric_name} is not a duration metric."
        )

    # load SEC company facts
    with RAW_DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    # extract metric
    facts = extract_concept(
        company_facts=data,
        concept=config["primary_concept"],
        unit=config["unit"],
    )

    facts = filter_financial_filings(facts)

    # build quarterly series
    quarterly = build_quarterly_series(facts)

    # save result
    output_path = (
        OUTPUT_DIR 
        / f"tsla_quarterly_{metric_name}.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    quarterly.to_csv(
        output_path,
        index=False,
    )

    display_columns =[
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

    print(f"\n=== {config['display_name']} ===")

    print(
        quarterly[display_columns]
        .to_string(index=False)
    )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()