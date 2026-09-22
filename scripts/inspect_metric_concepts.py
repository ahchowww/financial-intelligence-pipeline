import argparse
import json
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/tsla_company_facts.json")

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Search Tesla US-GAAP concepts using keywords."
        )
    )

    parser.add_argument(
        "terms",
        nargs="+",
        help="Search terms, e.g. gross profit",
    )

    args = parser.parse_args()

    search_terms = [
        term.lower()
        for term in args.terms
    ]

    with RAW_DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    us_gaap = data["facts"]["us-gaap"]

    matches = []

    for concept, metadata in us_gaap.items():
        label = metadata.get(
            "label",
            "",
        )

        description = metadata.get(
            "description", 
            "",
        )

        searchable_text = (
            f"{concept} "
            f"{label} "
            f"{description}"
        ).lower()

        if all(
            term in searchable_text
            for term in search_terms
        ):

            units = metadata.get(
                "units",
                {}
            )

            total_facts = sum(
                len(records)
                for records in units.values()
            )

            matches.append(
                {
                    "concept": concept,
                    "label": label,
                    "units": list(
                        units.keys()
                    ),
                    "fact_count": total_facts,
                }
            )

    if not matches:
        print("No matching concepts found.")
        return

    matches = sorted(
        matches,
        key=lambda item: item[
            "fact_count"
        ],
        reverse=True,
    )

    print("\nMatching US-GAAP concepts:\n")

    for match in matches:
        print(
            f"Concept: "
            f"{match['concept']}"
        )

        print(
            f"Label: "
            f"{match['label']}"
        )

        print(
            f"Units: "
            f"{match['units']}"
        )

        print(
            f"Fact Count: "
            f"{match['fact_count']}"
        )

        print("=" * 70)


if __name__ == "__main__":
    main()