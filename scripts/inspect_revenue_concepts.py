import json
from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path(
    "data/raw/tsla_company_facts.json"
)


def print_target_years(
    concept_name: str,
    concept_data: dict,
) -> None:
    """
    Print observations between 2019 and 2022
    for a revenue-related concept.
    """

    units = concept_data.get("units") or {}

    if "USD" not in units:
        return

    observations = units["USD"]

    if not observations:
        return

    df = pd.DataFrame(observations)

    if df.empty:
        return

    # We need these columns for inspection.
    if "end" not in df.columns:
        return

    if "form" not in df.columns:
        return

    # Convert date column.
    df["end"] = pd.to_datetime(
        df["end"],
        errors="coerce",
    )

    # Keep normal SEC financial filings.
    allowed_forms = [
        "10-Q",
        "10-K",
        "10-Q/A",
        "10-K/A",
    ]

    df = df[
        df["form"].isin(allowed_forms)
    ].copy()

    # Keep only 2019–2022.
    df = df[
        df["end"].dt.year.between(
            2019,
            2022,
        )
    ].copy()

    if df.empty:
        return

    useful_columns = [
        "start",
        "end",
        "val",
        "form",
        "filed",
        "fy",
        "fp",
        "frame",
    ]

    available_columns = [
        column
        for column in useful_columns
        if column in df.columns
    ]

    print(
        f"\n### {concept_name} — 2019 to 2022"
    )

    print(
        df[available_columns]
        .sort_values(
            by=["end", "filed"]
        )
        .to_string(index=False)
    )


def main() -> None:
    # --------------------------------------------------
    # 1. Load SEC Company Facts
    # --------------------------------------------------
    with RAW_DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    us_gaap = data["facts"]["us-gaap"]

    # --------------------------------------------------
    # 2. Find concepts related to revenue
    # --------------------------------------------------
    revenue_concepts = []

    for concept_name, concept_data in us_gaap.items():

        # SEC may return None,
        # so convert None to an empty string.
        label = (
            concept_data.get("label")
            or ""
        )

        description = (
            concept_data.get("description")
            or ""
        )

        searchable_text = " ".join(
            [
                concept_name,
                label,
                description,
            ]
        ).lower()

        if "revenue" in searchable_text:
            revenue_concepts.append(
                (
                    concept_name,
                    concept_data,
                )
            )

    print(
        f"Found {len(revenue_concepts)} "
        f"revenue-related concepts.\n"
    )

    # --------------------------------------------------
    # 3. Inspect each concept
    # --------------------------------------------------
    for concept_name, concept_data in revenue_concepts:

        units = concept_data.get("units") or {}

        # We are currently interested in
        # monetary revenue facts.
        if "USD" not in units:
            continue

        observations = units["USD"]

        if not observations:
            continue

        df = pd.DataFrame(observations)

        if df.empty:
            continue

        if "end" not in df.columns:
            continue

        if "form" not in df.columns:
            continue

        df["end"] = pd.to_datetime(
            df["end"],
            errors="coerce",
        )

        allowed_forms = [
            "10-Q",
            "10-K",
            "10-Q/A",
            "10-K/A",
        ]

        df = df[
            df["form"].isin(allowed_forms)
        ].copy()

        if df.empty:
            continue

        min_date = df["end"].min()
        max_date = df["end"].max()

        label = (
            concept_data.get("label")
            or "N/A"
        )

        print("=" * 80)

        print(
            f"Concept: {concept_name}"
        )

        print(
            f"Label: {label}"
        )

        print(
            "Date coverage:",
            min_date.date(),
            "→",
            max_date.date(),
        )

        print(
            "Number of financial filing facts:",
            len(df),
        )

        print("\nForms:")
        print(
            df["form"].value_counts()
        )

        # --------------------------------------------------
        # 4. Show the period we are missing
        # --------------------------------------------------
        print_target_years(
            concept_name,
            concept_data,
        )

        print()


if __name__ == "__main__":
    main()