from pathlib import Path

import pandas as pd

METADATA_PATH = Path(
    "data/processed/tsla_mda_clean_metadata.csv"
)


def main() -> None:
    # 1. Load metadata
    metadata = pd.read_csv(
        METADATA_PATH,
        parse_dates=[
            "report_date",
            "filed",
        ],
    )

    errors = []

    print("\n" + "=" * 60)
    print("VALIDATE CLEAN TESLA MD&A DATASET")
    print("=" * 60)

    # 2. Basic dataset check
    print(f"Rows: {len(metadata)}")

    if len(metadata) != 65:
        errors.append(
            f"Expected 65 rows, found {len(metadata)}"
        )

    if metadata["accession_number"].duplicated().any():
        errors.append(
            "Duplicate accession numbers found."
        )

    # 3. Validate each clean text file
    for index, row in metadata.iterrows():
        clean_path = Path(row["clean_mda_path"])

        label = (
            f"{row['report_date'].date()} "
            f"{row['form']}"
        )

        if not clean_path.exists():
            errors.append(
                f"{label}: "
                f"missing clean file {clean_path}"
            )

            continue

        text = clean_path.read_text(
            encoding="utf-8",
        )

        actual_chars = len(text)

        actual_lines = len(text.splitlines())

        # Metadata must match actual file
        if (actual_chars != int(row["clean_mda_chars"])):
            errors.append(
                f"{label}: "
                f"character-count mismatch"
            )

        if (actual_lines != int(row["clean_mda_lines"])):
            errors.append(
                f"{label}: "
                f"line-count mismatch"
            )

        # Text should not be suspiciously short
        if actual_chars < 10000:
            errors.append(
                f"{label}: "
                f"suspiciously short MD&A "
                f"({actual_chars:,} chars)"
            )

        # Verify expected section heading
        compact_start = (
            "".join(
                text[:500].lower().split()
            )
            .replace(
                "’",
                "'",
            )
        )

        if ("management'sdiscussionandanalysis" not in compact_start):
            errors.append(
                f"{label}: "
                f"MD&A heading missing near beginning"
            )


        # cleaning should never add characters
        raw_chars = int(row["mda_chars"])

        if actual_chars > raw_chars:
            errors.append(
                f"{label}: "
                f"clean text is longer than raw text"
            ) 

        # detect excessive cleaning
        removed_fraction = (
            (raw_chars - actual_chars) / raw_chars
        )

        if removed_fraction > 0.05:
            errors.append(
                f"{label}: "
                f"more than 5% of text removed "
                f"({removed_fraction:.2%})"
            )

        # point-in-time chronology
        if (row["filed"] < row["report_date"]):
            errors.append(
                f"{label}: "
                f"filed date precedes report date"
            )

    # 4. Summary statistics
    print("\nCoverage:")
    print(
        f"{metadata['report_date'].min().date()} "
        f"-> "
        f"{metadata['report_date'].max().date()}"
    )

    print("\nForms: ")
    print(metadata["form"].value_counts())

    print("\nClean MD&A character statistics: ")
    print(metadata["clean_mda_chars"].describe())

    # 5. Final result
    print("\n" + "=" * 60)

    if errors:
        print("VALIDATION RESULT: FAIL")
        print("=" * 60)

        print(
            f"Errors: {len(errors)}"
        )

        for error in errors:
            print(
                f"- {error}"
            )

    else:
        print("VALIDATION RESULT: PASS")
        print("=" * 60)

        print("All 65 clean MD&A files passed validation")


if __name__ == "__main__":
    main()