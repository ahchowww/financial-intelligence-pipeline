from pathlib import Path

import pandas as pd

from src.processing.filing_text import(
    clean_mda_dataset,
)

METADATA_PATH = Path(
    "data/processed/tsla_mda_metadata.csv"
)

CLEAN_MDA_DIR = Path(
    "data/processed/mda_clean"
)

OUTPUT_METADATA_PATH = Path(
    "data/processed/tsla_mda_clean_metadata.csv"
)

"""
Main Idea:
1. Read every raw MD&A file listed in tsla_mda_metadata.csv.
2. Apply clean_mda_dataset() to each one.
3. Save the cleaned version into a new folder, and create a new metadata CSV
   describing the cleaned dataset.

### apply the same cleaning to all files.
"""

def main() -> None:
    # 1. Load MD&A metadata
    metadata = pd.read_csv(
        METADATA_PATH,
        parse_dates=[
            "report_date",
            "filed",
        ],
    )

    CLEAN_MDA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    total = len(metadata)

    print("\n" + "=" * 60)
    print("BUILD CLEAN TESLA MD&A DATASET")
    print("=" * 60)

    # 2. Clean each MD&A file
    for index, row in metadata.iterrows():
        raw_path = Path(row["mda_path"])

        if not raw_path.exists():
            raise FileNotFoundError(
                f"Missing MD&A file: {raw_path}"
            )

        raw_text = raw_path.read_text(
            encoding="utf-8",
        )

        clean_text = clean_mda_dataset(raw_text)

        accession_no_dashes = row["accession_number"].replace(
            "-",
            "",
        )

        clean_path = (
            CLEAN_MDA_DIR
            / (
                f"{accession_no_dashes}_"
                f"mda_clean.txt"
            )
        )

        clean_path.write_text(
            clean_text,
            encoding="utf-8",
        )

        # Convert the existing metadata row into dictionary
        record = row.to_dict()

        record["clean_mda_path"] = str(clean_path)
        record["clean_mda_chars"] = len(clean_text)
        record["clean_mda_lines"] = len(clean_text.splitlines())
        record["chars_removed"] = (
            len(raw_text) - len(clean_text)
        )

        records.append(record)

        print(
            f"[{index+1} / {total}] "
            f"{row['report_date'].date()} "
            f"{row['form']} "
            f"-> "
            f"{len(raw_text):,} "
            f"to "
            f"{len(clean_text):,} chars"
        )

    # 3. Save clean metadata
    result = pd.DataFrame(records)

    result.to_csv(
        OUTPUT_METADATA_PATH,
        index=False,
    )

    # 4. Summary
    print("\n" + "=" * 60)
    print("CLEAN MD&A DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Rows: {len(result)}"
    )

    print(
        f"Clean text files: "
        f"{len(list(CLEAN_MDA_DIR.glob('*.txt')))}"
    )

    print(
        f"Metadata saved to: {OUTPUT_METADATA_PATH}"
    )

    print(
        f"Clean MD&A directory: {CLEAN_MDA_DIR}"
    )

    print("\nCharacter statistics: ")
    print(
        result[
            [
                "mda_chars",
                "clean_mda_chars",
                "chars_removed",
            ]
        ]
        .describe()
    )

    print("\nLargest cleaning reductions:")
    print(
        result[
            [
                "report_date",
                "form",
                "mda_chars",
                "clean_mda_chars",
                "chars_removed",
            ]
        ]
        .sort_values(
            "chars_removed",
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()