from pathlib import Path

import pandas as pd

from src.processing.filing_text import(
    extract_filing_text,
    extract_mda,
)

METADATA_PATH = Path(
    "data/raw/tsla_filings.csv"
)

FILINGS_DIR = Path(
    "data/raw/filings"
)

MDA_DIR = Path(
    "data/processed/mda"
)

OUTPUT_METADATA_PATH = Path(
    "data/processed/tsla_mda_metadata.csv"
)


def main() -> None:
    # 1. Load filing metadata
    filings = pd.read_csv(
        METADATA_PATH,
        parse_dates=[
            "report_date",
            "filed",
        ],
    )

    # Original filings only
    filings = (
        filings[
            filings["form"].isin(
                [
                    "10-Q",
                    "10-K",
                ]
            )
        ]
        .copy()
        .sort_values(
            "filed"
        )
        .reset_index(
            drop=True
        )
    )

    MDA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    total = len(filings)

    print("\n" + "=" * 60)
    print("BUILD TESLA MD&A DATASET")
    print("=" * 60)

    # 2. Extract and save each MD&A
    for index, row in filings.iterrows():
        accession_number = row["accession_number"]
        accession_no_dashes = accession_number.replace(
            "-",
            "",
        )

        filing_path = (
            FILINGS_DIR
            / (
                f"{accession_no_dashes}_"
                f"{row['primary_document']}"
            )
        )

        full_text = extract_filing_text(filing_path)

        mda = extract_mda(
            text=full_text,
            form=row["form"],
        )


        # Use accession number for stable unique filename
        mda_path = (
            MDA_DIR
            / (
                f"{accession_no_dashes}_mda.txt"
            )
        )

        mda_path.write_text(
            mda,
            encoding="utf-8",
        )

        records.append(
            {
                "accession_number": accession_number,
                "form": row["form"],
                "report_date": row["report_date"],
                "filed": row["filed"],
                "primary_document": row["primary_document"],
                "filing_url": row["filing_url"],
                "filing_path": str(filing_path),
                "mda_path": str(mda_path),
                "mda_chars": len(mda),
                "mda_lines": len(
                    mda.splitlines()
                ),
            }
        )

        print(
            f"[{index + 1} / {total}] "
            f"{row['report_date'].date()}"
            f"{row['form']} "
            f"-> {len(mda):,} chars"
        )

    # 3. Build MD&A metadata table
    result = pd.DataFrame(records)

    OUTPUT_METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_METADATA_PATH,
        index=False,
    )

    # 4. Summary
    print("\n" + "=" * 60)

    print("MD&A DATASET SUMMARY")

    print("=" * 60)

    print(
        f"Rows: {len(result)}"
    )

    print(
        f"Text files: "
        f"{len(list(MDA_DIR.glob('*.txt')))}"
    )

    print(
        f"Metadata saved to: "
        f"{OUTPUT_METADATA_PATH}"
    )

    print(
        f"MD&A directory: "
        f"{MDA_DIR}"
    )

    print("\nCoverage:")

    print(
        f"{result['report_date'].min().date()} "
        f"-> "
        f"{result['report_date'].max().date()}"
    )

    print("\nSample:")

    print(
        result[
            [
                "report_date",
                "filed",
                "form",
                "mda_chars",
                "mda_path",
            ]
        ]
        .tail(10)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()