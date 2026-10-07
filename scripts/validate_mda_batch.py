from pathlib import Path

import pandas as pd

from src.processing.filing_text import(
    extract_filing_text,
    extract_mda,
)

METADATA_PATH = Path(
    "data/raw/tsla_filings.csv"
)

FILLINGS_DIR = Path(
    "data/raw/filings"
)


def main() -> None:
    # 1. Load metadata
    filings = pd.read_csv(
        METADATA_PATH,
        parse_dates=[
            "report_date",
            "filed",
        ],
    )

    # Keep original filings only
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

    total = len(filings)

    successes = []
    failures = []


    print("\n" + "=" * 60)
    print("TESLA MD&A BATCH VALIDATION")
    print("=" * 60)

    print(
        f"Filings to evaluate: {total}"
    )


    # 2. Process each filing
    for index, row in filings.iterrows():
        accession_number = row["accession_number"]

        accession_no_dashes = accession_number.replace(
            "-",
            "",
        )

        primary_document = row["primary_document"]

        filing_path = (
            FILLINGS_DIR
            / (
                f"{accession_no_dashes}_"
                f"{primary_document}"
            )
        )

        label = (
            f"[{index + 1} / {total}] "
            f"{row['report_date'].date()}"
            f"{row['form']}"
        )

        # Missing local file
        if not filing_path.exists():
            print(
                f"{label} -> MISSING FILE"
            )

            failures.append(
                {
                    "accession_number": accession_number,
                    "form": row["form"],
                    "report_date": row["report_date"],
                    "error": "Local filing file missing",
                }
            )

            continue

        # Extract filing text + MD&A
        try: 
            full_text = extract_filing_text(filing_path)

            mda = extract_mda(
                text=full_text,
                form=row["form"],
            )

            mda_chars = len(mda)

            mda_lines = len(mda.splitlines())

            print(
                f"{label} -> OK "
                f"{mda_chars:,} chars, "
                f"{mda_lines:,} lines"
            )

            successes.append(
                {
                    "accession_number": accession_number,
                    "form": row["form"],
                    "report_date": row["report_date"],
                    "filed": row["filed"],
                    "full_text_chars": len(full_text),
                    "mda_chars": mda_chars,
                    "mda_lines": mda_lines,
                }
            )

        except Exception as exc:
            print(
                f"{label} -> FAILED: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            failures.append(
                {
                    "accession_number": accession_number,
                    "form": row["form"],
                    "report_date": row["report_date"],
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

    # 3. Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(
        f"Total filings: {total}"
    )

    print(
        f"Successful: {len(successes)}"
    )

    print(
        f"Failed: {len(failures)}"
    )

    # 4. Successful extraction statistics
    if successes:
        success_df = pd.DataFrame(successes)

        print("\nMD&A character statistics: ")
        print(
            success_df["mda_chars"].describe()
        )
        
        print("\nSmallest MD&A extractions: ")
        print(
            success_df
            .sort_values(
                "mda_chars"
            )
            [
                [
                    "report_date",
                    "form",
                    "mda_chars",
                    "mda_lines",
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )


    # 5. Failures
    if failures:
        failure_df = pd.DataFrame(failures)

        print("\n" + "=" * 60)
        print("FAILED FILINGS")
        print("=" * 60)

        print(
            failure_df.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()