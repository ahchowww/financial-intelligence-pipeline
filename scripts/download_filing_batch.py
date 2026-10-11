import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.ingestion.sec_filings import(
    download_filing_document,
)

METADATA_PATH = Path(
    "data/raw/tsla_filings.csv"
)

FILINGS_DIR = Path(
    "data/raw/filings"
)

# Conservative pause between SEC requests
REQUEST_DELAY_SECONDS = 0.25


def main() -> None:
    load_dotenv()

    # 1. Load filing metadata
    filings = pd.read_csv(
        METADATA_PATH,
        parse_dates=[
            "report_date",
            "filed",
        ],
    )

    # 2. Keep original filings only
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

    FILINGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = len(filings)

    downloaded = 0
    skipped = 0
    failed = 0

    print("\n" + "=" * 60)
    print("TESLA SEC FILING BATCH DOWNLOAD")
    print("=" * 60)

    print(
        f"Filings to check: {total}"
    )

    # 3. Process filings one by one
    for index, row in filings.iterrows():
        accession_number = row["accession_number"]
        accession_no_dashes = accession_number.replace(
            "-",
            "",
        )

        primary_document = row["primary_document"]

        output_path = (
            FILINGS_DIR 
            / (
                f"{accession_no_dashes}_"
                f"{primary_document}"
            )
        )

        progress = (
            f"[{index + 1} / {total}]"
        )

        filing_label = (
            f"{row['report_date'].date()} "
            f"{row['form']}"
        )

        # skip files already downloaded 
        if output_path.exists():
            skipped += 1

            print(
                f"{progress} "
                f"SKIP "
                f"{filing_label} "
                f"({output_path.name})"
            )

            continue

        # Download missing filings
        print(
            f"{progress} "
            f"DOWNLOAD "
            f"{filing_label} "
        )


        try:
            saved_path = download_filing_document(
                filing_url=row["filing_url"],
                output_path=output_path,
            )

            file_size = saved_path.stat().st_size

            downloaded += 1
        
            print(
                f"SAVED "
                f"{file_size:,} bytes"
            )

        except requests.RequestException as exc:
            failed += 1
            print(
                f"FAILED "
                f"{type(exc).__name__}"
                f"{exc}"
            )

        # Pause only after a network request
        time.sleep(REQUEST_DELAY_SECONDS)


    # 4. Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    print(
        f"Total original filings: {total}"
    )

    print(
        f"Downloaded now: {downloaded}"
    )

    print(
        f"Already existed: {skipped}"
    )

    print(
        f"Available locally: "
        f"{downloaded + skipped}"
    )


if __name__ == "__main__":
    main()