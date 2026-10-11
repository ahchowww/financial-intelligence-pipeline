from pathlib import Path

import pandas as pd

from dotenv import load_dotenv

from src.ingestion.sec_filings import(
    download_filing_document,
)

METADATA_PATH = "data/raw/tsla_filings.csv"

TARGET_REPORT_DATE = "2010-12-31"

TARGET_FORM = "10-K"


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

    # 2. Select one Tesla filing
    target = filings[
        (filings["report_date"] == pd.Timestamp(TARGET_REPORT_DATE))
        &
        (filings["form"] == TARGET_FORM)
    ].copy()

    if len(target) != 1:
        raise ValueError(
            f"Expected exactly one filing, found {len(target)}"
        )

    # Get the selected row
    # target -> 1-row DataFrame
    row = target.iloc[0]

    accession_number = row["accession_number"]

    primary_document = row["primary_document"]

    # filing_url -> use to download the document
    filing_url = row["filing_url"]


    # 3. Build local output path
    accession_no_dashes = accession_number.replace(
        "-",
        "",
    )

    output_path = Path(
        "data/raw/filings"
    ) / (
        f"{accession_no_dashes}_"
        f"{primary_document}"
    )

    # 4. Download
    print("\n" + "=" * 60)
    print("DOWNLOAD SINGLE SEC FILING")
    print("=" * 60)

    print(
        f"Form: {row['form']}"
    )

    print(
        f"Report date: "
        f"{row['report_date'].date()}"
    )

    print(
        f"Filed: "
        f"{row['filed'].date()}"
    )

    print(
        f"Accession: "
        f"{accession_number}"
    )

    print(
        f"URL: "
        f"{filing_url}"
    )

    saved_path = download_filing_document(
        filing_url=filing_url,
        output_path=output_path,
    )

    # 5. Basic verification
    file_size = saved_path.stat().st_size

    print("\nDownload successful.")

    print(
        f"Saved to: {saved_path}"
    )

    print(
        f"File size: "
        f"{file_size:,} bytes"
    )

    print("\nFirst 300 characters:")

    html_preview = (
        saved_path
        .read_text(
            encoding="utf-8",
            errors="ignore",
        )
    )

    print(html_preview[:300])


if __name__ == "__main__":
    main()