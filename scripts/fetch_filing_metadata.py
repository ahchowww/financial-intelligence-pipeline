from dotenv import load_dotenv

from pathlib import Path

from src.ingestion.sec_filings import(
    fetch_company_submissions,
    fetch_submission_file,
    extract_recent_filings,
    extract_historical_filings,
    combine_filings,
    filter_financial_filings,
    prepare_filing_metadata,
)


TESLA_CIK = "1318605"


def main() -> None:
    load_dotenv()

    submissions = fetch_company_submissions(TESLA_CIK)

    # Recent filings
    recent_filings = extract_recent_filings(submissions)

    # Historical filings
    historical_file_info = (
        submissions[
            "filings"
        ].get(
            "files",
            []
        )
    )

    historical_frames = []

    for file_info in historical_file_info:
        file_name = file_info["name"]

        print(
            f"Fetching historical file: {file_name}"
        )

        historical_json = fetch_submission_file(file_name)

        historical_df = extract_historical_filings(historical_json)

        historical_frames.append(historical_df)


    # Combine recent + historical
    all_filings = combine_filings(
        recent_filings=recent_filings,
        historical_filings=historical_frames,
    )

    financial_filings = filter_financial_filings(all_filings)

    clean_filings = prepare_filing_metadata(
        filings=financial_filings,
        cik=TESLA_CIK,
    )

    print("\n" + "=" * 60)
    print("TESLA 10-Q / 10-K FILINGS")
    print("=" * 60)

    print(
        f"Rows: "
        f"{len(financial_filings)}"
    )

    columns = [
        "accessionNumber",
        "filingDate",
        "reportDate",
        "form",
        "primaryDocument",
    ]

    print(
        financial_filings[columns].head(20).to_string(index=False)
    )


    print("\n" + "=" * 60)
    print("FILING DATE COVERAGE")
    print("=" * 60)

    print(
        "Newest filing date:", financial_filings["filingDate"].max()
    )

    print(
        "Oldest filing date: ", financial_filings["filingDate"].min()
    )

    print(
        "Newest report date:", financial_filings["reportDate"].max()
    )
    
    print(
        "Oldest report date: ", financial_filings["reportDate"].min()
    )


    print("\nOldest filings:")
    print(
        financial_filings[columns].tail(10).to_string(index=False)
    )


    historical_files = submissions["filings"].get(
        "files",
        []
    )

    print("\n" + "=" * 60)
    print("SEC HISTORICAL SUBMISSION FILES")
    print("=" * 60)

    print(
        f"Historical files: {len(historical_files)}"
    )

    for file_info in historical_files:
        print(file_info)


    # Save
    output_path = Path("data/raw/tsla_filings.csv")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_filings.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("SAVED FILING METADATA")
    print("=" * 60)

    print(f"Rows: {len(clean_filings)}")
    print(f"Saved to: {output_path}")

    print("\nSample:")
    print(
        clean_filings
        .tail(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()