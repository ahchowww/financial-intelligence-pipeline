from pathlib import Path

import pandas as pd

METADATA_PATH = Path(
    "data/raw/tsla_filings.csv"
)

FILINGS_DIR = Path(
    "data/raw/filings"
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

    # 2. Keep the original filings only
    #    Exclude 10-Q/A & 10-K/A
    originals = (
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

    # 3. Build expected local file path
    originals["local_path"] = originals.apply(
        lambda row: 
        FILINGS_DIR / (
            row["accession_number"].replace(
                "-",
                "",
            )
            + "_"
            + row["primary_document"]
        ),
        axis=1,
    )

    # 4. Check which filings already exists
    originals["downloaded"] = originals["local_path"].apply(
        lambda path:
        Path(path).exists()
    )

    downloaded_count = int(
        originals["downloaded"].sum()
    )

    missing_count = len(originals) - downloaded_count


    # 5. Print summary
    print("\n" + "=" * 60)
    print("TESLA ORIGINAL FILING DOWNLOAD PLAN")
    print("=" * 60)

    print(
        f"Total metadata rows: "
        f"{len(filings)}"
    )

    print(
        f"Original 10-Q / 10-K filings: "
        f"{len(originals)}"
    )

    print(
        f"Already downloaded: "
        f"{downloaded_count}"
    )

    print(
        f"Still missing: "
        f"{missing_count}"
    )


    # 6. Form counts
    print("\nForm counts: ")

    print(
        originals["form"].value_counts()
    )

    # 7. Coverage
    print("\nCoverage: ")

    print("Oldest: ")
    print(
        originals[
            [
                "report_date",
                "filed",
                "form",
                "accession_number",
            ]
        ]
        .head(5)
        .to_string(
            index=False
        )
    )

    print("Newest: ")
    print(
        originals[
            [
                "report_date",
                "filed",
                "form",
                "accession_number",
            ]
        ]
        .tail(5)
        .to_string(
            index=False
        )
    )


    # 8. Show existing downloaded files
    downloaded = originals[
        originals["downloaded"]
    ]

    print("\n" + "=" * 60)
    print("ALREADY DOWNLOADED")
    print("=" * 60)

    if downloaded.empty:
        print("None")
    else:
        print(
            downloaded[
                [
                    "report_date",
                    "form",
                    "local_path",
                ]
            ]
            .to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
