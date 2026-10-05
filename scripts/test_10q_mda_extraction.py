import re

from pathlib import Path

from src.processing.filing_text import(
    extract_filing_text,
    extract_10q_mda,
)

FILING_PATH = Path(
    "data/raw/filings/"
    "000119312511221497_d10q.htm"
)


def main() -> None:
    # 1. Extract full filing text
    text = extract_filing_text(FILING_PATH)

    # 2. Extract MD&A
    mda = extract_10q_mda(text)

    # 3. Inspect result
    print("\n" + "=" * 60)
    print("10-Q MD&A EXTRACTION")
    print("=" * 60)

    print(
        f"Full filing characters: {len(text):,}"
    )

    print(
        f"MD&A characters: {len(mda):,}"
    )

    print(
        f"MD&A lines: {len(mda.splitlines()):,}"
    )

    print("\n" + "=" * 60)
    print("MD&A START")
    print("=" * 60)

    print(mda[:2000])

    print("\n" + "=" * 60)
    print("MD&A END")
    print("=" * 60)

    print(mda[-1500:])


if __name__ == "__main__":
    main()