from pathlib import Path

from src.processing.filing_text import (
    extract_filing_text,
)


FILING_PATH = Path(
    "data/raw/filings/"
    "000162828024032662_"
    "tsla-20240630.htm"
)


def main() -> None:
    text = extract_filing_text(FILING_PATH)

    print("\n" + "=" * 60)
    print("SEC FILING TEXT INSPECTION")
    print("=" * 60)

    print(
        f"Characters: "
        f"{len(text):,}"
    )

    print(
        f"Lines: "
        f"{len(text.splitlines()):,}"
    )

    print("\n" + "=" * 60)
    print("FIRST 3,000 CHARACTERS")
    print("=" * 60)

    print(text[:3000])


    # Search for MD&A heading
    search_terms = [
        (
            "Management's Discussion and Analysis"
        ),
        (
            "Management’s Discussion and Analysis"
        ),
        "ITEM 2",
        "Item 2",
    ]

    print("\n" + "=" * 60)
    print("MD&A HEADING SEARCH")
    print("=" * 60)

    lower_text = text.lower()

    for term in search_terms:
        position = lower_text.find(
            term.lower()
        )

        print(
            f"{term!r}: "
            f"{position}"
        )

        if position != -1:
            start = max(
                0,
                position - 300,
            )

            end = min(
                len(text),
                position + 1500,
            )

            print("\nContext:")

            print(text[start:end])

            print("\n" + "-" * 60)


if __name__ == "__main__":
    main()