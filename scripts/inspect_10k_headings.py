import re

from pathlib import Path

from src.processing.filing_text import(
    extract_filing_text,
)

FILING_PATH = Path(
    "data/raw/filings/"
    "000162828024002390_"
    "tsla-20231231.htm"
)


def main() -> None:
    text = extract_filing_text(FILING_PATH)

    print("\n" + "=" * 60)
    print("DEBUG 10-K ITEM 7 / ITEM 7A HEADINGS")
    print("=" * 60)

    # Find every Item 7
    item7_matches = list(
        re.finditer(
            r"\bitem\s*7\s*\.?",
            text,
            flags=re.IGNORECASE,
        )
    )

    print("\n" + "=" * 60)
    print(
        f"\nItem 7 matches: "
        f"{len(item7_matches)}"
    )

    for i, match in enumerate(item7_matches, start=1):
        position = match.start()

        print("\n" + "=" * 60)
        print(
            f"ITEM 7 #{i}"
        )
        print(
            f"Position: {position}"
        )

        print("=" * 60)

        print(text[position: position + 800])


    # Find every Item 7A
    item7a_matches = list(
        re.finditer(
            r"\bitem\s*7a\s*\.?",
            text,
            flags=re.IGNORECASE,
        )
    )

    print("\n" + "=" * 60)
    print(
        f"\nItem 7a matches: "
        f"{len(item7a_matches)}"
    )

    for i, match in enumerate(item7a_matches, start=1):
        position = match.start()

        print("\n" + "=" * 60)
        print(
            f"ITEM 7A #{i}"
        )
        print(
            f"Position: {position}"
        )

        print("=" * 60)

        print(text[position: position + 500])


if __name__ == "__main__":
    main()