import re

from pathlib import Path

from src.processing.filing_text import(
    extract_filing_text,
)

FILING_PATH = Path(
    "data/raw/filings/"
    "000156459017009968_"
    "tsla-10q_20170331.htm"
)

"""
Main Idea:
1. Take one problematic filing, convert to clean text.
2. Search broadly for MD&A related phases & Item 2 / Item 3 headings.
"""

def show_matches(
        text: str,
        pattern: str,
        label: str,
        context: int = 800,
) -> None:

    matches = list(
        re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
    )

    print("\n" + "=" * 60)
    print(
        f"{label}: {len(matches)} matches"
    )
    print("=" * 60)

    for i, match in enumerate(matches, start=1):
        position = match.start()

        print("\n" + "=" * 60)
        print(
            f"{label} #{i}"
            f"\nPosition: {position}"
        )
        print("=" * 60)

        start = max(
            0, 
            position - 200,
        )

        end = min(
            len(text),
            position + context,
        )

        print(text[start: end])


def main() -> None:
    text = extract_filing_text(FILING_PATH)

    print(
        f"Full filing characters: "
        f"{len(text):,}"
    )

    show_matches(
        text=text,
        pattern=r"management.{0,20}discussion",
        label="MANAGEMENT DISCUSSION",
    )

    show_matches(
        text=text,
        pattern=r"quantitative.{0,30}qualitative",
        label="QUANTITATIVE QUALITATIVE",
    )

    show_matches(
        text=text,
        pattern=r"\bitem\s*2\b",
        label="ITEM 2",
    )

    show_matches(
        text=text,
        pattern=r"\bitem\s*3\b",
        label="ITEM 3",
    )


if __name__ == "__main__":
    main()