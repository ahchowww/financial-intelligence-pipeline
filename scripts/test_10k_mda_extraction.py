from pathlib import Path

from src.processing.filing_text import(
    extract_filing_text,
    extract_10k_mda,
)

FILING_PATH = Path(
    "data/raw/filings/"
    "000119312511054847_d10k.htm"
)


def main() -> None:
    text = extract_filing_text(FILING_PATH)

    mda = extract_10k_mda(text)

    print("\n" + "=" * 60)
    print("10-K MD&A EXTRACTION")
    print("=" * 60)

    print(
        f"Full filing characters: "
        f"{len(text):,}"
    )

    print(
        f"MD&A characters: "
        f"{len(mda):,}"
    )

    print(
        f"MD&A lines: "
        f"{len(mda.splitlines()):,}"
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