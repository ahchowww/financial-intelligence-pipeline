from pathlib import Path

from src.processing.filing_text import(
    clean_mda_dataset,
)

MDA_PATH = Path(
    "data/processed/mda/"
    "000119312511221497_mda.txt"
)


def main() -> None:
    raw_text = MDA_PATH.read_text(
        encoding="utf-8",
    )

    clean_text = clean_mda_dataset(raw_text)

    print("\n" + "=" * 60)
    print("MD&A CLEANING TEST")
    print("=" * 60)

    print(
        f"Raw characters: {len(raw_text):,}"
    )

    print(
        f"Clean characters: {len(clean_text):,}"
    )

    print(
        f"Raw lines: {len(raw_text.splitlines()):,}"
    )

    print(
        f"Clean lines: {len(clean_text.splitlines()):,}"
    )

    print("\n" + "=" * 60)
    print("CLEAN TEXT START")
    print("=" * 60)

    print(clean_text[:2000])

    print("\n" + "=" * 60)
    print("CLEAN TEXT END")
    print("=" * 60)

    print(clean_text[-1500:])


if __name__ == "__main__":
    main()