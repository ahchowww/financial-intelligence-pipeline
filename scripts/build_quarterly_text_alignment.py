from pathlib import Path

import pandas as pd

from src.processing.text_alignment import(
    build_quarterly_text_alignment,
)

QUARTERLY_PATH = Path(
    "data/processed/"
    "tsla_quarterly_master.csv"
)

MDA_METADATA_PATH = Path(
    "data/processed/"
    "tsla_mda_clean_metadata.csv"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "tsla_quarterly_text_alignment.csv"
)


def main() -> None:
    # 1. Load dataset
    quarterly = pd.read_csv(
        QUARTERLY_PATH
    )

    mda_metadata = pd.read_csv(
        MDA_METADATA_PATH
    )

    # 2. Align financial quarters + MD&A
    aligned = build_quarterly_text_alignment(
        quarterly=quarterly,
        mda_metadata=mda_metadata,
    ) 

    # 3. Save
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    aligned.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # 4. Summary
    print("\n" + "=" * 60)
    print("QUARTERLY + MD&A ALIGNMENT")
    print("=" * 60)

    print(
        f"Quarterly master rows: {len(quarterly)}"
    )

    print(
        f"MD&A metadata rows: {len(mda_metadata)}"
    )

    print(
        f"Aligned rows: {len(aligned)}"
    )

    print(
        f"Rows with MD&A: {int(aligned['has_mda'].sum())}"
    )

    print(
        f"Rows without MD&A: {int((~aligned['has_mda']).sum())}"
    )

    print(
        f"Complete financial + MD&A rows: "
        f"{int(aligned['multimodal_available_date'].notna().sum())}"
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )

    print("\nFirst rows with MD&A: ")
    sample = (
        aligned[
            aligned["has_mda"]
        ][
            [
                "year",
                "quarter",
                "quarter_end",
                "form",
                "mda_filed",
                "available_date",
                "multimodal_available_date",
                "mda_chars",
            ]
        ]
        .head(10)
    )

    print(
        sample.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()