from pathlib import Path

import pandas as pd

from src.processing.text_alignment import(
    build_multimodal_modeling_dataset,
)

ALIGNMENT_PATH = Path(
    "data/processed/"
    "tsla_quarterly_text_alignment.csv"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "tsla_multimodal_modeling.csv"
)


def main() -> None:
    # 1. Load aligned dataset
    aligned = pd.read_csv(
        ALIGNMENT_PATH
    )

    # 2. Build continuous multimodal dataset
    modeling = build_multimodal_modeling_dataset(aligned)
    
    # 3. Save
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    modeling.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # 4. Summary
    print("\n" + "=" * 60) 

    print("MULTIMODAL MODELING DATASET")

    print("=" * 60)

    print(
        f"Rows: {len(modeling)}"
    )

    print(
        f"Coverage: "
        f"{modeling.iloc[0]['year']}"
        f"{modeling.iloc[0]['quarter']} "
        f"-> "
        f"{modeling.iloc[-1]['year']}"
        f"{modeling.iloc[-1]['quarter']}"
    )

    print(
        f"First quarter end: "
        f"{modeling.iloc[0]['quarter_end']}"
    )

    print(
        f"Last quarter end: "
        f"{modeling.iloc[-1]['quarter_end']}"
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )

    print(
        "\nFirst 5 rows:"
    )

    print(
        modeling[
            [
                "year",
                "quarter",
                "quarter_end",
                "form",
                "available_date",
                "mda_filed",
                "multimodal_available_date",
                "mda_chars",
            ]
        ]
        .head()
        .to_string(
            index=False
        )
    )

    print(
        "\nLast 5 rows:"
    )

    print(
        modeling[
            [
                "year",
                "quarter",
                "quarter_end",
                "form",
                "available_date",
                "mda_filed",
                "multimodal_available_date",
                "mda_chars",
            ]
        ]
        .tail()
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()