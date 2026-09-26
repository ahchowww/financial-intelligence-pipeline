from pathlib import Path

from src.processing.dataset import(
    build_quarterly_dataset,
)


PROCESSED_DIR = Path("data/processed")

MASTER_OUTPUT_PATH = (PROCESSED_DIR / "tsla_quarterly_master.csv")

MODELING_OUTPUT_PATH = (PROCESSED_DIR / "tsla_quarterly_modeling.csv")

def main() -> None:
    """
    Build and save the merged quarterly financial datasets.

    Outputs:
    1. Master dataset
       - preserves all available historicl quarters
       - may contain missing metrics

    2. Modeling dataset
       - continuous quarterly history
       - all required metrics are available
    """

    master, modeling = (
        build_quarterly_dataset(PROCESSED_DIR)
    )

    # Save master dataset
    master.to_csv(
        MASTER_OUTPUT_PATH,
        index=False,
    )

    # Save modeling dataset
    modeling.to_csv(
        MODELING_OUTPUT_PATH,
        index=False,
    )

    # Summary
    print("\n" + "=" * 60)

    print("QUARTERLY DATASET BUILD")

    print("=" * 60)
    print(f"Master rows: {len(master)}")
    print(f"Modeling rows: {len(modeling)}")

    print("\nMaster Coverage: ")
    print(
        f"{master.iloc[0]['year']}"
        f"{master.iloc[0]['quarter']}"
        " -> "
        f"{master.iloc[-1]['year']}"
        f"{master.iloc[-1]['quarter']}"
    )

    print("\nModeling Coverage: ")
    print(
        f"{modeling.iloc[0]['year']}"
        f"{modeling.iloc[0]['quarter']}"
        " -> "
        f"{modeling.iloc[-1]['year']}"
        f"{modeling.iloc[-1]['quarter']}"
    )

    print("\nIncomplete master rows:")
    print(
        (~master["is_complete"]).sum()
    )

    print("\nSaved: ")
    print(MASTER_OUTPUT_PATH)
    print(MODELING_OUTPUT_PATH)


if __name__ == "__main__":
    main()
    