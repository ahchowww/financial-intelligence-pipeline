from pathlib import Path

import pandas as pd

from src.validation.dataset import(
    FILED_COLUMNS,
    validate_quarterly_dataset,
)

PROCESSED_DIR = Path("data/processed")

MASTER_PATH = (PROCESSED_DIR / "tsla_quarterly_master.csv")

MODELING_PATH = (PROCESSED_DIR / "tsla_quarterly_modeling.csv")


def load_dataset(
        path: Path,
) -> pd.DataFrame:
    """
    Load a merged quarterly dataset with correct date types.
    """

    date_columns = [
        "quarter_end",
        "available_date",
        *FILED_COLUMNS,
    ]

    df = pd.read_csv(
        path,
        parse_dates=date_columns,
    )

    df["year"] = (
        df["year"]
        .astype("Int64")
    )

    df["quarter_number"] = (
        df["quarter_number"]
        .astype("Int64")
    )

    return df


def main() -> None:

    master = load_dataset(
        MASTER_PATH
    )

    modeling = load_dataset(
        MODELING_PATH
    )

    print("\n" + "=" * 60)

    print("QUARTERLY DATASET VALIDATION")

    print("=" * 60)

    passed = (
        validate_quarterly_dataset(
            master=master,
            modeling=modeling,
            processed_dir=PROCESSED_DIR,
        )
    )

    print("\n" + "=" * 60)

    print("DATASET SUMMARY")

    print("=" * 60)

    print(f"Master rows: {len(master)}")

    print(f"Modeling rows: {len(modeling)}")

    print(
        "Modeling coverage: "
        f"{modeling.iloc[0]['year']}"
        f"{modeling.iloc[0]['quarter']}"
        " -> "
        f"{modeling.iloc[-1]['year']}"
        f"{modeling.iloc[-1]['quarter']}"
    )

    print("\n"+ "=" * 60)

    if passed:
        print("VALIDATION RESULT: PASS")
        print("Dataset passed all checks.")
    else:
        print("VALIDATION RESULT: FAIL")
        print("Dataset failed one or more checks.")


if __name__ == "__main__":
    main()