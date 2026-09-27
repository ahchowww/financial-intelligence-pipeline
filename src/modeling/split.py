import pandas as pd

def chronological_train_val_test_split(
        df: pd.DataFrame,
        validation_size: int = 8,
        test_size: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-series dataset chronologically.
    No random shuffling is performed.

    Earlier observations -> training
    Middle observation -> validation
    Latest observation -> test
    """

    if validation_size <= 0:
        raise ValueError(
            "Validation_size must be greater than 0."
        )

    if test_size <= 0:
        raise ValueError(
            "Test_size must be greater than 0."
        )

    required_rows = (
        validation_size + test_size + 1
    )

    if len(df) < required_rows:
        raise ValueError(
            "Dataset is too small for the required split."
        )

    required_columns = {
        "year",
        "quarter",
        "quarter_end",
        "target_quarter_end",
        "target_revenue",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )


    # 1. Ensure chronological order
    data = (
        df
        .sort_values(
            "quarter_end"
        )
        .reset_index(drop=True)
    )

    # 2. Calculate split boundaries
    train_end = (
        len(data) - validation_size - test_size
    )

    validation_end = (
        len(data) - test_size
    )

    # 3. Split
    train = (
        data.iloc[:train_end]
        .copy()
        .reset_index(drop=True)
    )

    validation = (
        data.iloc[train_end: validation_end]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        data.iloc[validation_end:]
        .copy()
        .reset_index(drop=True)
    )

    # 4. Time leakage protection
    if (train["target_quarter_end"].max() >= validation["target_quarter_end"].min()):
        raise ValueError(
            "Training and validation periods overlap."
        )

    if (validation["target_quarter_end"].max() >= test["target_quarter_end"].min()):
        raise ValueError(
            "Validation and test periods overlap."
        )

    return train, validation, test