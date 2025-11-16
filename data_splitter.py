import pandas as pd
from pathlib import Path

def split_csv_file(
    input_file: str,
    train_output: str = "train_split.csv",
    test_output: str = "test_split.csv",
    test_size: float = 0.2,
    random_seed: int = 42,
    stratify_col: str = None
):
    """
    Split a CSV file into train/test with optional stratification.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_file} not found.")

    df = pd.read_csv(input_path)

    if stratify_col and stratify_col in df.columns:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_seed, stratify=df[stratify_col]
        )
    else:
        train_df = df.sample(frac=1 - test_size, random_state=random_seed)
        test_df = df.drop(train_df.index)

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    print(f"CSV split completed:")
    print(f"  Total: {len(df)}")
    print(f"  Train: {len(train_df)} -> {train_output}")
    print(f"  Test : {len(test_df)} -> {test_output}")


# === RUN EXAMPLE ===
if __name__ == "__main__":
    split_csv_file(
        input_file="dataset_apartment_rentals.csv",
        train_output="train_split.csv",
        test_output="test_split.csv",
        test_size=0.2,
        random_seed=42
    )