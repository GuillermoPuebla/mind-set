import pandas as pd


def adjust_csv(csv_file_path: str, col1: str, col2: str, new_col_name: str):
    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Ensure the specified columns exist in the DataFrame
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(
            f"One or both of the specified columns ({col1}, {col2}) do not exist in the DataFrame"
        )

    # Compute the difference between the two columns
    df[new_col_name] = df[col1] - df[col2]

    return df
