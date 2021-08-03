import pandas as pd


def add_missing_val_indicators(df: pd.DataFrame):
    """
    Adds [column]_was_missing columns for each [column] with missing values.
    """
    cols_with_missing = [col for col in df.columns
                         if df[col].isnull().any()]

    indicated_df = df.copy()
    for col in cols_with_missing:
        indicated_df[col + '_was_missing'] = indicated_df[col].isnull()

    return indicated_df
