"""Data quality assessment script.

Searches for the dataset in common locations, loads it with pandas, and prints
data types, descriptive statistics for `TotalPremium` and `TotalClaims`, missing
value counts, and converts `TransactionMonth` to datetime.
"""
from pathlib import Path
import pandas as pd
import glob
import sys


def find_data_file():
    # Candidate locations
    candidates = [
        Path('data') / 'MachineLearningRating_v3.txt',
        Path('data') / 'insurance.csv',
        Path('SM') / 'data' / 'insurance.csv',
        Path('demo') / 'SM' / 'data' / 'insurance.csv'
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback: search for any .csv or .txt in data or SM/data
    for pattern in ['data/*.[ct]sv', 'data/*.txt', 'SM/data/*.[ct]sv', 'SM/data/*.txt']:
        found = list(Path('.').glob(pattern))
        if found:
            return found[0]
    return None


def load_data(path):
    # determine delimiter
    sep = '|'
    # try reading header first line to decide sep
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            first = f.readline()
            if first.count(',') > first.count('|'):
                sep = ','
    except Exception:
        pass
    print(f'Loading {path} with sep="{sep}"')
    df = pd.read_csv(path, sep=sep, low_memory=False)
    return df


def main():
    path = find_data_file()
    if path is None:
        print('No dataset found in expected locations. Please place the file in `data/` or `SM/data/`.')
        sys.exit(2)

    df = load_data(path)

    print('\nData shape:', df.shape)

    # Print dtypes
    print('\nColumn dtypes:')
    print(df.dtypes)

    # Convert TransactionMonth if present
    if 'TransactionMonth' in df.columns:
        print('\nConverting TransactionMonth to datetime...')
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
        print(df['TransactionMonth'].head())

    # Descriptive stats for financials
    for col in ['TotalPremium', 'TotalClaims']:
        if col in df.columns:
            print(f'\nDescriptive statistics for {col}:')
            # coerce numeric
            s = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce')
            print(s.describe())
        else:
            print(f'Column {col} not found in dataset')

    # Missing values per column
    print('\nMissing values count per column:')
    miss = df.isna().sum()
    print(miss[miss > 0].sort_values(ascending=False))

    # Save a small processed sample for EDA (keep additional categorical/geography/vehicle columns)
    out = Path('data') / 'processed_sample_from_data_quality.csv'
    # keep a broader set of columns useful for EDA and stats
    desired = [
        'UnderwrittenCoverID','PolicyID','TransactionMonth','TotalPremium','TotalClaims',
        'Province','PostalCode','Gender','VehicleType','make','Model','CustomValueEstimate'
    ]
    keep = [c for c in desired if c in df.columns]
    df_sample = df.loc[:, keep].head(400000)
    df_sample.to_csv(out, index=False)
    print('\nSaved sample to', out)


if __name__ == '__main__':
    main()
