"""Statistical tests for Task 3: validate or reject hypotheses about risk drivers.

This script computes Claim Frequency, Claim Severity, and Margin, then performs:
- Chi-square tests for differences in claim frequency across groups
- Kruskal-Wallis (non-parametric) tests for claim severity and margin differences
- Pairwise tests where appropriate

Outputs p-values and plain-language interpretations.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


def find_input():
    candidates = [
        Path('data') / 'processed_sample_from_notebook.csv',
        Path('data') / 'processed_sample_from_data_quality.csv',
        Path('data') / 'MachineLearningRating_v3.txt'
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback: any csv in data
    for p in Path('data').glob('*.csv'):
        return p
    return None


def load_data(p):
    if p.suffix == '.txt':
        sep = '|'
    else:
        sep = ','
    # parse TransactionMonth if present
    try:
        df = pd.read_csv(p, sep=sep, parse_dates=['TransactionMonth'], low_memory=False)
    except Exception:
        df = pd.read_csv(p, sep=sep, low_memory=False)
    # normalize
    df.columns = [c.strip() for c in df.columns]
    return df


def prepare(df):
    # numeric conversions
    for c in ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '').str.strip(), errors='coerce')
    # create metrics
    df['ClaimOccurred'] = (df['TotalClaims'].fillna(0) > 0).astype(int) if 'TotalClaims' in df.columns else 0
    df['ClaimSeverity'] = df['TotalClaims'].where(df['ClaimOccurred'] == 1)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims'] if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns else np.nan
    return df


def chi2_test_frequency(df, group_col):
    # contingency table
    ct = pd.crosstab(df[group_col].fillna('MISSING'), df['ClaimOccurred'])
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    return chi2, p, ct


def kw_test_numeric(df, group_col, value_col, min_group_size=20):
    groups = []
    labels = []
    for name, g in df.groupby(group_col):
        vals = g[value_col].dropna().values
        if len(vals) >= min_group_size:
            groups.append(vals)
            labels.append(name)
    if len(groups) < 2:
        return None, None, labels
    stat, p = stats.kruskal(*groups)
    return stat, p, labels


def summary_overall(df):
    total_premium = df['TotalPremium'].sum(skipna=True)
    total_claims = df['TotalClaims'].sum(skipna=True)
    loss_ratio = (total_claims / total_premium) if total_premium else np.nan
    print(f"Overall: TotalPremium={total_premium:.2f}, TotalClaims={total_claims:.2f}, LossRatio={loss_ratio:.4f}")


def run_tests(df):
    print('\n--- Overall Summary ---')
    summary_overall(df)

    # H0: no risk differences across provinces (frequency & severity)
    if 'Province' in df.columns:
        print('\nTest: Claim frequency across Provinces (Chi-square)')
        chi2, p_freq, ct = chi2_test_frequency(df, 'Province')
        print('Chi2=%.3f, p=%.4g' % (chi2, p_freq))
        print('Contingency table (top rows):')
        print(ct.head())
        if p_freq < 0.05:
            print('RESULT: Reject H0 — claim frequency differs across provinces (p < 0.05)')
        else:
            print('RESULT: Fail to reject H0 — no evidence of frequency differences across provinces')

        print('\nTest: Claim severity across Provinces (Kruskal-Wallis)')
        stat, p_sev, labels = kw_test_numeric(df[df['ClaimOccurred'] == 1], 'Province', 'ClaimSeverity')
        if p_sev is None:
            print('Not enough groups with sufficient claims for severity test')
        else:
            print('KW_stat=%.3f, p=%.4g' % (stat, p_sev))
            if p_sev < 0.05:
                print('RESULT: Reject H0 — claim severity differs across provinces (p < 0.05)')
            else:
                print('RESULT: Fail to reject H0 — no evidence of severity differences across provinces')
    else:
        print('Province column not found — skipping province tests')

    # H0: no risk differences between zip codes (PostalCode)
    if 'PostalCode' in df.columns:
        # limit to top 10 postal codes by count to keep tests meaningful
        top_zips = df['PostalCode'].value_counts().nlargest(10).index.tolist()
        df_z = df[df['PostalCode'].isin(top_zips)].copy()
        print('\nTop postal codes used for tests:', top_zips)
        print('\nTest: Claim frequency across top PostalCodes (Chi-square)')
        chi2, p_zip_freq, ct_zip = chi2_test_frequency(df_z, 'PostalCode')
        print('Chi2=%.3f, p=%.4g' % (chi2, p_zip_freq))
        if p_zip_freq < 0.05:
            print('RESULT: Reject H0 — claim frequency differs across top postal codes (p < 0.05)')
        else:
            print('RESULT: Fail to reject H0 — no evidence of frequency differences across top postal codes')

        print('\nTest: Margin differences across top PostalCodes (Kruskal-Wallis)')
        stat_m, p_m, labels = kw_test_numeric(df_z, 'PostalCode', 'Margin')
        if p_m is None:
            print('Not enough data for margin test across postal codes')
        else:
            print('KW_stat=%.3f, p=%.4g' % (stat_m, p_m))
            if p_m < 0.05:
                print('RESULT: Reject H0 — margin differs across top postal codes (p < 0.05)')
            else:
                print('RESULT: Fail to reject H0 — no evidence of margin differences across top postal codes')
    else:
        print('PostalCode column not found — skipping zip code tests')

    # H0: no significant risk difference between Women and Men
    if 'Gender' in df.columns:
        print('\nTest: Claim frequency between Genders (Chi-square)')
        # restrict to common gender labels
        genders = df['Gender'].fillna('MISSING')
        if len(genders.unique()) >= 2:
            chi2_g, p_g, ct_g = chi2_test_frequency(df, 'Gender')
            print('Chi2=%.3f, p=%.4g' % (chi2_g, p_g))
            if p_g < 0.05:
                print('RESULT: Reject H0 — claim frequency differs between genders (p < 0.05)')
            else:
                print('RESULT: Fail to reject H0 — no evidence of frequency differences between genders')

            print('\nTest: Claim severity between Genders (Mann-Whitney U)')
            male = df[df['Gender'] == 'Male']['ClaimSeverity'].dropna()
            female = df[df['Gender'] == 'Female']['ClaimSeverity'].dropna()
            if len(male) >= 10 and len(female) >= 10:
                stat_u, p_u = stats.mannwhitneyu(male, female, alternative='two-sided')
                print('Mann-Whitney U stat=%.3f, p=%.4g' % (stat_u, p_u))
                if p_u < 0.05:
                    print('RESULT: Reject H0 — claim severity differs between genders (p < 0.05)')
                else:
                    print('RESULT: Fail to reject H0 — no evidence of severity differences between genders')
            else:
                print('Insufficient gender-labeled claims for Mann-Whitney test')
        else:
            print('Not enough gender categories for chi-square test')
    else:
        print('Gender column not found — skipping gender tests')


def main():
    p = find_input()
    if p is None:
        print('No input data found under data/ — please run data pipeline first')
        return
    print('Loading', p)
    df = load_data(p)
    df = prepare(df)
    # Run tests and print results
    run_tests(df)


if __name__ == '__main__':
    main()
