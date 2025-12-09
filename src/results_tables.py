from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


def find_input():
    candidates = [
        Path('../data') / 'processed_sample_from_data_quality.csv',
        Path('../data') / 'processed_sample_from_notebook.csv',
        Path('../data') / 'MachineLearningRating_v3.txt'
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_data(p):
    sep = '|' if p.suffix == '.txt' else ','
    df = pd.read_csv(p, sep=sep, parse_dates=['TransactionMonth'], low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for c in ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '').str.strip(), errors='coerce')
    df['ClaimOccurred'] = (df['TotalClaims'].fillna(0) > 0).astype(int)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    df['ClaimSeverity'] = df['TotalClaims'].where(df['ClaimOccurred'] == 1)
    return df


def two_prop_ztest(k1, n1, k2, n2):
    # proportions k1/n1 and k2/n2, test H0: p1 = p2
    p1 = k1 / n1 if n1 else 0
    p2 = k2 / n2 if n2 else 0
    p_pool = (k1 + k2) / (n1 + n2) if (n1 + n2) else 0
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) if n1 and n2 else np.nan
    if se == 0 or np.isnan(se):
        return np.nan, np.nan
    z = (p1 - p2) / se
    pval = 2 * stats.norm.sf(abs(z))
    return z, pval


def summarize_group(df, group_col, top_n=None):
    # group_col expected in df
    grp = df.groupby(group_col).agg(
        policies=('PolicyID', 'nunique') if 'PolicyID' in df.columns else ('ClaimOccurred', 'count'),
        total_premium=('TotalPremium', 'sum'),
        total_claims=('TotalClaims', 'sum'),
        claims_count=('ClaimOccurred', 'sum')
    ).reset_index()
    # overall totals
    overall_policies = grp['policies'].sum()
    overall_claims = grp['claims_count'].sum()
    overall_premium = grp['total_premium'].sum()

    # Keep top_n if requested
    if top_n:
        grp = grp.sort_values('policies', ascending=False).head(top_n)

    results = []
    for _, row in grp.iterrows():
        name = row[group_col]
        n = row['policies']
        k = row['claims_count']
        tp = row['total_premium']
        tc = row['total_claims']
        loss_ratio = tc / tp if tp else np.nan
        claim_freq = k / n if n else np.nan
        # compare to rest
        rest_n = overall_policies - n
        rest_k = overall_claims - k
        z, p_freq = two_prop_ztest(k, n, rest_k, rest_n) if rest_n > 0 else (np.nan, np.nan)

        # severity test: compare ClaimSeverity for group vs rest using Mann-Whitney U
        # prepare series
        s_group = df[df[group_col] == name]['ClaimSeverity'].dropna()
        s_rest = df[df[group_col] != name]['ClaimSeverity'].dropna()
        if len(s_group) >= 10 and len(s_rest) >= 10:
            try:
                stat_u, p_sev = stats.mannwhitneyu(s_group, s_rest, alternative='two-sided')
            except Exception:
                stat_u, p_sev = np.nan, np.nan
        else:
            stat_u, p_sev = np.nan, np.nan

        results.append({
            group_col: name,
            'policies': n,
            'claims_count': k,
            'claim_freq': claim_freq,
            'total_premium': tp,
            'total_claims': tc,
            'loss_ratio': loss_ratio,
            'z_freq_vs_rest': z,
            'p_freq_vs_rest': p_freq,
            'mw_stat_severity_vs_rest': stat_u,
            'p_severity_vs_rest': p_sev
        })

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values('policies', ascending=False)
    return out_df


def run_all(df):
    outdir = Path('outputs') / 'results'
    outdir.mkdir(parents=True, exist_ok=True)

    # Province summary
    if 'Province' in df.columns:
        prov = summarize_group(df, 'Province')
        prov.to_csv(outdir / 'province_summary.csv', index=False)
        print('Wrote', outdir / 'province_summary.csv')

    # PostalCode top 10
    if 'PostalCode' in df.columns:
        pc = summarize_group(df, 'PostalCode', top_n=10)
        pc.to_csv(outdir / 'postalcode_top10_summary.csv', index=False)
        print('Wrote', outdir / 'postalcode_top10_summary.csv')

    # Make top 20
    if 'make' in df.columns:
        mk = summarize_group(df, 'make', top_n=20)
        mk.to_csv(outdir / 'make_summary_top20.csv', index=False)
        print('Wrote', outdir / 'make_summary_top20.csv')

    # Model top 20
    if 'Model' in df.columns:
        md = summarize_group(df, 'Model', top_n=20)
        md.to_csv(outdir / 'model_summary_top20.csv', index=False)
        print('Wrote', outdir / 'model_summary_top20.csv')


def main():
    p = find_input()
    if p is None:
        print('No input found; run data pipeline first')
        return
    print('Loading', p)
    df = load_data(p)
    print('Running aggregations and tests...')
    run_all(df)


if __name__ == '__main__':
    main()
