"""Exploratory Data Analysis plots focused on Risk and Profitability.

Produces and saves plots for LossRatio, distributions, bivariate relationships,
province-level loss ratio, outlier boxplots, and vehicle/gender claims breakdown.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_processed():
    # prefer processed sample saved by data_quality or the raw file
    p1 = Path('data') / 'processed_sample_from_data_quality.csv'
    p2 = Path('data') / 'processed_sample.csv'
    fallback = Path('data') / 'MachineLearningRating_v3.txt'
    if p1.exists():
        df = pd.read_csv(p1, parse_dates=['TransactionMonth'], low_memory=False)
    elif p2.exists():
        df = pd.read_csv(p2, parse_dates=['TransactionMonth'], low_memory=False)
    elif fallback.exists():
        df = pd.read_csv(fallback, sep='|', parse_dates=['TransactionMonth'], low_memory=False)
    else:
        raise FileNotFoundError('No input dataset found in data/. Run src/data_quality.py first.')
    return df


def prepare(df):
    # numeric conversions
    for c in ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '').str.strip(), errors='coerce')

    # Loss ratio
    if 'LossRatio' not in df.columns:
        df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace({0: pd.NA})

    return df


def ensure_outdir():
    out = Path('outputs') / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_univariate(df, outdir):
    sns.set(style='whitegrid', palette='muted')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if 'TotalPremium' in df.columns:
        sns.histplot(df['TotalPremium'].dropna(), ax=axes[0], bins=80)
        axes[0].set_title('Distribution of TotalPremium')
        axes[0].set_xlabel('TotalPremium')
    if 'TotalClaims' in df.columns:
        sns.histplot(df['TotalClaims'].dropna(), ax=axes[1], bins=80)
        axes[1].set_title('Distribution of TotalClaims')
        axes[1].set_xlabel('TotalClaims')
    plt.tight_layout()
    fig.savefig(outdir / 'univariate_premiums_claims.png', dpi=150)
    plt.close(fig)


def plot_bivariate(df, outdir):
    sns.set(style='white')
    plt.figure(figsize=(8, 6))
    hue_col = 'PostalCode' if 'PostalCode' in df.columns else 'Province' if 'Province' in df.columns else None
    if hue_col:
        sns.scatterplot(data=df, x='TotalPremium', y='TotalClaims', hue=hue_col, palette='tab10', alpha=0.6, s=30)
    else:
        sns.scatterplot(data=df, x='TotalPremium', y='TotalClaims', color='tab:blue', alpha=0.6, s=30)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('TotalPremium (log scale)')
    plt.ylabel('TotalClaims (log scale)')
    plt.title('TotalPremium vs TotalClaims')
    plt.tight_layout()
    plt.savefig(outdir / 'bivariate_premium_claims.png', dpi=150)
    plt.close()


def plot_province_lossratio(df, outdir):
    if 'Province' not in df.columns:
        print('Province column not present, skipping province plot')
        return
    agg = df.groupby('Province', dropna=True).apply(lambda g: pd.Series({
        'LossRatio': g['TotalClaims'].sum() / g['TotalPremium'].sum() if g['TotalPremium'].sum() else pd.NA,
        'count': len(g)
    }))
    agg = agg.sort_values('LossRatio', ascending=False).reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=agg, x='Province', y='LossRatio', palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average Loss Ratio (sum claims / sum premium)')
    plt.title('Loss Ratio by Province')
    plt.tight_layout()
    plt.savefig(outdir / 'lossratio_by_province.png', dpi=150)
    plt.close()


def plot_outliers(df, outdir):
    cols = [c for c in ['TotalClaims', 'TotalPremium', 'CustomValueEstimate'] if c in df.columns]
    if not cols:
        return
    plt.figure(figsize=(10, 6))
    df_long = df[cols].melt(var_name='variable', value_name='value')
    sns.boxplot(x='variable', y='value', data=df_long)
    plt.yscale('symlog')
    plt.title('Outlier boxplots (symlog scale)')
    plt.tight_layout()
    plt.savefig(outdir / 'boxplots_outliers.png', dpi=150)
    plt.close()


def plot_vehicle_gender(df, outdir):
    if 'VehicleType' not in df.columns or 'Gender' not in df.columns or 'TotalClaims' not in df.columns:
        print('Missing columns for vehicle/gender plot; skipping')
        return
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='VehicleType', y='TotalClaims', hue='Gender', data=df, showfliers=False)
    plt.yscale('symlog')
    plt.xticks(rotation=45, ha='right')
    plt.title('TotalClaims by VehicleType and Gender')
    plt.tight_layout()
    plt.savefig(outdir / 'claims_by_vehicle_gender.png', dpi=150)
    plt.close()


def main():
    df = load_processed()
    df = prepare(df)
    outdir = ensure_outdir()
    plot_univariate(df, outdir)
    plot_bivariate(df, outdir)
    plot_province_lossratio(df, outdir)
    plot_outliers(df, outdir)
    plot_vehicle_gender(df, outdir)
    print('Plots saved to', outdir)


if __name__ == '__main__':
    main()
