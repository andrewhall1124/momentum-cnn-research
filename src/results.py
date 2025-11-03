import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl

df = pl.read_parquet('predictions.parquet')

labels = [str(i) for i in range(10)]
df_portfolios = (
    df
    .with_columns(
        pl.col('predicted_return').rank(method='random', seed=42).qcut(10, labels=labels).cast(pl.String).over('date').alias('bin')
    )
)

print(df_portfolios)

df_returns = (
    df_portfolios
    .group_by('date', 'bin')
    .agg(
        pl.col('fwd_return').mean().alias('return')
    )
    .sort('date', 'bin')
    .pivot(index='date', on='bin', values='return')
    .with_columns(
        pl.col('9').sub(pl.col('0')).alias('spread')
    )
)

print(df_returns)

df_cumulative_returns = (
    df_returns
    .with_columns(
        pl.exclude('date').log1p().cum_sum()
    )
)

print(df_cumulative_returns)

plt.figure(figsize=(10, 6))

colors = sns.color_palette(palette='coolwarm', n_colors=len(labels))

for label, color in zip(labels, colors):
    sns.lineplot(df_cumulative_returns, x='date', y=label, color=color)

sns.lineplot(df_cumulative_returns, x='date', y='spread', color='green')

plt.title('Backtest')
plt.xlabel(None)
plt.ylabel('Cumulative Log Return (%)')

plt.savefig('test.png', dpi=300)
