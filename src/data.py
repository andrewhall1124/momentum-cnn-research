import sf_quant.data as sfd
import datetime as dt
import polars as pl
import os
from tqdm import tqdm

os.makedirs('data', exist_ok=True)

start = dt.date(2000, 1, 1)
end = dt.date(2024, 12, 31)

columns = [
    'date',
    'barrid',
    'ticker',
    'price',
    'return'
]

data = sfd.load_assets(
    start=start,
    end=end,
    columns=columns,
    in_universe=True
)

print(data)

years = data['date'].dt.year().unique().sort().to_list()

for year in tqdm(years, desc="Generating dataset"):
    df_year = data.filter(
        pl.col('date').dt.year().is_between(year - 1, year + 1)
    )

    df_year_clean = (
        df_year
        .with_columns(
            [
                pl.col('return').shift(t).over('barrid').alias(f"return_{t}")
                for t in range(1, 253)
            ]
        )
        .with_columns(
            pl.col('return').shift(-1).over('barrid').alias('fwd_return')
        )
        .filter(
            pl.col('date').dt.year().eq(year),
            pl.col('price').gt(5),
            pl.all().is_not_null()
        )
        .sort('barrid', 'date')
    )

    df_year_clean.write_parquet(f'data/data_{year}.parquet')
