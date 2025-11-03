import torch
from torch.utils.data import Dataset
import sf_quant.data as sfd
import datetime as dt
import polars as pl
import os
from tqdm import tqdm

class DailyReturnsDataset(Dataset):
    """Dataset for stock returns with sliding windows."""

    def __init__(self, folder: str = 'data'):
        self.folder = folder
        self.data_start = dt.date(2000, 1, 1)
        self.data_end = dt.date(2024, 12, 31)

        # Generate expected years from full date range
        expected_years = list(range(self.data_start.year, self.data_end.year + 1))

        # Check if data folder exists, if not create and generate data
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        existing_files = os.listdir(folder)

        if len(expected_years) != len(existing_files):
            print(f"Generating data for years {self.data_start.year} to {self.data_end.year}...")

            columns = [
                'date',
                'barrid',
                'ticker',
                'price',
                'return'
            ]

            data = sfd.load_assets(
                start=self.data_start,
                end=self.data_end,
                columns=columns,
                in_universe=True
            ).with_columns(
                pl.col('return').truediv(100)
            )

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
                            for t in range(1, 252)
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

                df_year_clean.write_parquet(f'{folder}/returns_{year}.parquet')

        # Initialize empty - data will be loaded via load()
        self.data = None
        self.features = None
        self.targets = None

    def load(self, start: dt.date = dt.date(2000, 1, 1), end: dt.date = dt.date(2024, 12, 31)):
        # Validate date range
        if start < self.data_start or end > self.data_end:
            raise ValueError(
                f"Date range must be within {self.data_start} to {self.data_end}. "
                f"Got {start} to {end}."
            )
        if start > end:
            raise ValueError(f"Start date {start} must be before end date {end}.")

        # Determine years needed
        years_needed = list(range(start.year, end.year + 1))

        # Load all data into memory using polars
        parquet_files = [
            f'{self.folder}/returns_{year}.parquet'
            for year in years_needed
        ]

        # Read all files and filter by date range
        self.data = pl.read_parquet(parquet_files).filter(
            pl.col('date').is_between(start, end)
        )

        # Separate features and target
        feature_cols = [col for col in self.data.columns if col.startswith('return_')]
        self.features = self.data.select(feature_cols).to_numpy()
        self.targets = self.data.select('fwd_return').to_numpy().squeeze()

        return self

    def __len__(self):
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call load() method first.")
        return len(self.data)

    def __getitem__(self, idx):
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call load() method first.")
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)
