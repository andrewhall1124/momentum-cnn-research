from train import train
import datetime as dt
import sf_quant.data as sfd
import os
import polars as pl
import torch
from models import Returns1DCNN
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    start = dt.date(2000, 1, 1)
    end = dt.date(2024, 12, 31) 

    dates = (
        sfd.load_assets(
            start=start,
            end=end,
            columns='date',
            in_universe=True
        )
        .filter(
            pl.col('date').is_between(dt.date(2024, 12, 1), dt.date(2024, 12, 31)),
        )
        ['date'].unique().sort().to_list()
    )

    # Train model for each date
    for date_ in dates:
        print("Training model for date:", date_)
        if f"best_model_{date_.strftime('%Y%m%d')}.pt" not in os.listdir('checkpoints'):
            train(date_, logging=True)
