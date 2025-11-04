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
            pl.col('date').is_between(dt.date(2024, 12, 1), dt.date(2024, 12, 30)),
        )
        ['date'].unique().sort().to_list()
    )

    # Predict returns for each date using saved models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} for predictions")
    predictions_list = []
    for date_ in dates:
        print(f"Generating predictions for {date_}...")

        # Load the model for this date
        checkpoint_path = f"checkpoints/best_model_{date_.strftime('%Y%m%d')}.pt"
        model = Returns1DCNN().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Get data for this date
        date_data = pl.read_parquet(f"data/returns_{date_.year}.parquet").filter(pl.col('date') == date_)

        # Prepare input features (past 251 days of returns, excluding today)
        feature_cols = ['return'] + [f'return_{t}' for t in range(1, 252)]
        X = date_data.select(feature_cols).to_numpy().astype(np.float32)

        # Generate predictions
        with torch.no_grad():
            X_tensor = torch.tensor(X, device=device)
            predictions = model(X_tensor).cpu().numpy()

            # Add predictions to dataframe
            date_predictions = date_data.select(['date', 'barrid', 'ticker', 'fwd_return']).with_columns(
                pl.Series('predicted_return', predictions)
            )

            predictions_list.append(date_predictions)

    # Combine all predictions
    all_predictions: pl.DataFrame = pl.concat(predictions_list)

    # Save predictions
    all_predictions.write_parquet('predictions.parquet')
    print("\nPredictions saved to predictions.parquet")
    print(f"Total predictions: {len(all_predictions)}")
