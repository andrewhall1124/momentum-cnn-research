from train import train
import datetime as dt
import sf_quant.data as sfd
import os
import polars as pl
import torch
from models import Returns1DCNN
import numpy as np

if __name__ == '__main__':
    start = dt.date(2000, 1, 1)
    end = dt.date(2024, 12, 31) 

    data = (
        sfd.load_assets(
            start=start,
            end=end,
            in_universe=True,
            columns=[
                'date',
                'barrid',
                'ticker',
                'price',
                'return'
            ]
        )
        .sort('barrid', 'date')
        .with_columns(
            pl.col('return').truediv(100)
        )
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
            # pl.col('date').is_between(dt.date(2024, 12, 1), dt.date(2024, 12, 4)),
            pl.col('price').gt(5),
            pl.all().is_not_null()
        )
        .sort('barrid', 'date')
    )

    dates = data['date'].unique().sort().to_list()

    # Train model for each date
    for date_ in dates:
        if f"best_model_{date_.strftime('%Y%m%d')}.pt" not in os.listdir('checkpoints'):
            train(date_)

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
        date_data = data.filter(pl.col('date') == date_)

        # Prepare input features (past 251 days of returns, excluding today)
        feature_cols = ['return'] + [f'return_{t}' for t in range(1, 252)]
        X = date_data.select(feature_cols).to_numpy().astype(np.float32)

        # Generate predictions
        with torch.no_grad():
            X_tensor = torch.tensor(X, device=device)
            predictions = model(X_tensor).cpu().numpy()

            # Show prediction statistics
            print(f"  Predictions for {len(predictions)} stocks:")
            print(f"    Mean: {predictions.mean():.6f}")
            print(f"    Std:  {predictions.std():.6f}")
            print(f"    Min:  {predictions.min():.6f}")
            print(f"    Max:  {predictions.max():.6f}")
            print(f"    Unique values: {len(np.unique(predictions))}")

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

    # Load and display predictions
    predictions = pl.read_parquet('predictions.parquet')
    print("\nSample predictions:")
    print(predictions.head(10))