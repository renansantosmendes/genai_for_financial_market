import yfinance as yf
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from config import TICKER, START, END, SEQ_LEN, BATCH_SIZE

def download_data():
    """Downloads the data for the specified asset."""
    print("Downloading data:", TICKER)
    df = yf.download(TICKER, start=START, end=END, progress=False)
    if df.empty:
        raise RuntimeError("No data returned from yfinance. Check the ticker and connection.")
    prices = df["Close"].values.reshape(-1, 1)
    dates = df.index.to_numpy()
    return prices, dates

def create_sequences(series, seq_len):
    """Creates data sequences for the model."""
    X, Y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        Y.append(series[i+1:i+seq_len+1])
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return torch.tensor(X), torch.tensor(Y)

def get_data_loader():
    """Returns the DataLoader with the training data."""
    prices, _ = download_data()
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices).flatten()

    X, Y = create_sequences(prices_scaled, SEQ_LEN)
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    print(f"Total points: {len(prices_scaled)}, sequences: {len(X)}")
    return loader, scaler, prices_scaled