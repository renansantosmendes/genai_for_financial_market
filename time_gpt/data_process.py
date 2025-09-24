import yfinance as yf
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from config import TICKER, START, END, SEQ_LEN, BATCH_SIZE

def download_data():
    """Baixa os dados do ativo especificado."""
    print("Baixando dados:", TICKER)
    df = yf.download(TICKER, start=START, end=END, progress=False)
    if df.empty:
        raise RuntimeError("Nenhum dado retornado do yfinance. Verifique o ticker e a conexão.")
    prices = df["Close"].values.reshape(-1, 1)
    dates = df.index.to_numpy()
    return prices, dates

def create_sequences(series, seq_len):
    """Cria sequências de dados para o modelo."""
    X, Y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        Y.append(series[i+1:i+seq_len+1])  # alvo deslocado em 1
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return torch.tensor(X), torch.tensor(Y)

def get_data_loader():
    """Retorna o DataLoader com os dados de treinamento."""
    prices, _ = download_data()
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices).flatten()

    X, Y = create_sequences(prices_scaled, SEQ_LEN)
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    print(f"Total pontos: {len(prices_scaled)}, sequências: {len(X)}")
    return loader, scaler, prices_scaled
