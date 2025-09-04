# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import acf

def plot_training_curves(losses, recon_losses, kld_losses):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(losses)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)

    axes[0, 1].plot(recon_losses, color='orange')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].grid(True)

    axes[1, 0].plot(kld_losses, color='red')
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KLD')
    axes[1, 0].grid(True)

    axes[1, 1].plot(recon_losses, label='Reconstruction', color='orange')
    axes[1, 1].plot(kld_losses, label='KL Divergence', color='red')
    axes[1, 1].set_title('Loss Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

def compare_distributions(real_data, synthetic_data, feature_names):
    """Compare distributions of real vs synthetic data."""
    n_features = min(4, len(feature_names))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(n_features):
        if i < len(feature_names):
            real_flat = real_data[:, :, i].flatten()
            synthetic_flat = synthetic_data[:, :, i].flatten()

            axes[i].hist(real_flat, alpha=0.5, label='Real', density=True, bins=50)
            axes[i].hist(synthetic_flat, alpha=0.5, label='Synthetic', density=True, bins=50)
            axes[i].set_title(f'Distribution: {feature_names[i]}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def create_candlestick_chart(data, title="Candlestick Data"):
    """Create candlestick chart using plotly."""
    fig = go.Figure(data=go.Candlestick(
        x=list(range(len(data))),
        open=data[:, 0],
        high=data[:, 1],
        low=data[:, 2],
        close=data[:, 3],
        name=title
    ))

    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_title='Time',
        template='plotly_white'
    )

    return fig

def plot_candlestick_grid(synthetic_dfs, n_plots=6, title="Synthetic Series - Candlestick"):
    """Create grid of candlestick charts."""
    n_plots = min(n_plots, len(synthetic_dfs))

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f'Series {i+1}' for i in range(n_plots)],
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )

    for i in range(n_plots):
        row = (i // 3) + 1
        col = (i % 3) + 1

        df = synthetic_dfs[i]

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=f'Series {i+1}',
                showlegend=False
            ),
            row=row, col=col
        )

    fig.update_layout(
        title=title,
        height=800,
        template='plotly_white'
    )

    return fig

def plot_price_evolution(synthetic_dfs, n_series=10):
    """Plot closing price evolution."""
    fig = go.Figure()
    colors = px.colors.qualitative.Set3

    for i, df in enumerate(synthetic_dfs[:n_series]):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name=f'Series {i+1}',
            line=dict(color=colors[i % len(colors)], width=1.5),
            opacity=0.8
        ))

    fig.update_layout(
        title=f'Closing Price Evolution - {n_series} Synthetic Series',
        xaxis_title='Date',
        yaxis_title='Closing Price',
        template='plotly_white',
        height=600
    )

    return fig

def plot_volume_analysis(synthetic_dfs, n_series=8):
    """Analyze volume characteristics of synthetic series."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    all_volumes = []
    daily_avg_volumes = []

    for df in synthetic_dfs[:n_series]:
        all_volumes.extend(df['Volume'].tolist())
        daily_avg_volumes.append(df['Volume'].mean())

    axes[0, 0].hist(all_volumes, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Volume Distribution - All Series')
    axes[0, 0].set_xlabel('Volume')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(range(1, len(daily_avg_volumes)+1), daily_avg_volumes,
                   color='lightcoral', alpha=0.8)
    axes[0, 1].set_title('Average Volume per Synthetic Series')
    axes[0, 1].set_xlabel('Series')
    axes[0, 1].set_ylabel('Average Volume')
    axes[0, 1].grid(True, alpha=0.3)

    price_vol_corr = []
    for df in synthetic_dfs[:n_series]:
        corr = df['Close'].corr(df['Volume'])
        price_vol_corr.append(corr)

    axes[1, 0].bar(range(1, len(price_vol_corr)+1), price_vol_corr,
                   color='gold', alpha=0.8)
    axes[1, 0].set_title('Price-Volume Correlation per Series')
    axes[1, 0].set_xlabel('Series')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)

    volume_data = [df['Volume'].values for df in synthetic_dfs[:n_series]]
    axes[1, 1].boxplot(volume_data, labels=[f'S{i+1}' for i in range(len(volume_data))])
    axes[1, 1].set_title('Volume Distribution by Series')
    axes[1, 1].set_xlabel('Series')
    axes[1, 1].set_ylabel('Volume')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_technical_indicators(synthetic_dfs, series_idx=0):
    """Plot technical indicators for a specific series."""
    df = synthetic_dfs[series_idx]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Price and Moving Averages', 'RSI',
            'MACD', 'Bollinger Bands',
            'Volatility', 'Volume'
        ],
        vertical_spacing=0.08,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_10'], name='SMA 10', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_30'], name='SMA 30', line=dict(color='green')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=1, col=2)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)

    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='red')), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='blue')), row=2, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='red')), row=2, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='green')), row=2, col=2)

    fig.add_trace(go.Scatter(x=df.index, y=df['Volatility'], name='Volatility', line=dict(color='orange')), row=3, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue'), row=3, col=2)

    fig.update_layout(
        title=f'Technical Indicators - Synthetic Series {series_idx + 1}',
        height=1000,
        template='plotly_white',
        showlegend=False
    )

    return fig

def plot_return_analysis(synthetic_dfs, n_series=10):
    """Detailed return analysis of synthetic series."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    all_returns = []
    all_log_returns = []

    for df in synthetic_dfs[:n_series]:
        all_returns.extend(df['Return'].dropna().tolist())
        all_log_returns.extend(df['Log_Return'].dropna().tolist())

    axes[0, 0].hist(all_returns, bins=50, alpha=0.7, color='lightblue', edgecolor='black', density=True)
    axes[0, 0].set_title('Return Distribution')
    axes[0, 0].set_xlabel('Return')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].grid(True, alpha=0.3)

    stats.probplot(all_returns, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot - Return Normality')
    axes[0, 1].grid(True, alpha=0.3)

    volatilities = [df['Return'].std() for df in synthetic_dfs[:n_series]]
    axes[0, 2].bar(range(1, len(volatilities)+1), volatilities, color='coral', alpha=0.8)
    axes[0, 2].set_title('Volatility per Series')
    axes[0, 2].set_xlabel('Series')
    axes[0, 2].set_ylabel('Volatility (Std)')
    axes[0, 2].grid(True, alpha=0.3)

    for i, df in enumerate(synthetic_dfs[:5]):
        cumret = (1 + df['Return'].fillna(0)).cumprod() - 1
        axes[1, 0].plot(cumret.index, cumret.values, label=f'Series {i+1}', alpha=0.8)
    axes[1, 0].set_title('Cumulative Returns')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Cumulative Return')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if len(all_returns) > 50:
        autocorr = acf(all_returns, nlags=20, fft=True)
        axes[1, 1].bar(range(len(autocorr)), autocorr, color='green', alpha=0.7)
        axes[1, 1].set_title('Return Autocorrelation')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].grid(True, alpha=0.3)

    sharpe_ratios = [df['Return'].mean() / df['Return'].std() if df['Return'].std() > 0 else 0
                     for df in synthetic_dfs[:n_series]]
    axes[1, 2].bar(range(1, len(sharpe_ratios)+1), sharpe_ratios, color='purple', alpha=0.8)
    axes[1, 2].set_title('Sharpe Ratio per Series')
    axes[1, 2].set_xlabel('Series')
    axes[1, 2].set_ylabel('Sharpe Ratio')
    axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
