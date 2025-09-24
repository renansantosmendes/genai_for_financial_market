import matplotlib.pyplot as plt
import numpy as np
from utils import to_numpy
from config import SEQ_LEN

def plot_generated_series_comparison(prices, generated_list):
    """Plota o histórico real e as séries geradas."""
    plt.figure(figsize=(14, 6))
    LAST_N = 300
    start_idx = max(0, len(prices) - LAST_N)
    plt.plot(range(start_idx, len(prices)), prices[start_idx:].flatten(), label="Histórico Real (fechamento)", linewidth=1.2)

    base_index = len(prices)

    for i, (g_orig, temp) in enumerate(generated_list):
        gen_part = g_orig[SEQ_LEN:]
        xs = range(base_index, base_index + len(gen_part))
        plt.plot(xs, gen_part, label=f"Série Gerada {i+1} (temp={temp})", alpha=0.9)

    plt.axvline(x=base_index, color='gray', linestyle='--', linewidth=0.8)
    plt.title(f"Séries sintéticas geradas (Transformer probabilístico)")
    plt.xlabel("Índice temporal (dias)")
    plt.ylabel("Preço (R$)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_zoom_comparison(generated_list, scaler, init_series_scaled):
    """Plota um zoom no início das gerações."""
    plt.figure(figsize=(14, 6))
    zoom_range = 120
    for i, (g_orig, temp) in enumerate(generated_list):
        gen_part = g_orig[SEQ_LEN:SEQ_LEN + zoom_range]
        xs = range(0, len(gen_part))
        plt.plot(xs, gen_part, label=f"Gerada {i+1} (temp={temp})")

    init_orig = scaler.inverse_transform(init_series_scaled.reshape(-1, 1)).flatten()
    plt.plot(range(-SEQ_LEN, 0), init_orig, label="Sequência inicial (real, janela)", color='black', linewidth=1.5)
    plt.title("Zoom: primeiras etapas das séries geradas (comparação)")
    plt.xlabel("Passos a partir do último ponto real")
    plt.ylabel("Preço (R$)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_generated_series(generated, n_series=5):
    """Plota as séries geradas."""
    plt.figure(figsize=(14, 10))
    for i in range(min(n_series, len(generated))):
        serie = generated[i]

        if isinstance(serie, (list, tuple)) and len(serie) >= 2:
            entrada, saida = to_numpy(serie[0]), to_numpy(serie[1])

            plt.subplot(n_series, 1, i + 1)
            if len(entrada) > 0:
                plt.plot(entrada, label=f"Entrada {i+1}", color="blue")
            if len(saida) > 0:
                plt.plot(range(len(entrada), len(entrada) + len(saida)),
                         saida, label=f"Saída Gerada {i+1}", color="purple")
        else:
            serie = to_numpy(serie)
            plt.subplot(n_series, 1, i + 1)
            plt.plot(serie, label=f"Série Gerada {i+1}", color="purple")

        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()
