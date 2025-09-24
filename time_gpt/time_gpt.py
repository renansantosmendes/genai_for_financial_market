import numpy as np
from config import (
    DEVICE, D_MODEL, NHEAD, NUM_LAYERS, SEQ_LEN, 
    FUTURE_STEPS, N_GENERATED_SERIES, TICKER
)
from data_process import get_data_loader, download_data
from generation import generate_series_stochastic
from modelos import ProbabilisticTransformer
from training import train_model
from utils import set_seed
from visualization import (
    plot_generated_series_comparison, 
    plot_zoom_comparison, 
    plot_generated_series
)

def main():
    """Função principal para executar o pipeline completo."""
    set_seed()
    print("Device:", DEVICE)

    # Carregar dados
    loader, scaler, prices_scaled = get_data_loader()
    prices, _ = download_data()

    # Inicializar e treinar o modelo
    model = ProbabilisticTransformer(d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS).to(DEVICE)
    train_model(model, loader)

    # Gerar séries temporais
    init_series_scaled = prices_scaled[-SEQ_LEN:]
    generated_scaled_list = []
    temperatures = [0.6, 1.0, 1.4, 2.0, 0.8]

    for i in range(N_GENERATED_SERIES):
        temp = temperatures[i % len(temperatures)]
        g = generate_series_stochastic(model, init_series_scaled, future_steps=FUTURE_STEPS, temp=temp)
        generated_scaled_list.append((g, temp))

    # Inverter a escala das séries geradas
    generated_list = []
    for g_scaled, temp in generated_scaled_list:
        g_2d = g_scaled.reshape(-1, 1)
        try:
            g_orig = scaler.inverse_transform(g_2d).flatten()
        except Exception as e:
            g_2d_clipped = np.clip(g_2d, 0.0, 1.0)
            g_orig = scaler.inverse_transform(g_2d_clipped).flatten()
        generated_list.append((g_orig, temp))

    # Visualizar os resultados
    plot_generated_series_comparison(prices, generated_list)
    plot_zoom_comparison(generated_list, scaler, init_series_scaled)
    plot_generated_series(generated_list, n_series=N_GENERATED_SERIES)

    # Observações finais
    print("Geração concluída. Dicas:")
    print("- Ajuste 'EPOCHS' para treinar mais (melhora qualidade).")
    print("- Aumentar 'D_MODEL' / 'NUM_LAYERS' melhora capacidade, mas exige mais dados/tempo.")
    print("- 'temperatures' controla variabilidade: maior -> mais volátil.")
    print("- Você pode aplicar clipping nos valores gerados antes de inverter a escala, se observar valores absurdos.")

if __name__ == "__main__":
    main()
