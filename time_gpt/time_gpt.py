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
    """Main function to run the complete pipeline."""
    set_seed()
    print("Device:", DEVICE)

    loader, scaler, prices_scaled = get_data_loader()
    prices, _ = download_data()

    model = ProbabilisticTransformer(d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS).to(DEVICE)
    train_model(model, loader)

    init_series_scaled = prices_scaled[-SEQ_LEN:]
    generated_scaled_list = []
    temperatures = [0.6, 1.0, 1.4, 2.0, 0.8]

    for i in range(N_GENERATED_SERIES):
        temp = temperatures[i % len(temperatures)]
        g = generate_series_stochastic(model, init_series_scaled, future_steps=FUTURE_STEPS, temp=temp)
        generated_scaled_list.append((g, temp))

    generated_list = []
    for g_scaled, temp in generated_scaled_list:
        g_2d = g_scaled.reshape(-1, 1)
        try:
            g_orig = scaler.inverse_transform(g_2d).flatten()
        except Exception as e:
            g_2d_clipped = np.clip(g_2d, 0.0, 1.0)
            g_orig = scaler.inverse_transform(g_2d_clipped).flatten()
        generated_list.append((g_orig, temp))

    plot_generated_series_comparison(prices, generated_list)
    plot_zoom_comparison(generated_list, scaler, init_series_scaled)
    plot_generated_series(generated_list, n_series=N_GENERATED_SERIES)

    print("Generation complete. Tips:")
    print("- Adjust 'EPOCHS' to train more (improves quality).")
    print("- Increasing 'D_MODEL' / 'NUM_LAYERS' improves capacity, but requires more data/time.")
    print("- 'temperatures' controls variability: higher -> more volatile.")
    print("- You can apply clipping to the generated values before inverting the scale, if you observe absurd values.")

if __name__ == "__main__":
    main()