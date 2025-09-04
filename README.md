# GenAI for Financial Market

This project uses a Variational Autoencoder (VAE) to generate synthetic financial time series data.

## Structure

- `genai_for_financial_market/`
  - `vae/`
    - `__init__.py`: Makes the `vae` directory a Python package.
    - `main.py`: The main script to train the VAE and generate synthetic data.
    - `visualization.py`: A module for visualizing the generated data.
  - `time_gan/`: Directory for the TimeGAN model.
  - `.gitignore`: Git ignore file.

## How to use

To train the VAE and generate synthetic data, run the following command:

```bash
python -m genai_for_financial_market.vae.main
```
