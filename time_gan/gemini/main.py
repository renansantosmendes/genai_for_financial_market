from typing import List, Optional

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from timegan_refactored.config import Config
from timegan_refactored.utils import setup_random_seeds, get_device
from timegan_refactored.data_processing import (
    FinancialDataDownloader,
    process_and_prepare_data,
)
from timegan_refactored.models import (
    TimeSeriesEmbedder,
    TimeSeriesRecovery,
    TimeSeriesGenerator,
    TimeSeriesSupervisor,
    TimeSeriesDiscriminator,
)
from timegan_refactored.training import TimeGANTrainer
from timegan_refactored.generation import (
    SyntheticDataGenerator,
    ModelDiversityChecker,
    generate_and_select_best_data,
)
from timegan_refactored.analysis import analyze_and_visualize_results


def main() -> None:
    """Main execution function for TimeGAN training and generation."""

    print("Setting up TimeGAN training environment...")
    setup_random_seeds(Config.RANDOM_SEED)
    device = get_device()

    print("\n" + "=" * 50)
    print("DATA COLLECTION AND PREPROCESSING")
    print("=" * 50)

    data_downloader = FinancialDataDownloader()
    try:
        raw_financial_data = data_downloader.download_stock_data(
            Config.STOCK_TICKERS, Config.DATA_START_DATE, Config.DATA_END_DATE
        )
        
        train_dataloader, _, data_scaler, num_features, train_sequences = process_and_prepare_data(
            raw_financial_data, Config
        )

    except Exception as e:
        print(f"An error occurred during data processing: {e}")
        return

    print("\n" + "=" * 50)
    print("INITIALIZING TIMEGAN MODELS")
    print("=" * 50)

    input_dimension = num_features
    hidden_dimension = Config.HIDDEN_DIMENSION
    noise_dimension = hidden_dimension

    embedder_network = TimeSeriesEmbedder(input_dimension, hidden_dimension, Config.RNN_LAYERS_COUNT).to(device)
    recovery_network = TimeSeriesRecovery(hidden_dimension, input_dimension, Config.RNN_LAYERS_COUNT).to(device)
    generator_network = TimeSeriesGenerator(noise_dimension, hidden_dimension, Config.RNN_LAYERS_COUNT).to(device)
    supervisor_network = TimeSeriesSupervisor(hidden_dimension, Config.RNN_LAYERS_COUNT).to(device)
    discriminator_network = TimeSeriesDiscriminator(hidden_dimension, Config.RNN_LAYERS_COUNT).to(device)

    print(f"Models initialized with:")
    print(f"  Input dimension: {input_dimension}")
    print(f"  Hidden dimension: {hidden_dimension}")
    print(f"  Number of layers: {Config.RNN_LAYERS_COUNT}")

    timegan_trainer = TimeGANTrainer(
        embedder=embedder_network,
        recovery=recovery_network,
        generator=generator_network,
        supervisor=supervisor_network,
        discriminator=discriminator_network,
        device=device,
        learning_rate=Config.LEARNING_RATE,
        hidden_dimension=Config.HIDDEN_DIMENSION,
        generator_training_steps=Config.GENERATOR_TRAINING_STEPS,
        discriminator_training_steps=Config.DISCRIMINATOR_TRAINING_STEPS,
        supervised_loss_weight=Config.SUPERVISED_LOSS_WEIGHT,
        moment_matching_weight=Config.MOMENT_MATCHING_WEIGHT,
    )

    print("\n" + "=" * 50)
    print("TIMEGAN TRAINING PHASES")
    print("=" * 50)

    print(f"\nPhase 1/3: Autoencoder Pre-training ({Config.AUTOENCODER_PRETRAIN_EPOCHS} epochs)...")
    timegan_trainer.train_autoencoder_pretraining(train_dataloader, Config.AUTOENCODER_PRETRAIN_EPOCHS)

    print(f"\nPhase 2/3: Supervisor Pre-training ({Config.SUPERVISOR_PRETRAIN_EPOCHS} epochs)...")
    timegan_trainer.train_supervisor_pretraining(train_dataloader, Config.SUPERVISOR_PRETRAIN_EPOCHS)

    print(f"\nPhase 3/3: Joint Training ({Config.JOINT_TRAINING_EPOCHS} epochs)...")
    timegan_trainer.train_joint_training(train_dataloader, Config.JOINT_TRAINING_EPOCHS)

    print("\n" + "=" * 50)
    print("MODEL QUALITY ASSESSMENT")
    print("=" * 50)

    diversity_checker = ModelDiversityChecker(
        generator=generator_network, 
        supervisor=supervisor_network, 
        recovery=recovery_network, 
        device=device,
        sequence_length=Config.SEQUENCE_LENGTH,
        hidden_dimension=Config.HIDDEN_DIMENSION,
    )
    diversity_checker.check_model_diversity()

    print("\n" + "=" * 50)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 50)

    synthetic_generator = SyntheticDataGenerator(
        embedder=embedder_network, 
        recovery=recovery_network, 
        generator=generator_network,
        supervisor=supervisor_network, 
        data_scaler=data_scaler, 
        device=device,
        hidden_dimension=Config.HIDDEN_DIMENSION,
    )

    final_synthetic_data = generate_and_select_best_data(
        synthetic_generator, Config, num_synthetic_series=100
    )

    analyze_and_visualize_results(
        final_synthetic_data,
        train_sequences,
        data_scaler,
        num_features,
        Config
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED!")
    print("=" * 50)
    print("Recommendations for improvement:")
    print("1. Increase JOINT_TRAINING_EPOCHS for better quality")
    print("2. Adjust HIDDEN_DIMENSION and RNN_LAYERS_COUNT")
    print("3. Experiment with different features beyond 'Close'")
    print("4. Implement more sophisticated metrics (Discriminative Score, Predictive Score)")
    print("5. Evaluate using TSTR (Train on Synthetic, Test on Real)")
    print("\nTips for improving diversity:")
    print("1. Verify training convergence")
    print("2. Increase temperature during generation")
    print("3. Add dropout to models during generation")
    print("4. Use different seeds for each series")
    print("5. Implement advanced sampling techniques")


if __name__ == "__main__":
    main()