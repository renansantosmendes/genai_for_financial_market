from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from .models import (
    TimeSeriesGenerator,
    TimeSeriesSupervisor,
    TimeSeriesRecovery,
    TimeSeriesEmbedder,
)
from .config import Config


class NoiseGenerator:
    """Utility class for generating random noise for the generator."""

    @staticmethod
    def sample_random_noise(batch_size: int,
                           sequence_length: int,
                           noise_dimension: int,
                           device: torch.device,
                           random_seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate random noise tensor for the generator.

        Args:
            batch_size: Size of the batch
            sequence_length: Length of sequences
            noise_dimension: Dimension of noise vectors
            device: Device to place the tensor on
            random_seed: Optional seed for reproducibility

        Returns:
            Random noise tensor
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)

        return torch.randn(batch_size, sequence_length, noise_dimension, device=device)


class SyntheticDataGenerator:
    """Generate synthetic time series data using trained TimeGAN."""

    def __init__(self,
                 embedder: TimeSeriesEmbedder,
                 recovery: TimeSeriesRecovery,
                 generator: TimeSeriesGenerator,
                 supervisor: TimeSeriesSupervisor,
                 data_scaler: MinMaxScaler,
                 device: torch.device,
                 hidden_dimension: int):
        """
        Initialize the synthetic data generator.

        Args:
            embedder: Trained embedder network
            recovery: Trained recovery network
            generator: Trained generator network
            supervisor: Trained supervisor network
            data_scaler: Fitted data scaler for denormalization
            device: Device for computations
            hidden_dimension: Dimension of the hidden space
        """
        self.embedder = embedder
        self.recovery = recovery
        self.generator = generator
        self.supervisor = supervisor
        self.data_scaler = data_scaler
        self.device = device
        self.noise_dimension = hidden_dimension

    def generate_synthetic_sequences(self,
                                   num_samples: int,
                                   sequence_length: int) -> np.ndarray:
        """
        Generate synthetic time series sequences.

        Args:
            num_samples: Number of sequences to generate
            sequence_length: Length of each sequence

        Returns:
            Array of synthetic sequences (denormalized)
        """
        print(f"Generating {num_samples} synthetic sequences...")

        self.embedder.eval()
        self.recovery.eval()
        self.generator.eval()
        self.supervisor.eval()

        with torch.no_grad():
            noise_input = NoiseGenerator.sample_random_noise(
                num_samples, sequence_length, self.noise_dimension, self.device
            )
            fake_hidden_states = self.generator(noise_input)
            supervised_fake_states = self.supervisor(fake_hidden_states)
            synthetic_sequences = self.recovery(supervised_fake_states)

            synthetic_data = synthetic_sequences.cpu().numpy()

        num_features = synthetic_data.shape[2]
        synthetic_2d = synthetic_data.reshape(-1, num_features)
        denormalized_2d = self.data_scaler.inverse_transform(synthetic_2d)
        denormalized_sequences = denormalized_2d.reshape(
            num_samples, sequence_length, num_features
        )

        print(f"Synthetic sequences generated: {denormalized_sequences.shape}")
        return denormalized_sequences

    def generate_diverse_synthetic_batch(self,
                                       num_samples: int,
                                       sequence_length: int,
                                       batch_size: int = 64,
                                       add_noise: bool = True,
                                       temperature: float = 1.0) -> np.ndarray:
        """
        Generate diverse synthetic sequences with enhanced variation.

        Args:
            num_samples: Number of sequences to generate
            sequence_length: Length of each sequence
            batch_size: Size of generation batches
            add_noise: Whether to add extra noise for diversity
            temperature: Temperature scaling for noise (higher = more diverse)

        Returns:
            Array of diverse synthetic sequences
        """
        print(f"Generating {num_samples} diverse synthetic sequences...")
        print(f"  Temperature: {temperature}")
        print(f"  Add noise: {add_noise}")

        if hasattr(self.generator, 'training'):
            self.generator.train()
        if hasattr(self.supervisor, 'training'):
            self.supervisor.train()

        all_synthetic_sequences = []

        np.random.seed(None)
        torch.manual_seed(np.random.randint(0, 10000))

        for batch_start in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - batch_start)

            with torch.no_grad():
                batch_seed = np.random.randint(0, 100000)
                noise_input = NoiseGenerator.sample_random_noise(
                    current_batch_size, sequence_length,
                    self.noise_dimension, self.device
                )

                noise_input = noise_input * temperature

                if add_noise:
                    extra_noise = torch.randn_like(noise_input) * 0.1
                    noise_input = noise_input + extra_noise

                fake_hidden_states = self.generator(noise_input)

                if add_noise:
                    hidden_noise = torch.randn_like(fake_hidden_states) * 0.05
                    fake_hidden_states = fake_hidden_states + hidden_noise

                supervised_fake_states = self.supervisor(fake_hidden_states)
                synthetic_batch = self.recovery(supervised_fake_states)

                if add_noise:
                    final_noise = torch.randn_like(synthetic_batch) * 0.01
                    synthetic_batch = synthetic_batch + final_noise

                all_synthetic_sequences.append(synthetic_batch.cpu().numpy())

                if (batch_start // batch_size + 1) % 10 == 0:
                    batch_num = batch_start // batch_size + 1
                    total_batches = (num_samples - 1) // batch_size + 1
                    print(f"  Batch {batch_num}/{total_batches} completed")

        synthetic_data = np.concatenate(all_synthetic_sequences, axis=0)

        print(f"  Variance of synthetic data (normalized): {synthetic_data.var():.6f}")
        print(f"  Range of synthetic data: {synthetic_data.min():.6f} to {synthetic_data.max():.6f}")

        num_features = synthetic_data.shape[2]
        synthetic_2d = synthetic_data.reshape(-1, num_features)
        denormalized_2d = self.data_scaler.inverse_transform(synthetic_2d)
        denormalized_sequences = denormalized_2d.reshape(
            num_samples, sequence_length, num_features
        )

        print(f"  Variance after denormalization: {denormalized_sequences.var():.6f}")
        print(f"  Range after denormalization: {denormalized_sequences.min():.6f} to {denormalized_sequences.max():.6f}")

        return denormalized_sequences


class ModelDiversityChecker:
    """Check if TimeGAN models generate diverse outputs."""

    def __init__(self,
                 generator: TimeSeriesGenerator,
                 supervisor: TimeSeriesSupervisor,
                 recovery: TimeSeriesRecovery,
                 device: torch.device,
                 sequence_length: int,
                 hidden_dimension: int):
        """
        Initialize the diversity checker.

        Args:
            generator: Generator network
            supervisor: Supervisor network
            recovery: Recovery network
            device: Device for computations
            sequence_length: Length of the sequences
            hidden_dimension: Dimension of the hidden space
        """
        self.generator = generator
        self.supervisor = supervisor
        self.recovery = recovery
        self.device = device
        self.sequence_length = sequence_length
        self.noise_dimension = hidden_dimension

    def check_model_diversity(self) -> None:
        """Check if models generate diverse outputs."""
        print("\n" + "=" * 50)
        print("MODEL DIVERSITY VERIFICATION")
        print("=" * 50)

        print("1. Testing with identical noise...")
        test_noise = torch.randn(5, self.sequence_length, self.noise_dimension, device=self.device)
        repeated_noise = test_noise.repeat(1, 1, 1)

        with torch.no_grad():
            self.generator.eval()
            self.supervisor.eval()
            self.recovery.eval()

            hidden_states_1 = self.generator(repeated_noise)
            supervised_states_1 = self.supervisor(hidden_states_1)
            sequences_1 = self.recovery(supervised_states_1)

            print(f"  Variance with identical noise: {sequences_1.var().item():.6f}")
            print(f"  Maximum difference between sequences: {(sequences_1.max() - sequences_1.min()).item():.6f}")

        print("\n2. Testing with different noise...")
        different_noise = torch.randn(5, self.sequence_length, self.noise_dimension, device=self.device)

        with torch.no_grad():
            hidden_states_2 = self.generator(different_noise)
            supervised_states_2 = self.supervisor(hidden_states_2)
            sequences_2 = self.recovery(supervised_states_2)

            print(f"  Variance with different noise: {sequences_2.var().item():.6f}")
            print(f"  Maximum difference between sequences: {(sequences_2.max() - sequences_2.min()).item():.6f}")

        print("\n3. Comparison:")
        with torch.no_grad():
            difference = torch.abs(sequences_1 - sequences_2).mean()
            print(f"  Mean difference between the two cases: {difference.item():.6f}")

        print("=" * 50)

def generate_and_select_best_data(
    synthetic_generator: SyntheticDataGenerator,
    config: Config,
    num_synthetic_series: int = 100
) -> np.ndarray:
    """
    Generates synthetic data by testing different configurations and selecting the best one.

    Args:
        synthetic_generator (SyntheticDataGenerator): The data generator instance.
        config (Config): The configuration object.
        num_synthetic_series (int): The number of synthetic series to generate.

    Returns:
        np.ndarray: The generated synthetic data.
    """
    generation_configs = [
        {"add_noise": False, "temperature": 1.0, "name": "Standard"},
        {"add_noise": True, "temperature": 1.0, "name": "With extra noise"},
        {"add_noise": True, "temperature": 1.5, "name": "High temperature + noise"},
        {"add_noise": True, "temperature": 2.0, "name": "Very high temperature"},
    ]

    best_synthetic_data = None
    best_diversity_score = 0

    print("Testing different generation configurations...")
    for gen_config in generation_configs:
        print(f"\nTesting configuration: {gen_config['name']}")
        test_synthetic = synthetic_generator.generate_diverse_synthetic_batch(
            20, config.SEQUENCE_LENGTH,
            add_noise=gen_config["add_noise"],
            temperature=gen_config["temperature"]
        )

        diversity_score = np.var([series.var() for series in test_synthetic[:, :, 0]])
        print(f"  Diversity score (variance of variances): {diversity_score:.6f}")

        first_series = test_synthetic[0, :, 0]
        identical_count = 0
        for series_idx in range(1, min(10, len(test_synthetic))):
            if np.allclose(first_series, test_synthetic[series_idx, :, 0], rtol=1e-3):
                identical_count += 1

        print(f"  Series approximately identical to first: {identical_count}/9")

        if diversity_score > best_diversity_score:
            best_diversity_score = diversity_score
            best_synthetic_data = test_synthetic
            print(f"  ✓ Best configuration so far!")

    print(f"\nGenerating {num_synthetic_series} synthetic series with best configuration...")

    if best_synthetic_data is not None:
        final_synthetic_data = synthetic_generator.generate_diverse_synthetic_batch(
            num_synthetic_series, config.SEQUENCE_LENGTH,
            add_noise=True, temperature=1.5
        )
    else:
        print("Using standard configuration...")
        final_synthetic_data = synthetic_generator.generate_synthetic_sequences(
            num_synthetic_series, config.SEQUENCE_LENGTH
        )
    
    return final_synthetic_data