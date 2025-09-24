"""
TimeGAN implementation for synthetic time series generation.

This module implements TimeGAN (Time-series Generative Adversarial Networks)
for generating synthetic financial time series data.

Original research: "Time-series Generative Adversarial Networks" by Jinsung Yoon et al.
"""

import os
import math
import random
import time
from typing import List, Tuple, Optional, Dict, Any, Union
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import stats
import seaborn as sns

warnings.filterwarnings('ignore')

RANDOM_SEED: int = 42
STOCK_TICKERS: List[str] = ["AAPL", "MSFT", "GOOGL"]
DATA_START_DATE: str = "2018-01-01"
DATA_END_DATE: Optional[str] = None
SEQUENCE_LENGTH: int = 24
TARGET_FEATURES: List[str] = ["Close"]
TRAINING_BATCH_SIZE: int = 64
HIDDEN_DIMENSION: int = 24
RNN_LAYERS_COUNT: int = 2
LEARNING_RATE: float = 1e-3

AUTOENCODER_PRETRAIN_EPOCHS: int = 30
SUPERVISOR_PRETRAIN_EPOCHS: int = 30
JOINT_TRAINING_EPOCHS: int = 100

GENERATOR_TRAINING_STEPS: int = 1
DISCRIMINATOR_TRAINING_STEPS: int = 1

SUPERVISED_LOSS_WEIGHT: float = 1.0
MOMENT_MATCHING_WEIGHT: float = 1e-4


def setup_random_seeds(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value for reproducible results
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Get the available device for PyTorch computations.

    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


class FinancialDataDownloader:
    """Handle downloading and preprocessing of financial data."""

    def __init__(self,
                 max_retry_attempts: int = 3,
                 retry_sleep_time: int = 2):
        """
        Initialize the financial data downloader.

        Args:
            max_retry_attempts: Maximum number of download retry attempts
            retry_sleep_time: Sleep time between retry attempts in seconds
        """
        self.max_retry_attempts = max_retry_attempts
        self.retry_sleep_time = retry_sleep_time

    def download_stock_data(self,
                           ticker_symbols: List[str],
                           start_date: str,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Download stock data from Yahoo Finance with robust error handling.

        Args:
            ticker_symbols: List of stock ticker symbols to download
            start_date: Start date for data download in YYYY-MM-DD format
            end_date: End date for data download in YYYY-MM-DD format (optional)

        Returns:
            Combined DataFrame with stock data for all tickers

        Raises:
            RuntimeError: If no valid data could be downloaded
        """
        downloaded_dataframes = []

        for ticker_symbol in ticker_symbols:
            ticker_data = self._download_single_ticker(
                ticker_symbol, start_date, end_date
            )
            if ticker_data is not None:
                downloaded_dataframes.append(ticker_data)

        if not downloaded_dataframes:
            raise RuntimeError(
                "No valid data downloaded. Please check ticker symbols and connection."
            )

        combined_data = pd.concat(downloaded_dataframes, ignore_index=False)
        print(f"Total records downloaded: {len(combined_data)}")
        return combined_data

    def _download_single_ticker(self,
                               ticker_symbol: str,
                               start_date: str,
                               end_date: Optional[str]) -> Optional[pd.DataFrame]:
        """
        Download data for a single ticker with retry logic.

        Args:
            ticker_symbol: Stock ticker symbol
            start_date: Start date for data download
            end_date: End date for data download

        Returns:
            DataFrame with ticker data or None if download failed
        """
        for attempt_number in range(self.max_retry_attempts):
            try:
                print(
                    f"Downloading {ticker_symbol} "
                    f"(attempt {attempt_number + 1}/{self.max_retry_attempts})..."
                )

                ticker_dataframe = yf.download(
                    ticker_symbol,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False
                )

                if ticker_dataframe.empty:
                    print(f"Warning: No data available for {ticker_symbol}")
                    break

                if ticker_dataframe.index.tz is None:
                    ticker_dataframe.index = ticker_dataframe.index.tz_localize("UTC")

                if isinstance(ticker_dataframe.columns, pd.MultiIndex):
                    ticker_dataframe.columns = ticker_dataframe.columns.droplevel(1)

                ticker_dataframe["Ticker"] = ticker_symbol
                ticker_dataframe = ticker_dataframe.dropna()

                if len(ticker_dataframe) > 0:
                    return ticker_dataframe
                else:
                    print(f"Warning: {ticker_symbol} has no valid data after cleaning")

                break  # Exit retry loop on successful download

            except Exception as download_error:
                print(
                    f"Error downloading {ticker_symbol} "
                    f"(attempt {attempt_number + 1}): {download_error}"
                )
                if attempt_number < self.max_retry_attempts - 1:
                    time.sleep(self.retry_sleep_time)
                else:
                    print(f"Failed to download {ticker_symbol}, skipping...")

        return None


class TimeSeriesSequenceGenerator:
    """Generate sequences from time series data for training."""

    def __init__(self, sequence_length: int, target_features: List[str]):
        """
        Initialize the sequence generator.

        Args:
            sequence_length: Length of each generated sequence
            target_features: List of feature column names to use
        """
        self.sequence_length = sequence_length
        self.target_features = target_features

    def create_panel_sequences(self,
                              dataframe: pd.DataFrame) -> np.ndarray:
        """
        Create sequences from panel data with robust preprocessing.

        Args:
            dataframe: Input DataFrame with financial data

        Returns:
            Array of sequences with shape (n_sequences, sequence_length, n_features)
        """
        generated_sequences = []

        for ticker_symbol, ticker_group in dataframe.groupby("Ticker"):
            print(f"Processing {ticker_symbol}...")
            ticker_group = ticker_group.sort_index().copy()

            ticker_sequences = self._process_ticker_data(
                ticker_group, ticker_symbol
            )
            generated_sequences.extend(ticker_sequences)

        if not generated_sequences:
            print("Warning: No valid sequences generated")
            return np.empty(
                (0, self.sequence_length, len(self.target_features)),
                dtype=np.float32
            )

        sequences_array = np.array(generated_sequences, dtype=np.float32)
        print(f"Generated sequences: {len(sequences_array)}")
        return sequences_array

    def _process_ticker_data(self,
                            ticker_group: pd.DataFrame,
                            ticker_symbol: str) -> List[np.ndarray]:
        """
        Process data for a single ticker and generate sequences.

        Args:
            ticker_group: DataFrame containing data for one ticker
            ticker_symbol: Symbol of the ticker being processed

        Returns:
            List of sequence arrays for the ticker
        """
        try:
            feature_data = ticker_group[self.target_features].copy()
        except KeyError as key_error:
            print(f"Feature not found for {ticker_symbol}: {key_error}")
            return []

        feature_data = self._clean_financial_data(feature_data)

        feature_data = self._calculate_log_returns(feature_data)

        feature_data = feature_data.dropna()

        if len(feature_data) < self.sequence_length:
            print(
                f"Insufficient data for {ticker_symbol}: "
                f"{len(feature_data)} < {self.sequence_length}"
            )
            return []

        feature_data = self._remove_outliers(feature_data)

        return self._create_sequences_from_array(feature_data.values)

    def _clean_financial_data(self, feature_data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean financial data by handling zeros, negatives, and infinities.

        Args:
            feature_data: DataFrame with financial features

        Returns:
            Cleaned DataFrame
        """
        for column_name in feature_data.columns:
            if column_name.lower() != "volume":
                feature_data[column_name] = feature_data[column_name].replace(
                    [0, np.inf, -np.inf], np.nan
                )
                feature_data[column_name] = feature_data[column_name].where(
                    feature_data[column_name] > 0
                )

        return feature_data

    def _calculate_log_returns(self, feature_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log returns for financial data.

        Args:
            feature_data: DataFrame with financial features

        Returns:
            DataFrame with log returns
        """
        for column_name in feature_data.columns:
            if column_name.lower() != "volume":
                feature_data[column_name] = np.log(feature_data[column_name]).diff()

        return feature_data

    def _remove_outliers(self,
                        feature_data: pd.DataFrame,
                        lower_percentile: float = 0.01,
                        upper_percentile: float = 0.99) -> pd.DataFrame:
        """
        Remove extreme outliers using percentile clipping.

        Args:
            feature_data: DataFrame with features
            lower_percentile: Lower percentile for clipping
            upper_percentile: Upper percentile for clipping

        Returns:
            DataFrame with outliers clipped
        """
        for column_name in feature_data.columns:
            lower_bound = feature_data[column_name].quantile(lower_percentile)
            upper_bound = feature_data[column_name].quantile(upper_percentile)
            feature_data[column_name] = feature_data[column_name].clip(
                lower_bound, upper_bound
            )

        return feature_data

    def _create_sequences_from_array(self, data_array: np.ndarray) -> List[np.ndarray]:
        """
        Create sequences from a numpy array.

        Args:
            data_array: Input data array

        Returns:
            List of sequence arrays
        """
        sequences = []

        for start_index in range(len(data_array) - self.sequence_length + 1):
            sequence = data_array[start_index:start_index + self.sequence_length]

            if not (np.isnan(sequence).any() or np.isinf(sequence).any()):
                sequences.append(sequence)

        return sequences


class DataSplitter:
    """Handle train/validation data splitting."""

    @staticmethod
    def split_train_validation(sequences: np.ndarray,
                              validation_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split sequences into training and validation sets.

        Args:
            sequences: Array of sequences to split
            validation_ratio: Fraction of data to use for validation

        Returns:
            Tuple of (training_sequences, validation_sequences)
        """
        total_sequences = len(sequences)
        if total_sequences == 0:
            return np.array([]), np.array([])

        sequence_indices = np.arange(total_sequences)
        np.random.shuffle(sequence_indices)

        training_cutoff = int(total_sequences * (1 - validation_ratio))
        training_indices = sequence_indices[:training_cutoff]
        validation_indices = sequence_indices[training_cutoff:]

        return sequences[training_indices], sequences[validation_indices]


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""

    def __init__(self, sequence_data: np.ndarray):
        """
        Initialize the dataset.

        Args:
            sequence_data: Array of sequences with shape (n_sequences, seq_len, n_features)
        """
        self.sequence_data = torch.tensor(sequence_data, dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequence_data)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Get a single sequence by index.

        Args:
            index: Index of the sequence to retrieve

        Returns:
            Tensor containing the sequence
        """
        return self.sequence_data[index]


class GRUBasedModel(nn.Module):
    """Base GRU model with common functionality."""

    def __init__(self,
                 input_dimension: int,
                 hidden_dimension: int,
                 num_layers: int,
                 output_dimension: int,
                 dropout_rate: float = 0.1):
        """
        Initialize the GRU-based model.

        Args:
            input_dimension: Dimension of input features
            hidden_dimension: Dimension of hidden states
            num_layers: Number of GRU layers
            output_dimension: Dimension of output
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        self.gru_network = nn.GRU(
            input_dimension,
            hidden_dimension,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.output_layer = nn.Linear(hidden_dimension, output_dimension)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_tensor: Input tensor with shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor with shape (batch_size, seq_len, output_dim)
        """
        hidden_states, _ = self.gru_network(input_tensor)
        hidden_states = self.dropout_layer(hidden_states)
        output = self.output_layer(hidden_states)
        return output


class TimeSeriesEmbedder(nn.Module):
    """Embedder network that maps real time series to latent space."""

    def __init__(self,
                 input_dimension: int,
                 hidden_dimension: int,
                 num_layers: int):
        """
        Initialize the embedder.

        Args:
            input_dimension: Dimension of input time series
            hidden_dimension: Dimension of hidden/latent space
            num_layers: Number of GRU layers
        """
        super().__init__()
        self.embedding_network = GRUBasedModel(
            input_dimension, hidden_dimension, num_layers, hidden_dimension
        )

    def forward(self, time_series_input: torch.Tensor) -> torch.Tensor:
        """
        Embed time series into latent space.

        Args:
            time_series_input: Input time series tensor

        Returns:
            Embedded representation in latent space
        """
        embedded_output = self.embedding_network(time_series_input)
        return torch.sigmoid(embedded_output)  # Keep values bounded


class TimeSeriesRecovery(nn.Module):
    """Recovery network that maps latent representations back to time series."""

    def __init__(self,
                 hidden_dimension: int,
                 output_dimension: int,
                 num_layers: int):
        """
        Initialize the recovery network.

        Args:
            hidden_dimension: Dimension of hidden/latent space
            output_dimension: Dimension of output time series
            num_layers: Number of GRU layers
        """
        super().__init__()
        self.recovery_network = GRUBasedModel(
            hidden_dimension, hidden_dimension, num_layers, output_dimension
        )

    def forward(self, latent_representation: torch.Tensor) -> torch.Tensor:
        """
        Recover time series from latent representation.

        Args:
            latent_representation: Latent space tensor

        Returns:
            Recovered time series
        """
        return self.recovery_network(latent_representation)


class TimeSeriesGenerator(nn.Module):
    """Generator network that creates synthetic latent representations from noise."""

    def __init__(self,
                 noise_dimension: int,
                 hidden_dimension: int,
                 num_layers: int):
        """
        Initialize the generator.

        Args:
            noise_dimension: Dimension of input noise
            hidden_dimension: Dimension of output hidden states
            num_layers: Number of GRU layers
        """
        super().__init__()
        self.generation_network = GRUBasedModel(
            noise_dimension, hidden_dimension, num_layers, hidden_dimension
        )

    def forward(self, noise_input: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic latent representations from noise.

        Args:
            noise_input: Random noise tensor

        Returns:
            Generated latent representation
        """
        generated_output = self.generation_network(noise_input)
        return torch.sigmoid(generated_output)  # Match embedder output range


class TimeSeriesSupervisor(nn.Module):
    """Supervisor network that provides next-step supervision in latent space."""

    def __init__(self,
                 hidden_dimension: int,
                 num_layers: int):
        """
        Initialize the supervisor.

        Args:
            hidden_dimension: Dimension of hidden states
            num_layers: Number of GRU layers
        """
        super().__init__()
        self.supervision_network = GRUBasedModel(
            hidden_dimension, hidden_dimension, num_layers, hidden_dimension
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply supervision to hidden states.

        Args:
            hidden_states: Input hidden states

        Returns:
            Supervised hidden states
        """
        supervised_output = self.supervision_network(hidden_states)
        return torch.sigmoid(supervised_output)


class TimeSeriesDiscriminator(nn.Module):
    """Discriminator network that distinguishes real from synthetic sequences."""

    def __init__(self,
                 hidden_dimension: int,
                 num_layers: int):
        """
        Initialize the discriminator.

        Args:
            hidden_dimension: Dimension of hidden states
            num_layers: Number of GRU layers
        """
        super().__init__()
        self.discriminator_rnn = nn.GRU(
            hidden_dimension,
            hidden_dimension,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        self.classification_layer = nn.Linear(hidden_dimension, 1)

    def forward(self, hidden_sequences: torch.Tensor) -> torch.Tensor:
        """
        Classify sequences as real or synthetic.

        Args:
            hidden_sequences: Input hidden state sequences

        Returns:
            Classification logits (averaged over time)
        """
        rnn_output, _ = self.discriminator_rnn(hidden_sequences)
        classification_logits = self.classification_layer(rnn_output)
        return classification_logits.mean(dim=1)  # Average over time dimension


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


class TimeGANTrainer:
    """Main trainer class for TimeGAN model."""

    def __init__(self,
                 embedder: TimeSeriesEmbedder,
                 recovery: TimeSeriesRecovery,
                 generator: TimeSeriesGenerator,
                 supervisor: TimeSeriesSupervisor,
                 discriminator: TimeSeriesDiscriminator,
                 device: torch.device,
                 learning_rate: float = LEARNING_RATE):
        """
        Initialize the TimeGAN trainer.

        Args:
            embedder: Embedder network
            recovery: Recovery network
            generator: Generator network
            supervisor: Supervisor network
            discriminator: Discriminator network
            device: Device for computations
            learning_rate: Learning rate for optimizers
        """
        self.embedder = embedder
        self.recovery = recovery
        self.generator = generator
        self.supervisor = supervisor
        self.discriminator = discriminator
        self.device = device

        self.embedder_optimizer = optim.Adam(
            self.embedder.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        self.recovery_optimizer = optim.Adam(
            self.recovery.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        self.generator_optimizer = optim.Adam(
            list(self.generator.parameters()) + list(self.supervisor.parameters()),
            lr=learning_rate,
            weight_decay=1e-5
        )
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.noise_dimension = HIDDEN_DIMENSION

    def train_autoencoder_pretraining(self,
                                    train_dataloader: DataLoader,
                                    num_epochs: int) -> None:
        """
        Pre-train the autoencoder (embedder + recovery).

        Args:
            train_dataloader: DataLoader for training data
            num_epochs: Number of training epochs
        """
        print("Starting autoencoder pre-training...")
        self.embedder.train()
        self.recovery.train()

        for epoch_num in range(1, num_epochs + 1):
            epoch_total_loss = 0.0
            num_batches = 0

            for batch_data in train_dataloader:
                batch_data = batch_data.to(self.device)

                self.embedder_optimizer.zero_grad()
                self.recovery_optimizer.zero_grad()

                embedded_data = self.embedder(batch_data)
                reconstructed_data = self.recovery(embedded_data)
                reconstruction_loss = self.mse_loss(reconstructed_data, batch_data)

                reconstruction_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.embedder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.recovery.parameters(), 1.0)

                self.embedder_optimizer.step()
                self.recovery_optimizer.step()

                epoch_total_loss += reconstruction_loss.item()
                num_batches += 1

            average_loss = epoch_total_loss / num_batches if num_batches > 0 else 0
            print(f"[Autoencoder] Epoch {epoch_num}/{num_epochs} - Loss: {average_loss:.6f}")

    def train_supervisor_pretraining(self,
                                   train_dataloader: DataLoader,
                                   num_epochs: int) -> None:
        """
        Pre-train the supervisor network.

        Args:
            train_dataloader: DataLoader for training data
            num_epochs: Number of training epochs
        """
        print("Starting supervisor pre-training...")
        self.embedder.eval()
        self.generator.train()
        self.supervisor.train()

        for epoch_num in range(1, num_epochs + 1):
            epoch_total_loss = 0.0
            num_batches = 0

            for batch_data in train_dataloader:
                batch_data = batch_data.to(self.device)
                batch_size, sequence_length = batch_data.size(0), batch_data.size(1)

                self.generator_optimizer.zero_grad()

                noise_input = NoiseGenerator.sample_random_noise(
                    batch_size, sequence_length, self.noise_dimension, self.device
                )
                fake_hidden_states = self.generator(noise_input)
                supervised_hidden_states = self.supervisor(fake_hidden_states)

                if sequence_length > 1:
                    supervisor_loss = self.mse_loss(
                        supervised_hidden_states[:, :-1, :],
                        fake_hidden_states[:, 1:, :]
                    )
                else:
                    supervisor_loss = self.mse_loss(
                        supervised_hidden_states, fake_hidden_states
                    )

                supervisor_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.supervisor.parameters(), 1.0)

                self.generator_optimizer.step()

                epoch_total_loss += supervisor_loss.item()
                num_batches += 1

            average_loss = epoch_total_loss / num_batches if num_batches > 0 else 0
            print(f"[Supervisor] Epoch {epoch_num}/{num_epochs} - Loss: {average_loss:.6f}")

    def train_joint_training(self,
                           train_dataloader: DataLoader,
                           num_epochs: int) -> None:
        """
        Joint training of all networks.

        Args:
            train_dataloader: DataLoader for training data
            num_epochs: Number of training epochs
        """
        print("Starting joint training...")

        for epoch_num in range(1, num_epochs + 1):
            self.embedder.train()
            self.recovery.train()
            self.generator.train()
            self.supervisor.train()
            self.discriminator.train()

            generator_loss_total = 0.0
            discriminator_loss_total = 0.0
            autoencoder_loss_total = 0.0
            num_batches = 0

            for batch_data in train_dataloader:
                batch_data = batch_data.to(self.device)
                batch_size, sequence_length = batch_data.size(0), batch_data.size(1)

                self._update_autoencoder(batch_data)

                discriminator_loss = self._update_discriminator(
                    batch_data, batch_size, sequence_length
                )

                generator_loss = self._update_generator(
                    batch_data, batch_size, sequence_length
                )

                with torch.no_grad():
                    embedded_data = self.embedder(batch_data)
                    reconstructed_data = self.recovery(embedded_data)
                    autoencoder_loss = self.mse_loss(reconstructed_data, batch_data)

                generator_loss_total += generator_loss
                discriminator_loss_total += discriminator_loss
                autoencoder_loss_total += autoencoder_loss.item()
                num_batches += 1

            if num_batches > 0:
                avg_generator_loss = generator_loss_total / (num_batches * GENERATOR_TRAINING_STEPS)
                avg_discriminator_loss = discriminator_loss_total / (num_batches * DISCRIMINATOR_TRAINING_STEPS)
                avg_autoencoder_loss = autoencoder_loss_total / num_batches

                print(
                    f"[Joint Training] Epoch {epoch_num}/{num_epochs} | "
                    f"Autoencoder: {avg_autoencoder_loss:.5f} | "
                    f"Discriminator: {avg_discriminator_loss:.5f} | "
                    f"Generator: {avg_generator_loss:.5f}"
                )

    def _update_autoencoder(self, batch_data: torch.Tensor) -> None:
        """
        Update autoencoder networks (embedder and recovery).

        Args:
            batch_data: Input batch data
        """
        self.embedder_optimizer.zero_grad()
        self.recovery_optimizer.zero_grad()

        embedded_data = self.embedder(batch_data)
        reconstructed_data = self.recovery(embedded_data)
        autoencoder_loss = self.mse_loss(reconstructed_data, batch_data)

        autoencoder_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.embedder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.recovery.parameters(), 1.0)

        self.embedder_optimizer.step()
        self.recovery_optimizer.step()

    def _update_discriminator(self,
                            batch_data: torch.Tensor,
                            batch_size: int,
                            sequence_length: int) -> float:
        """
        Update discriminator network.

        Args:
            batch_data: Input batch data
            batch_size: Size of the batch
            sequence_length: Length of sequences

        Returns:
            Average discriminator loss over training steps
        """
        total_discriminator_loss = 0.0

        for _ in range(DISCRIMINATOR_TRAINING_STEPS):
            self.discriminator_optimizer.zero_grad()

            with torch.no_grad():
                real_hidden_states = self.embedder(batch_data)

            noise_input = NoiseGenerator.sample_random_noise(
                batch_size, sequence_length, self.noise_dimension, self.device
            )
            fake_hidden_states = self.generator(noise_input)
            supervised_fake_states = self.supervisor(fake_hidden_states)

            real_predictions = self.discriminator(real_hidden_states.detach())
            fake_predictions = self.discriminator(supervised_fake_states.detach())

            real_labels = torch.ones_like(real_predictions)
            fake_labels = torch.zeros_like(fake_predictions)

            real_loss = self.bce_loss(real_predictions, real_labels)
            fake_loss = self.bce_loss(fake_predictions, fake_labels)
            discriminator_loss = real_loss + fake_loss

            discriminator_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.discriminator_optimizer.step()

            total_discriminator_loss += discriminator_loss.item()

        return total_discriminator_loss / DISCRIMINATOR_TRAINING_STEPS

    def _update_generator(self,
                        batch_data: torch.Tensor,
                        batch_size: int,
                        sequence_length: int) -> float:
        """
        Update generator and supervisor networks.

        Args:
            batch_data: Input batch data
            batch_size: Size of the batch
            sequence_length: Length of sequences

        Returns:
            Average generator loss over training steps
        """
        total_generator_loss = 0.0

        for _ in range(GENERATOR_TRAINING_STEPS):
            self.generator_optimizer.zero_grad()

            noise_input = NoiseGenerator.sample_random_noise(
                batch_size, sequence_length, self.noise_dimension, self.device
            )
            fake_hidden_states = self.generator(noise_input)
            supervised_fake_states = self.supervisor(fake_hidden_states)

            fake_predictions = self.discriminator(supervised_fake_states)
            real_labels = torch.ones_like(fake_predictions)
            adversarial_loss = self.bce_loss(fake_predictions, real_labels)

            if sequence_length > 1:
                supervised_loss = self.mse_loss(
                    supervised_fake_states[:, :-1, :],
                    fake_hidden_states[:, 1:, :]
                )
            else:
                supervised_loss = torch.tensor(0.0, device=self.device)

            moment_matching_loss = torch.tensor(0.0, device=self.device)
            if MOMENT_MATCHING_WEIGHT > 0:
                with torch.no_grad():
                    real_hidden_states = self.embedder(batch_data)

                moment_matching_loss = self._calculate_moment_matching_loss(
                    real_hidden_states, supervised_fake_states
                )

            total_loss = (adversarial_loss +
                         SUPERVISED_LOSS_WEIGHT * supervised_loss +
                         MOMENT_MATCHING_WEIGHT * moment_matching_loss)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.supervisor.parameters(), 1.0)
            self.generator_optimizer.step()

            total_generator_loss += total_loss.item()

        return total_generator_loss / GENERATOR_TRAINING_STEPS

    def _calculate_moment_matching_loss(self,
                                      real_hidden_states: torch.Tensor,
                                      fake_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Calculate moment matching loss between real and fake hidden states.

        Args:
            real_hidden_states: Hidden states from real data
            fake_hidden_states: Hidden states from fake data

        Returns:
            Moment matching loss tensor
        """
        real_mean = real_hidden_states.mean(dim=(0, 1))
        fake_mean = fake_hidden_states.mean(dim=(0, 1))
        real_variance = real_hidden_states.var(dim=(0, 1))
        fake_variance = fake_hidden_states.var(dim=(0, 1))

        mean_loss = (real_mean - fake_mean).pow(2).mean()
        variance_loss = (real_variance - fake_variance).pow(2).mean()

        return mean_loss + variance_loss


class SyntheticDataGenerator:
    """Generate synthetic time series data using trained TimeGAN."""

    def __init__(self,
                 embedder: TimeSeriesEmbedder,
                 recovery: TimeSeriesRecovery,
                 generator: TimeSeriesGenerator,
                 supervisor: TimeSeriesSupervisor,
                 data_scaler: MinMaxScaler,
                 device: torch.device):
        """
        Initialize the synthetic data generator.

        Args:
            embedder: Trained embedder network
            recovery: Trained recovery network
            generator: Trained generator network
            supervisor: Trained supervisor network
            data_scaler: Fitted data scaler for denormalization
            device: Device for computations
        """
        self.embedder = embedder
        self.recovery = recovery
        self.generator = generator
        self.supervisor = supervisor
        self.data_scaler = data_scaler
        self.device = device
        self.noise_dimension = HIDDEN_DIMENSION

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
                 device: torch.device):
        """
        Initialize the diversity checker.

        Args:
            generator: Generator network
            supervisor: Supervisor network
            recovery: Recovery network
            device: Device for computations
        """
        self.generator = generator
        self.supervisor = supervisor
        self.recovery = recovery
        self.device = device
        self.noise_dimension = HIDDEN_DIMENSION

    def check_model_diversity(self) -> None:
        """Check if models generate diverse outputs."""
        print("\n" + "=" * 50)
        print("MODEL DIVERSITY VERIFICATION")
        print("=" * 50)

        print("1. Testing with identical noise...")
        test_noise = torch.randn(5, SEQUENCE_LENGTH, self.noise_dimension, device=self.device)
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
        different_noise = torch.randn(5, SEQUENCE_LENGTH, self.noise_dimension, device=self.device)

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


class TimeSeriesVisualizer:
    """Visualize time series data and analysis results."""

    @staticmethod
    def plot_individual_series(synthetic_data: np.ndarray,
                             num_series: int = 10,
                             feature_index: int = 0,
                             title_prefix: str = "Series") -> None:
        """
        Plot individual time series.

        Args:
            synthetic_data: Array of synthetic sequences
            num_series: Number of series to plot
            feature_index: Index of feature to plot
            title_prefix: Prefix for plot titles
        """
        plt.figure(figsize=(15, 8))

        num_cols = 5
        num_rows = (num_series + num_cols - 1) // num_cols

        for series_idx in range(min(num_series, len(synthetic_data))):
            plt.subplot(num_rows, num_cols, series_idx + 1)
            plt.plot(synthetic_data[series_idx, :, feature_index], linewidth=2)
            plt.title(f'{title_prefix} {series_idx + 1}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_overlay_series(synthetic_data: np.ndarray,
                          num_series: int = 20,
                          feature_index: int = 0,
                          alpha_value: float = 0.6) -> None:
        """
        Plot overlaid time series.

        Args:
            synthetic_data: Array of synthetic sequences
            num_series: Number of series to overlay
            feature_index: Index of feature to plot
            alpha_value: Transparency value for plots
        """
        plt.figure(figsize=(12, 6))

        for series_idx in range(min(num_series, len(synthetic_data))):
            plt.plot(synthetic_data[series_idx, :, feature_index],
                    alpha=alpha_value, linewidth=1.5)

        plt.title(f'Overlay of {min(num_series, len(synthetic_data))} Synthetic Series')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def plot_statistical_analysis(synthetic_data: np.ndarray,
                                feature_index: int = 0) -> None:
        """
        Plot statistical analysis of generated series.

        Args:
            synthetic_data: Array of synthetic sequences
            feature_index: Index of feature to analyze
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        all_values = synthetic_data[:, :, feature_index].flatten()
        series_means = synthetic_data[:, :, feature_index].mean(axis=1)
        series_stds = synthetic_data[:, :, feature_index].std(axis=1)
        series_mins = synthetic_data[:, :, feature_index].min(axis=1)
        series_maxs = synthetic_data[:, :, feature_index].max(axis=1)

        axes[0, 0].hist(all_values, bins=50, alpha=0.7, density=True, color='skyblue')
        axes[0, 0].set_title('Distribution of All Values')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].grid(True, alpha=0.3)

        stats.probplot(all_values, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normality)')
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].hist(series_means, bins=30, alpha=0.7, color='lightcoral')
        axes[0, 2].set_title('Distribution of Series Means')
        axes[0, 2].set_xlabel('Mean')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].hist(series_stds, bins=30, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Distribution of Standard Deviations')
        axes[1, 0].set_xlabel('Standard Deviation')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

        box_data = [series_mins, series_maxs]
        axes[1, 1].boxplot(box_data, labels=['Minimums', 'Maximums'])
        axes[1, 1].set_title('Box Plot: Extreme Values per Series')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].grid(True, alpha=0.3)

        mean_evolution = synthetic_data[:, :, feature_index].mean(axis=0)
        std_evolution = synthetic_data[:, :, feature_index].std(axis=0)

        time_axis = range(len(mean_evolution))
        axes[1, 2].plot(time_axis, mean_evolution, 'b-', linewidth=2, label='Mean')
        axes[1, 2].fill_between(time_axis,
                               mean_evolution - std_evolution,
                               mean_evolution + std_evolution,
                               alpha=0.3, label='±1 Standard Deviation')
        axes[1, 2].set_title('Average Temporal Evolution')
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_autocorrelation_analysis(synthetic_data: np.ndarray,
                                    feature_index: int = 0,
                                    max_lags: int = 10) -> None:
        """
        Plot autocorrelation analysis of synthetic series.

        Args:
            synthetic_data: Array of synthetic sequences
            feature_index: Index of feature to analyze
            max_lags: Maximum number of lags to analyze
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        num_series = min(50, len(synthetic_data))
        autocorrelations = []

        for series_idx in range(num_series):
            time_series = synthetic_data[series_idx, :, feature_index]
            series_autocorr = []
            for lag in range(1, max_lags + 1):
                if len(time_series) > lag:
                    autocorr = np.corrcoef(time_series[:-lag], time_series[lag:])[0, 1]
                    series_autocorr.append(autocorr if not np.isnan(autocorr) else 0)
                else:
                    series_autocorr.append(0)
            autocorrelations.append(series_autocorr)

        autocorrelations = np.array(autocorrelations)

        sns.heatmap(autocorrelations[:20],  # Show only 20 series
                    annot=False,
                    xticklabels=range(1, max_lags + 1),
                    yticklabels=[f'Series {i+1}' for i in range(min(20, num_series))],
                    cmap='coolwarm',
                    center=0,
                    ax=axes[0, 0])
        axes[0, 0].set_title('Autocorrelation by Series')
        axes[0, 0].set_xlabel('Lag')

        mean_autocorr = np.nanmean(autocorrelations, axis=0)
        std_autocorr = np.nanstd(autocorrelations, axis=0)

        lags = range(1, max_lags + 1)
        axes[0, 1].bar(lags, mean_autocorr, yerr=std_autocorr,
                       alpha=0.7, capsize=5, color='steelblue')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Average Autocorrelation by Lag')
        axes[0, 1].set_xlabel('Lag')
        axes[0, 1].set_ylabel('Autocorrelation')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].hist(autocorrelations[:, 0], bins=20, alpha=0.7, color='orange')
        axes[1, 0].set_title('Distribution of Lag-1 Autocorrelation')
        axes[1, 0].set_xlabel('Autocorrelation')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

        for series_idx in range(min(5, num_series)):
            axes[1, 1].plot(lags, autocorrelations[series_idx],
                           alpha=0.6, marker='o', linewidth=1)
        axes[1, 1].set_title('Autocorrelation: Individual Examples')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_with_original_data(synthetic_data: np.ndarray,
                                 original_data: np.ndarray,
                                 feature_index: int = 0,
                                 num_comparison: int = 10) -> None:
        """
        Compare synthetic data with original data.

        Args:
            synthetic_data: Array of synthetic sequences
            original_data: Array of original sequences
            feature_index: Index of feature to compare
            num_comparison: Number of series to compare
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        for series_idx in range(min(num_comparison, len(synthetic_data), len(original_data))):
            axes[0, 0].plot(original_data[series_idx, :, feature_index],
                           alpha=0.7, linewidth=1, color='blue')
            axes[0, 1].plot(synthetic_data[series_idx, :, feature_index],
                           alpha=0.7, linewidth=1, color='red')

        axes[0, 0].set_title(f'Original Series (n={min(num_comparison, len(original_data))})')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_title(f'Synthetic Series (n={min(num_comparison, len(synthetic_data))})')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].grid(True, alpha=0.3)

        original_values = original_data[:, :, feature_index].flatten()
        synthetic_values = synthetic_data[:, :, feature_index].flatten()

        axes[1, 0].hist(original_values, bins=50, alpha=0.6, label='Original',
                        density=True, color='blue')
        axes[1, 0].hist(synthetic_values, bins=50, alpha=0.6, label='Synthetic',
                        density=True, color='red')
        axes[1, 0].set_title('Distribution Comparison')
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        original_stats = [original_values.mean(), original_values.std(),
                         np.percentile(original_values, 25), np.percentile(original_values, 75)]
        synthetic_stats = [synthetic_values.mean(), synthetic_values.std(),
                          np.percentile(synthetic_values, 25), np.percentile(synthetic_values, 75)]

        stat_labels = ['Mean', 'Std Dev', 'Q1', 'Q3']
        x_positions = np.arange(len(stat_labels))

        bar_width = 0.35
        axes[1, 1].bar(x_positions - bar_width/2, original_stats, bar_width,
                       label='Original', color='blue', alpha=0.7)
        axes[1, 1].bar(x_positions + bar_width/2, synthetic_stats, bar_width,
                       label='Synthetic', color='red', alpha=0.7)

        axes[1, 1].set_title('Comparative Statistics')
        axes[1, 1].set_xlabel('Metric')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_xticks(x_positions)
        axes[1, 1].set_xticklabels(stat_labels)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class GenerationSummaryReporter:
    """Generate comprehensive reports on synthetic data generation."""

    @staticmethod
    def print_generation_summary(synthetic_data: np.ndarray) -> None:
        """
        Print comprehensive summary of generated synthetic data.

        Args:
            synthetic_data: Array of synthetic sequences
        """
        num_series, sequence_length, num_features = synthetic_data.shape

        print("\n" + "=" * 60)
        print("SYNTHETIC SERIES GENERATION SUMMARY")
        print("=" * 60)
        print(f"Number of generated series: {num_series}")
        print(f"Length of each series: {sequence_length}")
        print(f"Number of features: {num_features}")

        for feature_idx in range(num_features):
            feature_values = synthetic_data[:, :, feature_idx].flatten()
            print(f"\nFeature {feature_idx}:")
            print(f"  Mean: {feature_values.mean():.6f}")
            print(f"  Standard Deviation: {feature_values.std():.6f}")
            print(f"  Minimum: {feature_values.min():.6f}")
            print(f"  Maximum: {feature_values.max():.6f}")
            print(f"  Median: {np.median(feature_values):.6f}")

            _, p_value = stats.normaltest(feature_values)
            print(f"  Normality test (p-value): {p_value:.6f}")

        print("=" * 60)


def main() -> None:
    """Main execution function for TimeGAN training and generation."""

    print("Setting up TimeGAN training environment...")
    setup_random_seeds(RANDOM_SEED)
    device = get_device()

    print("\n" + "=" * 50)
    print("DATA COLLECTION AND PREPROCESSING")
    print("=" * 50)

    data_downloader = FinancialDataDownloader()
    try:
        raw_financial_data = data_downloader.download_stock_data(
            STOCK_TICKERS, DATA_START_DATE, DATA_END_DATE
        )
    except Exception as download_error:
        print(f"Error during data download: {download_error}")
        return

    print("\nGenerating training sequences...")
    sequence_generator = TimeSeriesSequenceGenerator(SEQUENCE_LENGTH, TARGET_FEATURES)
    training_sequences = sequence_generator.create_panel_sequences(raw_financial_data)

    if len(training_sequences) == 0:
        print("Error: No sequences generated. Please check the data.")
        return

    print(f"Generated sequences shape: {training_sequences.shape}")

    num_sequences, sequence_length, num_features = training_sequences.shape

    if np.isnan(training_sequences).any() or np.isinf(training_sequences).any():
        print("Removing sequences with NaN/Inf values...")
        valid_mask = ~(np.isnan(training_sequences).any(axis=(1, 2)) |
                      np.isinf(training_sequences).any(axis=(1, 2)))
        training_sequences = training_sequences[valid_mask]
        num_sequences, sequence_length, num_features = training_sequences.shape
        print(f"Valid sequences remaining: {num_sequences}")

    if num_sequences == 0:
        print("Error: No valid sequences after data cleaning.")
        return

    print("Normalizing data...")
    data_scaler = MinMaxScaler(feature_range=(0, 1))
    sequences_2d = training_sequences.reshape(-1, num_features)
    normalized_sequences_2d = data_scaler.fit_transform(sequences_2d)
    normalized_sequences = normalized_sequences_2d.reshape(num_sequences, sequence_length, num_features)

    data_splitter = DataSplitter()
    train_sequences, validation_sequences = data_splitter.split_train_validation(
        normalized_sequences, validation_ratio=0.1
    )

    if len(train_sequences) == 0:
        print("Error: No training sequences after data splitting.")
        return

    print(f"Training sequences: {len(train_sequences)}, Validation sequences: {len(validation_sequences)}")

    train_dataset = TimeSeriesDataset(train_sequences)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=min(TRAINING_BATCH_SIZE, len(train_sequences)),
        shuffle=True,
        drop_last=True
    )

    if len(validation_sequences) > 0:
        validation_dataset = TimeSeriesDataset(validation_sequences)
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=min(TRAINING_BATCH_SIZE, len(validation_sequences)),
            shuffle=False,
            drop_last=False
        )

    print("\n" + "=" * 50)
    print("INITIALIZING TIMEGAN MODELS")
    print("=" * 50)

    input_dimension = num_features
    hidden_dimension = HIDDEN_DIMENSION
    noise_dimension = hidden_dimension

    embedder_network = TimeSeriesEmbedder(input_dimension, hidden_dimension, RNN_LAYERS_COUNT).to(device)
    recovery_network = TimeSeriesRecovery(hidden_dimension, input_dimension, RNN_LAYERS_COUNT).to(device)
    generator_network = TimeSeriesGenerator(noise_dimension, hidden_dimension, RNN_LAYERS_COUNT).to(device)
    supervisor_network = TimeSeriesSupervisor(hidden_dimension, RNN_LAYERS_COUNT).to(device)
    discriminator_network = TimeSeriesDiscriminator(hidden_dimension, RNN_LAYERS_COUNT).to(device)

    print(f"Models initialized with:")
    print(f"  Input dimension: {input_dimension}")
    print(f"  Hidden dimension: {hidden_dimension}")
    print(f"  Number of layers: {RNN_LAYERS_COUNT}")

    timegan_trainer = TimeGANTrainer(
        embedder_network,
        recovery_network,
        generator_network,
        supervisor_network,
        discriminator_network,
        device,
        LEARNING_RATE
    )

    print("\n" + "=" * 50)
    print("TIMEGAN TRAINING PHASES")
    print("=" * 50)

    print(f"\nPhase 1/3: Autoencoder Pre-training ({AUTOENCODER_PRETRAIN_EPOCHS} epochs)...")
    timegan_trainer.train_autoencoder_pretraining(train_dataloader, AUTOENCODER_PRETRAIN_EPOCHS)

    print(f"\nPhase 2/3: Supervisor Pre-training ({SUPERVISOR_PRETRAIN_EPOCHS} epochs)...")
    timegan_trainer.train_supervisor_pretraining(train_dataloader, SUPERVISOR_PRETRAIN_EPOCHS)

    print(f"\nPhase 3/3: Joint Training ({JOINT_TRAINING_EPOCHS} epochs)...")
    timegan_trainer.train_joint_training(train_dataloader, JOINT_TRAINING_EPOCHS)

    print("\n" + "=" * 50)
    print("MODEL QUALITY ASSESSMENT")
    print("=" * 50)

    diversity_checker = ModelDiversityChecker(
        generator_network, supervisor_network, recovery_network, device
    )
    diversity_checker.check_model_diversity()

    print("\n" + "=" * 50)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 50)

    synthetic_generator = SyntheticDataGenerator(
        embedder_network, recovery_network, generator_network,
        supervisor_network, data_scaler, device
    )

    generation_configs = [
        {"add_noise": False, "temperature": 1.0, "name": "Standard"},
        {"add_noise": True, "temperature": 1.0, "name": "With extra noise"},
        {"add_noise": True, "temperature": 1.5, "name": "High temperature + noise"},
        {"add_noise": True, "temperature": 2.0, "name": "Very high temperature"},
    ]

    best_synthetic_data = None
    best_diversity_score = 0

    print("Testing different generation configurations...")
    for config in generation_configs:
        print(f"\nTesting configuration: {config['name']}")
        test_synthetic = synthetic_generator.generate_diverse_synthetic_batch(
            20, SEQUENCE_LENGTH,
            add_noise=config["add_noise"],
            temperature=config["temperature"]
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

    num_synthetic_series = 100
    print(f"\nGenerating {num_synthetic_series} synthetic series with best configuration...")

    if best_synthetic_data is not None:
        final_synthetic_data = synthetic_generator.generate_diverse_synthetic_batch(
            num_synthetic_series, SEQUENCE_LENGTH,
            add_noise=True, temperature=1.5
        )
    else:
        print("Using standard configuration...")
        final_synthetic_data = synthetic_generator.generate_synthetic_sequences(
            num_synthetic_series, SEQUENCE_LENGTH
        )

    print("\n" + "=" * 50)
    print("FINAL DIVERSITY VERIFICATION")
    print("=" * 50)

    series_means = [series[:, 0].mean() for series in final_synthetic_data]
    series_stds = [series[:, 0].std() for series in final_synthetic_data]

    print(f"Diversity of means: {np.std(series_means):.6f}")
    print(f"Diversity of standard deviations: {np.std(series_stds):.6f}")

    correlations = []
    for i in range(min(10, len(final_synthetic_data))):
        for j in range(i + 1, min(10, len(final_synthetic_data))):
            correlation = np.corrcoef(final_synthetic_data[i, :, 0],
                                    final_synthetic_data[j, :, 0])[0, 1]
            if not np.isnan(correlation):
                correlations.append(abs(correlation))

    if correlations:
        print(f"Average correlation between series: {np.mean(correlations):.6f}")
        print(f"Maximum correlation: {np.max(correlations):.6f}")

        if np.mean(correlations) > 0.8:
            print("⚠️  WARNING: High correlations - series may be too similar")
        elif np.mean(correlations) < 0.3:
            print("✓ Good diversity - low correlations between series")

    summary_reporter = GenerationSummaryReporter()
    summary_reporter.print_generation_summary(final_synthetic_data)

    if np.std(series_means) > 1e-6:  # Minimum diversity threshold
        print("\n" + "=" * 50)
        print("GENERATING VISUALIZATIONS AND ANALYSIS")
        print("=" * 50)

        visualizer = TimeSeriesVisualizer()

        print("Generating individual series plots...")
        visualizer.plot_individual_series(final_synthetic_data, num_series=12, feature_index=0)

        print("Generating overlay plot...")
        visualizer.plot_overlay_series(final_synthetic_data, num_series=50, feature_index=0)

        print("Generating statistical analysis...")
        visualizer.plot_statistical_analysis(final_synthetic_data, feature_index=0)

        print("Generating autocorrelation analysis...")
        visualizer.plot_autocorrelation_analysis(final_synthetic_data, feature_index=0)

        if len(train_sequences) > 0:
            print("Generating comparison with original data...")
            original_2d = train_sequences.reshape(-1, num_features)
            original_denormalized_2d = data_scaler.inverse_transform(original_2d)
            original_denormalized = original_denormalized_2d.reshape(
                len(train_sequences), SEQUENCE_LENGTH, num_features
            )
            visualizer.compare_with_original_data(
                final_synthetic_data, original_denormalized, feature_index=0
            )

        print("Visualization completed!")
    else:
        print("\n⚠️  Series are too similar - skipping visualizations")
        print("Possible solutions:")
        print("1. Re-train the model with more epochs")
        print("2. Increase noise dimension (Z_dim)")
        print("3. Modify generator architecture")
        print("4. Adjust training hyperparameters")

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