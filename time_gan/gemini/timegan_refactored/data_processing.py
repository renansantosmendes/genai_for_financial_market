import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config

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


def process_and_prepare_data(
    raw_data: pd.DataFrame,
    config: Config
) -> Tuple[DataLoader, Optional[DataLoader], MinMaxScaler, int, np.ndarray]:
    """
    Processes raw data and prepares DataLoader for training.

    Args:
        raw_data (pd.DataFrame): The raw financial data.
        config (Config): The configuration object.

    Returns:
        Tuple[DataLoader, Optional[DataLoader], MinMaxScaler, int, np.ndarray]:
            - train_dataloader
            - validation_dataloader
            - data_scaler
            - num_features
            - train_sequences (for visualization)
    """
    print("\nGenerating training sequences...")
    sequence_generator = TimeSeriesSequenceGenerator(config.SEQUENCE_LENGTH, config.TARGET_FEATURES)
    training_sequences = sequence_generator.create_panel_sequences(raw_data)

    if len(training_sequences) == 0:
        raise ValueError("No sequences generated. Please check the data.")

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
        raise ValueError("No valid sequences after data cleaning.")

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
        raise ValueError("No training sequences after data splitting.")

    print(f"Training sequences: {len(train_sequences)}, Validation sequences: {len(validation_sequences)}")

    train_dataset = TimeSeriesDataset(train_sequences)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=min(config.TRAINING_BATCH_SIZE, len(train_sequences)),
        shuffle=True,
        drop_last=True
    )

    validation_dataloader = None
    if len(validation_sequences) > 0:
        validation_dataset = TimeSeriesDataset(validation_sequences)
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=min(config.TRAINING_BATCH_SIZE, len(validation_sequences)),
            shuffle=False,
            drop_last=False
        )

    return train_dataloader, validation_dataloader, data_scaler, num_features, train_sequences