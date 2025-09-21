
from typing import List, Optional

class Config:
    """Configuration class for the TimeGAN model."""

    # --- General ---
    RANDOM_SEED: int = 42

    # --- Data ---
    STOCK_TICKERS: List[str] = ["AAPL", "MSFT", "GOOGL"]
    DATA_START_DATE: str = "2018-01-01"
    DATA_END_DATE: Optional[str] = None
    SEQUENCE_LENGTH: int = 24
    TARGET_FEATURES: List[str] = ["Close"]

    # --- Model ---
    HIDDEN_DIMENSION: int = 24
    RNN_LAYERS_COUNT: int = 2

    # --- Training ---
    TRAINING_BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-3
    AUTOENCODER_PRETRAIN_EPOCHS: int = 30
    SUPERVISOR_PRETRAIN_EPOCHS: int = 30
    JOINT_TRAINING_EPOCHS: int = 100
    GENERATOR_TRAINING_STEPS: int = 1
    DISCRIMINATOR_TRAINING_STEPS: int = 1
    SUPERVISED_LOSS_WEIGHT: float = 1.0
    MOMENT_MATCHING_WEIGHT: float = 1e-4
