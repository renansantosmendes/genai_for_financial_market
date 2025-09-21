
import torch
import torch.nn as nn

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
