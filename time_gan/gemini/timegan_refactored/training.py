
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .models import (
    TimeSeriesEmbedder,
    TimeSeriesRecovery,
    TimeSeriesGenerator,
    TimeSeriesSupervisor,
    TimeSeriesDiscriminator,
)
from .generation import NoiseGenerator


class TimeGANTrainer:
    """Main trainer class for TimeGAN model."""

    def __init__(self,
                 embedder: TimeSeriesEmbedder,
                 recovery: TimeSeriesRecovery,
                 generator: TimeSeriesGenerator,
                 supervisor: TimeSeriesSupervisor,
                 discriminator: TimeSeriesDiscriminator,
                 device: torch.device,
                 learning_rate: float,
                 hidden_dimension: int,
                 generator_training_steps: int = 1,
                 discriminator_training_steps: int = 1,
                 supervised_loss_weight: float = 1.0,
                 moment_matching_weight: float = 1e-4):
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
            hidden_dimension: Dimension of the hidden space
            generator_training_steps: Number of generator steps per batch
            discriminator_training_steps: Number of discriminator steps per batch
            supervised_loss_weight: Weight for the supervised loss
            moment_matching_weight: Weight for the moment matching loss
        """
        self.embedder = embedder
        self.recovery = recovery
        self.generator = generator
        self.supervisor = supervisor
        self.discriminator = discriminator
        self.device = device

        self.generator_training_steps = generator_training_steps
        self.discriminator_training_steps = discriminator_training_steps
        self.supervised_loss_weight = supervised_loss_weight
        self.moment_matching_weight = moment_matching_weight

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

        self.noise_dimension = hidden_dimension

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
                avg_generator_loss = generator_loss_total / (num_batches * self.generator_training_steps)
                avg_discriminator_loss = discriminator_loss_total / (num_batches * self.discriminator_training_steps)
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

        for _ in range(self.discriminator_training_steps):
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

        return total_discriminator_loss / self.discriminator_training_steps

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

        for _ in range(self.generator_training_steps):
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
            if self.moment_matching_weight > 0:
                with torch.no_grad():
                    real_hidden_states = self.embedder(batch_data)

                moment_matching_loss = self._calculate_moment_matching_loss(
                    real_hidden_states, supervised_fake_states
                )

            total_loss = (adversarial_loss +
                         self.supervised_loss_weight * supervised_loss +
                         self.moment_matching_weight * moment_matching_loss)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.supervisor.parameters(), 1.0)
            self.generator_optimizer.step()

            total_generator_loss += total_loss.item()

        return total_generator_loss / self.generator_training_steps

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
