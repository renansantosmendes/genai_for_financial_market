

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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

