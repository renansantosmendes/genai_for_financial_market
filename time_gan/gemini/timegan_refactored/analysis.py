
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .config import Config
from .visualization import GenerationSummaryReporter, TimeSeriesVisualizer

def analyze_and_visualize_results(
    final_synthetic_data: np.ndarray,
    train_sequences: np.ndarray,
    data_scaler: MinMaxScaler,
    num_features: int,
    config: Config
) -> None:
    """Analyzes and visualizes the generated data."""

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
                len(train_sequences), config.SEQUENCE_LENGTH, num_features
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
