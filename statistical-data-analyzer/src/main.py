import os

from stats_core import (
    load_data,
    descriptive_stats,
    probability_distributions,
    hypothesis_tests
)

from visualize import plot_distribution
from clt_simulation import clt_simulation


# -----------------------------
# Path handling (robust)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "sample.csv")


def main():
    # -----------------------------
    # Load data
    # -----------------------------
    data = load_data(DATA_PATH)
    x = data["value"]

    # -----------------------------
    # Descriptive Statistics
    # -----------------------------
    print("\n📌 Descriptive Statistics")
    stats_summary = descriptive_stats(x)
    for k, v in stats_summary.items():
        print(f"{k}: {v:.3f}")

    # -----------------------------
    # Probability Distributions
    # -----------------------------
    print("\n📌 Probability Models")
    prob_models = probability_distributions(x)
    for k, v in prob_models.items():
        print(f"{k}: {v}")

    # -----------------------------
    # Hypothesis Testing
    # -----------------------------
    print("\n📌 Hypothesis Testing")
    tests = hypothesis_tests(x)
    for k, v in tests.items():
        print(f"{k}: {v}")

    # -----------------------------
    # Distribution Visualization
    # -----------------------------
    print("\n📌 Generating Distribution Plot...")
    plot_distribution(x)
    print("Saved to plots/distribution.png")

    # -----------------------------
    # Central Limit Theorem
    # -----------------------------
    print("\n📌 Central Limit Theorem Simulation")
    clt_mean, clt_std = clt_simulation(
        x,
        sample_size=5,
        num_samples=1000
    )
    print(f"CLT Mean ≈ {clt_mean:.3f}")
    print(f"CLT Std  ≈ {clt_std:.3f}")
    print("Saved to plots/clt_simulation.png")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
