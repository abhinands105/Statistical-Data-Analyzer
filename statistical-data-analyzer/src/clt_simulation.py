import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import os

def clt_simulation(data, sample_size=5, num_samples=1000):
    data = np.array(data)
    sample_means = []

    for _ in range(num_samples):
        sample = np.random.choice(data, size=sample_size, replace=True)
        sample_means.append(np.mean(sample))

    sample_means = np.array(sample_means)

    # Create plots directory safely
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    PLOT_DIR = os.path.join(BASE_DIR, "plots")
    os.makedirs(PLOT_DIR, exist_ok=True)

    plt.figure()
    sns.histplot(sample_means, kde=True, stat="density")

    mu, std = norm.fit(sample_means)
    x = np.linspace(sample_means.min(), sample_means.max(), 100)
    plt.plot(x, norm.pdf(x, mu, std))

    plt.title(
        f"CLT Simulation\nSample Size={sample_size}, Samples={num_samples}"
    )
    plt.xlabel("Sample Mean")
    plt.ylabel("Density")

    plt.savefig(os.path.join(PLOT_DIR, "clt_simulation.png"))
    plt.close()

    return mu, std
