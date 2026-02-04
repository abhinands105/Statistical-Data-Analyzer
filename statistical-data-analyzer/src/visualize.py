import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import os

def plot_distribution(x):
    # Ensure plots directory exists
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    PLOT_DIR = os.path.join(BASE_DIR, "plots")
    os.makedirs(PLOT_DIR, exist_ok=True)

    plt.figure()
    sns.histplot(x, kde=True, stat="density")

    mu, std = norm.fit(x)
    xmin, xmax = plt.xlim()
    xs = np.linspace(xmin, xmax, 100)
    plt.plot(xs, norm.pdf(xs, mu, std))

    plt.title("Data Distribution with Normal Fit")
    plt.savefig(os.path.join(PLOT_DIR, "distribution.png"))
    plt.close()
