import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.weightstats import ztest


def load_data(path):
    return pd.read_csv(path)

def descriptive_stats(data):
    x = data.values.flatten()
    return {
        "mean": np.mean(x),
        "median": np.median(x),
        "mode": stats.mode(x, keepdims=True)[0][0],
        "variance": np.var(x, ddof=1),
        "std_dev": np.std(x, ddof=1)
    }

def correlation_covariance(x, y):
    return {
        "covariance": np.cov(x, y)[0, 1],
        "correlation": np.corrcoef(x, y)[0, 1]
    }

def probability_distributions(x):
    return {
        "normal_fit": stats.norm.fit(x),
        "binomial_pmf": stats.binom.pmf(k=5, n=10, p=0.5),
        "poisson_pmf": stats.poisson.pmf(k=3, mu=np.mean(x))
    }

def hypothesis_tests(x):
    return {
        "z_test": ztest(x, value=15),
        "t_test": stats.ttest_1samp(x, popmean=15),
        "chi_square": stats.chisquare(x)
    }
