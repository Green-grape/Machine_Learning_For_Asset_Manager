import numpy as np


def get_optimal_bin_width(N: int) -> int:
    """
    Calculate the optimal bin width for discretized data using Hacine-Gharbi's method.
    """
    optimal_tau = np.cbrt(8 + 324 * N + 12 * np.sqrt(36 * N + 729 * N**2))
    optimal_bin_width = np.round(optimal_tau / 6 + 2 / (3 * optimal_tau) + 1 / 3)
    return int(optimal_bin_width)


def get_optimal_bin_width_2d(N: int, corr: float):
    temp = 1 + 24 * N / (
        (1 - corr**2) + 1e-4
    )  # Adding a small value to avoid division by zero
    temp = 1 + np.sqrt(temp)
    return np.round(np.sqrt(temp) * (1 / np.sqrt(2)))
