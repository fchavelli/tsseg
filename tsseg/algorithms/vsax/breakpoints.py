"""SAX breakpoints for normal distribution."""

import numpy as np
from scipy.stats import norm


def get_breakpoints(alphabet_size: int) -> np.ndarray:
    """Get SAX breakpoints for a given alphabet size.

    Parameters
    ----------
    alphabet_size : int
        The size of the SAX alphabet. Must be >= 1.

    Returns
    -------
    np.ndarray
        Array of breakpoints of size (alphabet_size - 1).
    """
    if alphabet_size < 1:
        raise ValueError("Alphabet size must be at least 1.")

    # Calculate breakpoints dynamically
    # We want to divide the standard normal distribution into 'alphabet_size' equiprobable regions.
    # The breakpoints are the quantiles (ppf) for probabilities 1/A, 2/A, ..., (A-1)/A.
    quantiles = np.linspace(0, 1, alphabet_size + 1)[1:-1]
    return norm.ppf(quantiles)
