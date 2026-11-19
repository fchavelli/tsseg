"""Autoencoders trained by TireDetector for each ``domain``."""

import numpy as np
import pytest

pytest.importorskip("torch")

from tsseg.algorithms import TireDetector  # noqa: E402

# The latent dimensions tell the two domains apart in the recorded calls.
SMALL = {
    "window_size": 16,
    "max_epochs": 3,
    "patience": 2,
    "intermediate_dim_td": 8,
    "latent_dim_td": 2,
    "nr_shared_td": 1,
    "nr_ae_td": 2,
    "intermediate_dim_fd": 8,
    "latent_dim_fd": 3,
    "nr_shared_fd": 1,
    "nr_ae_fd": 2,
    "n_segments": 4,
    "random_state": 0,
}
DOMAIN_OF_LATENT_DIM = {SMALL["latent_dim_td"]: "TD", SMALL["latent_dim_fd"]: "FD"}


def _steps(n_per=150, rng=0):
    rng = np.random.default_rng(rng)
    return np.concatenate([rng.normal(m, 0.3, n_per) for m in (0.0, 2.0, -1.0, 1.5)])


@pytest.mark.parametrize(
    ("domain", "expected"),
    [("TD", ["TD"]), ("FD", ["FD"]), ("both", ["TD", "FD"])],
)
def test_only_the_autoencoders_of_the_domain_are_trained(monkeypatch, domain, expected):
    trained = []
    original = TireDetector._train_autoencoder

    def recording(self, windows, intermediate_dim, latent_dim, *args, **kwargs):
        trained.append(DOMAIN_OF_LATENT_DIM[latent_dim])
        return original(self, windows, intermediate_dim, latent_dim, *args, **kwargs)

    monkeypatch.setattr(TireDetector, "_train_autoencoder", recording)
    TireDetector(**SMALL, domain=domain).fit_predict(_steps())
    assert trained == expected


def test_frequency_domain_ignores_the_time_domain_parameters():
    """In ``"FD"`` the time domain settings change neither the scores nor the CPs.

    Training the unused time domain autoencoder drew from the torch random
    stream before the frequency domain one was initialised, so its size
    changed the frequency domain result at equal seed.
    """
    x = _steps()
    reference = TireDetector(**SMALL, domain="FD")
    pred = reference.fit_predict(x)
    other_td = {**SMALL, "intermediate_dim_td": 0, "latent_dim_td": 1}
    other = TireDetector(**other_td, domain="FD")
    np.testing.assert_array_equal(other.fit_predict(x), pred)
    np.testing.assert_array_equal(other.change_scores_, reference.change_scores_)
