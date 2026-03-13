"""Shared fixtures for transpailer tests."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest


@pytest.fixture
def normal_model():
    """Simple Normal model: mu ~ N(0,10), sigma ~ HalfNormal(5), y ~ N(mu, sigma)."""
    rng = np.random.default_rng(42)
    y_obs = rng.normal(5.0, 1.2, size=50)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)

    return model


@pytest.fixture
def linreg_model():
    """Linear regression: y = alpha + beta * x + noise."""
    rng = np.random.default_rng(42)
    n = 30
    x = rng.uniform(-2, 2, size=n)
    y = 1.5 + 2.0 * x + rng.normal(0, 0.5, size=n)

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        mu = alpha + beta * x
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    return model


@pytest.fixture
def hierarchical_model():
    """Hierarchical model with group-level effects."""
    rng = np.random.default_rng(42)
    n_groups = 4
    n_per_group = 20
    group_idx = np.repeat(np.arange(n_groups), n_per_group)
    true_mu = 5.0
    true_effects = rng.normal(0, 1.0, size=n_groups)
    y = rng.normal(true_mu + true_effects[group_idx], 0.5)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma_group = pm.HalfNormal("sigma_group", sigma=2)
        offset = pm.Normal("offset", mu=0, sigma=1, shape=n_groups)
        group_effect = mu + sigma_group * offset
        sigma_y = pm.HalfNormal("sigma_y", sigma=5)
        pm.Normal("y", mu=group_effect[group_idx], sigma=sigma_y, observed=y)

    return model
