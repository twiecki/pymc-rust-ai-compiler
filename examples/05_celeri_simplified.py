"""Example 5: Simplified celeri-style geoscience model.

Inspired by https://github.com/brendanjmeade/celeri/blob/main/celeri/solve_mcmc.py
A tectonic block model: estimate block rotation rates and fault slip rates
from GPS station velocities.

Key features that go beyond previous examples:
- StudentT likelihood (heavy-tailed, robust to outliers)
- Linear operator (design matrix @ parameters)
- Censored observations (bounded fault slip rates)
- Mixed Normal + StudentT likelihoods
- Matrix-vector multiplication in the likelihood

PyMC model:
    # Block rotation rates (3 components per block)
    rotation_raw ~ Normal(0, 1, shape=n_blocks*3)
    rotation = rotation_raw * rotation_scale

    # Fault slip rates (2 components: strike-slip + dip-slip)
    slip_rate ~ Normal(0, slip_prior_sigma, shape=n_faults*2)

    # GPS station velocities (StudentT for robustness)
    predicted_velocity = G_rotation @ rotation + G_slip @ slip_rate
    station_velocity ~ StudentT(nu=6, mu=predicted_velocity, sigma=sigma_obs, observed)

    # Geologic slip rate constraints (Censored Normal)
    Censored(Normal(mu=slip_rate[bounded], sigma=bound_sigma),
             lower=lower_bounds, upper=upper_bounds, observed=bound_values)

    # Regularization on slip rates
    slip_regularization ~ StudentT(nu=5, mu=0, sigma=gamma, observed=slip_rate)
"""

import numpy as np
import pymc as pm

from pymc_rust_compiler import compile_model

# --- Synthetic tectonic data ---
np.random.seed(42)

n_blocks = 3        # tectonic blocks
n_faults = 4        # fault segments
n_stations = 25     # GPS stations
n_bounded = 3       # faults with geologic slip rate bounds

# True parameters
true_rotation = np.array([
    0.5, -0.3, 0.1,   # Block 1: wx, wy, wz (rad/Gyr)
    -0.2, 0.4, -0.1,  # Block 2
    0.1, -0.1, 0.3,   # Block 3
])
rotation_scale = np.array([1.0, 1.0, 0.5] * n_blocks)  # prior scales

true_slip = np.array([
    2.0, 0.5,    # Fault 1: strike-slip, dip-slip (mm/yr)
    -1.5, 1.0,   # Fault 2
    0.8, -0.3,   # Fault 3
    -0.5, 0.2,   # Fault 4
])
slip_prior_sigma = 5.0

# Design matrices (Green's functions)
# G_rotation: how block rotations produce station velocities
G_rotation = np.random.randn(n_stations * 2, n_blocks * 3) * 0.5
# G_slip: how fault slip rates produce station velocities
G_slip = np.random.randn(n_stations * 2, n_faults * 2) * 0.3

# Observed GPS velocities (with outliers)
true_velocity = G_rotation @ true_rotation + G_slip @ true_slip
sigma_obs = np.abs(np.random.normal(0.8, 0.2, n_stations * 2))
velocity_obs = true_velocity + np.random.normal(0, sigma_obs)
# Add a few outliers (heavy tails → StudentT is important)
outlier_idx = np.random.choice(n_stations * 2, 3, replace=False)
velocity_obs[outlier_idx] += np.random.normal(0, 5, 3)

# Geologic slip rate bounds (e.g., from paleoseismology)
bounded_fault_idx = np.array([0, 2, 4])  # indices into slip_rate vector
lower_bounds = np.array([0.5, 0.0, -2.0])
upper_bounds = np.array([4.0, 2.0, 1.0])
bound_sigma = np.array([0.5, 0.5, 0.5])
bound_values = np.clip(true_slip[bounded_fault_idx], lower_bounds, upper_bounds)

# Regularization
gamma = 2.0  # regularization strength

print(f"Tectonic block model:")
print(f"  {n_blocks} blocks ({n_blocks * 3} rotation params)")
print(f"  {n_faults} faults ({n_faults * 2} slip rate params)")
print(f"  {n_stations} GPS stations ({n_stations * 2} velocity observations)")
print(f"  {n_bounded} faults with geologic bounds")
print(f"  Total params: {n_blocks * 3 + n_faults * 2}")
print()

SOURCE = f"""
# Tectonic block model (celeri-style)
# {n_blocks} blocks, {n_faults} faults, {n_stations} GPS stations

# Non-centered rotation rates
rotation_raw ~ Normal(0, 1, shape={n_blocks * 3})
rotation = rotation_raw * rotation_scale

# Fault slip rates
slip_rate ~ Normal(0, {slip_prior_sigma}, shape={n_faults * 2})

# GPS velocity: StudentT for robustness to outliers
predicted = G_rotation @ rotation + G_slip @ slip_rate
station_velocity ~ StudentT(nu=6, mu=predicted, sigma=sigma_obs, observed)

# Geologic bounds: Censored Normal
Censored(Normal(mu=slip_rate[bounded_idx], sigma=bound_sigma),
         lower=lower_bounds, upper=upper_bounds, observed=bound_values)

# Slip rate regularization (Potential with StudentT logp)
slip_regularization = Potential(StudentT_logp(nu=5, mu=0, sigma={gamma}, x=slip_rate))
"""

with pm.Model() as model:
    # Block rotation rates (non-centered parameterization)
    rotation_raw = pm.Normal("rotation_raw", mu=0, sigma=1, shape=n_blocks * 3)
    rotation = rotation_raw * rotation_scale

    # Fault slip rates
    slip_rate = pm.Normal("slip_rate", mu=0, sigma=slip_prior_sigma, shape=n_faults * 2)

    # Predicted GPS velocities via design matrices
    predicted_velocity = (
        pm.math.dot(G_rotation, rotation)
        + pm.math.dot(G_slip, slip_rate)
    )

    # GPS station velocity likelihood (StudentT for heavy tails)
    pm.StudentT(
        "station_velocity",
        nu=6,
        mu=predicted_velocity,
        sigma=sigma_obs,
        observed=velocity_obs,
    )

    # Geologic slip rate bounds (Censored Normal)
    bounded_slip = slip_rate[bounded_fault_idx]
    censored_dist = pm.Normal.dist(mu=bounded_slip, sigma=bound_sigma)
    pm.Censored(
        "geologic_bounds",
        dist=censored_dist,
        lower=lower_bounds,
        upper=upper_bounds,
        observed=bound_values,
    )

    # Slip rate regularization via Potential (StudentT-like penalty)
    # Can't use slip_rate as observed, so add logp directly
    slip_logp = pm.logp(pm.StudentT.dist(nu=5, mu=0, sigma=gamma), slip_rate).sum()
    pm.Potential("slip_regularization", slip_logp)

n_free = sum(v.size for v in model.initial_point().values())
print(f"Free RVs: {[rv.name for rv in model.free_RVs]}")
print(f"Unconstrained parameters: {n_free}")
print()

result = compile_model(
    model,
    source_code=SOURCE,
    build_dir="compiled_models/celeri",
    verbose=True,
)

if result.success:
    print(f"\nCompilation successful!")
    print(f"  Builds: {result.n_attempts}")
    print(f"  Tool calls: {result.n_tool_calls}")
    print(f"  Turns: {result.conversation_turns}")

    # Sample with nutpie
    print("\nSampling with nutpie...")
    idata = pm.sample(
        draws=1000,
        tune=500,
        chains=4,
        nuts_sampler="nutpie",
        model=model,
        random_seed=42,
        progressbar=True,
    )

    import arviz as az

    print("\n--- Posterior summary (rotation) ---")
    print(az.summary(idata, var_names=["rotation_raw"]))

    print("\n--- Posterior summary (slip rates) ---")
    print(az.summary(idata, var_names=["slip_rate"]))

    print(f"\nTrue rotation: {true_rotation}")
    print(f"True slip rates: {true_slip}")
else:
    print(f"\nCompilation FAILED after {result.n_attempts} builds, {result.n_tool_calls} tool calls")
    for err in result.validation_errors[:5]:
        print(f"  - {err}")
