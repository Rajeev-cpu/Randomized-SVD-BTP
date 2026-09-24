import numpy as np

# Matrix dimensions
n = 300

# Random number generator
rng = np.random.default_rng(42)

# Different sketch sizes
for ell in [5, 10, 20, 30, 40, 50, 75, 100]:

    # Generate Gaussian random matrix Omega of size n x ell
    Omega = rng.standard_normal((n, ell))

    # Compute numerical rank
    rank = np.linalg.matrix_rank(Omega)

    # Compute singular values
    singular_values = np.linalg.svd(Omega, compute_uv=False)

    # Largest and smallest singular values
    sigma_max = singular_values[0]
    sigma_min = singular_values[-1]

    # Condition number
    condition_number = sigma_max / sigma_min

    print(
        f"ell = {ell:3d} | "
        f"rank = {rank:3d} | "
        f"sigma_max = {sigma_max:8.3f} | "
        f"sigma_min = {sigma_min:8.3f} | "
        f"condition = {condition_number:8.3f}"
    )