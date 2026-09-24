import numpy as np


# --------------------------------------------------
# Matrix dimensions
# --------------------------------------------------

m = 500
n = 300

k = 20

seed = 42

rng = np.random.default_rng(seed)


# --------------------------------------------------
# Generate matrix
# --------------------------------------------------

A = rng.standard_normal((m, n))


# --------------------------------------------------
# Compute full SVD
# --------------------------------------------------

U, s, Vt = np.linalg.svd(
    A,
    full_matrices=False
)


# --------------------------------------------------
# Construct rank-k approximation
# --------------------------------------------------

Uk = U[:, :k]

Sk = np.diag(s[:k])

Vtk = Vt[:k, :]

A_k = Uk @ Sk @ Vtk


# --------------------------------------------------
# Frobenius error
# --------------------------------------------------

actual_error = np.linalg.norm(
    A - A_k,
    "fro"
)


# --------------------------------------------------
# Theoretical Eckart-Young error
# --------------------------------------------------

theoretical_error = np.sqrt(
    np.sum(s[k:] ** 2)
)


# --------------------------------------------------
# Spectral error
# --------------------------------------------------

spectral_error = np.linalg.norm(
    A - A_k,
    2
)

theoretical_spectral_error = s[k]


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("=" * 70)

print(f"{'Quantity':<40} {'Value':>20}")

print("=" * 70)

print(
    f"{'Actual Frobenius error':<40} "
    f"{actual_error:>20.10f}"
)

print(
    f"{'Theoretical Frobenius error':<40} "
    f"{theoretical_error:>20.10f}"
)

print(
    f"{'Difference':<40} "
    f"{abs(actual_error - theoretical_error):>20.10e}"
)

print()

print(
    f"{'Actual spectral error':<40} "
    f"{spectral_error:>20.10f}"
)

print(
    f"{'Theoretical spectral error':<40} "
    f"{theoretical_spectral_error:>20.10f}"
)

print(
    f"{'Difference':<40} "
    f"{abs(spectral_error - theoretical_spectral_error):>20.10e}"
)

print("=" * 70)