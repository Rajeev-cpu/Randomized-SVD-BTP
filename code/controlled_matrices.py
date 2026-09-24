import numpy as np


def create_matrix(m, n, singular_values, seed=42):
    """
    Create A = U Sigma V^T with prescribed singular values.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.

    singular_values : array
        Desired singular values.

    seed : int
        Random seed.

    Returns
    -------
    A : ndarray
        Constructed matrix.
    """

    rng = np.random.default_rng(seed)

    # Number of singular values
    r = len(singular_values)

    # Random matrix for U
    G_U = rng.standard_normal((m, r))

    # Random matrix for V
    G_V = rng.standard_normal((n, r))

    # QR factorization
    U, _ = np.linalg.qr(G_U, mode="reduced")
    V, _ = np.linalg.qr(G_V, mode="reduced")

    # Construct Sigma
    Sigma = np.diag(singular_values)

    # Construct A
    A = U @ Sigma @ V.T

    return A


# --------------------------------------------------
# Parameters
# --------------------------------------------------

m = 500
n = 300

r = 100

# Singular-value index
i = np.arange(1, r + 1)


# --------------------------------------------------
# 1. Exponential decay
# --------------------------------------------------

alpha_exp = 0.08

sigma_exp = np.exp(-alpha_exp * i)

A_exp = create_matrix(m, n, sigma_exp, seed=42)


# --------------------------------------------------
# 2. Polynomial decay
# --------------------------------------------------

alpha_poly = 1.0

sigma_poly = i ** (-alpha_poly)

A_poly = create_matrix(m, n, sigma_poly, seed=42)


# --------------------------------------------------
# Print information
# --------------------------------------------------

print("Exponential decay matrix")
print("Shape:", A_exp.shape)
print("First 10 singular values:")
print(sigma_exp[:10])

print("\nPolynomial decay matrix")
print("Shape:", A_poly.shape)
print("First 10 singular values:")
print(sigma_poly[:10])