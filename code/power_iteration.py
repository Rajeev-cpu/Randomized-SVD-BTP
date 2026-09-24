import numpy as np


def generate_matrix(m, n, singular_values, seed=42):

    rng = np.random.default_rng(seed)

    G1 = rng.standard_normal((m, m))
    U, _ = np.linalg.qr(G1)

    G2 = rng.standard_normal((n, n))
    V, _ = np.linalg.qr(G2)

    Sigma = np.zeros((m, n))

    r = min(m, n)

    Sigma[:r, :r] = np.diag(singular_values[:r])

    A = U @ Sigma @ V.T

    return A


def power_iteration(A, Omega, q):

    Y = A @ Omega

    Q, _ = np.linalg.qr(Y, mode="reduced")

    for _ in range(q):

        Z = A.T @ Q

        Z, _ = np.linalg.qr(Z, mode="reduced")

        Y = A @ Z

        Q, _ = np.linalg.qr(Y, mode="reduced")

    return Q


def projection_error(A, Q):

    A_approx = Q @ (Q.T @ A)

    error = np.linalg.norm(A - A_approx, "fro")

    relative_error = error / np.linalg.norm(A, "fro")

    return error, relative_error


# --------------------------------------------------
# Parameters
# --------------------------------------------------

m = 500
n = 300

k = 20
p = 10
ell = k + p

r = min(m, n)


# --------------------------------------------------
# Singular-value spectra
# --------------------------------------------------

sigma_exp = np.exp(-0.15 * np.arange(r))

sigma_poly = np.arange(1, r + 1) ** (-1.0)

sigma_slow = np.arange(1, r + 1) ** (-0.1)


# --------------------------------------------------
# Generate matrices
# --------------------------------------------------

A_exp = generate_matrix(m, n, sigma_exp, seed=42)
A_poly = generate_matrix(m, n, sigma_poly, seed=42)
A_slow = generate_matrix(m, n, sigma_slow, seed=42)


matrices = {
    "Exponential": A_exp,
    "Polynomial": A_poly,
    "Slow": A_slow
}


# --------------------------------------------------
# Test different q
# --------------------------------------------------

q_values = [0, 1, 2, 3]

print("=" * 75)
print("Power Iteration Experiment")
print("=" * 75)

for name, A in matrices.items():

    print("\n" + name)

    for q in q_values:

        rng = np.random.default_rng(123)

        Omega = rng.standard_normal((n, ell))

        Q = power_iteration(A, Omega, q)

        error, relative_error = projection_error(A, Q)

        print(
            f"q = {q} | "
            f"Frobenius Error = {error:.6e} | "
            f"Relative Error = {relative_error:.6e}"
        )