import numpy as np


# ---------------------------------------------------------
# 1. Generate a matrix with a chosen singular-value spectrum
# ---------------------------------------------------------

def generate_matrix(m, n, singular_values, seed=42):

    rng = np.random.default_rng(seed)

    # Random orthogonal matrix U
    G1 = rng.standard_normal((m, m))
    U, _ = np.linalg.qr(G1)

    # Random orthogonal matrix V
    G2 = rng.standard_normal((n, n))
    V, _ = np.linalg.qr(G2)

    r = min(m, n)

    # Construct rectangular Sigma
    Sigma = np.zeros((m, n))
    Sigma[:r, :r] = np.diag(singular_values[:r])

    # Construct A = U Sigma V^T
    A = U @ Sigma @ V.T

    return A


# ---------------------------------------------------------
# 2. Stable power iteration
# ---------------------------------------------------------

def power_iteration(A, Omega, q):

    # Initial random sketch
    Y = A @ Omega

    # Initial QR
    Q, _ = np.linalg.qr(Y, mode="reduced")

    # Power iterations
    for _ in range(q):

        Z = A.T @ Q

        Z, _ = np.linalg.qr(Z, mode="reduced")

        Y = A @ Z

        Q, _ = np.linalg.qr(Y, mode="reduced")

    return Q


# ---------------------------------------------------------
# 3. Complete RSVD
# ---------------------------------------------------------

def rsvd(A, k, p, q, seed=42):

    m, n = A.shape

    ell = k + p

    rng = np.random.default_rng(seed)

    # Random test matrix
    Omega = rng.standard_normal((n, ell))

    # Find approximate range of A
    Q = power_iteration(A, Omega, q)

    # Compress A
    B = Q.T @ A

    # SVD of small matrix B
    U_tilde, S, Vt = np.linalg.svd(
        B,
        full_matrices=False
    )

    # Recover approximate left singular vectors
    U = Q @ U_tilde

    # Keep only k components
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    # Reconstruct rank-k approximation
    A_approx = U_k @ np.diag(S_k) @ Vt_k

    return A_approx, S_k


# ---------------------------------------------------------
# 4. Error calculation
# ---------------------------------------------------------

def relative_frobenius_error(A, A_approx):

    error = np.linalg.norm(
        A - A_approx,
        "fro"
    )

    relative_error = error / np.linalg.norm(A, "fro")

    return error, relative_error


# ---------------------------------------------------------
# 5. Main experiment
# ---------------------------------------------------------

m = 500
n = 300
k = 20
p = 10

r = min(m, n)

# Singular-value spectra
sigma_exp = np.exp(-0.15 * np.arange(r))

sigma_poly = np.arange(1, r + 1) ** (-1.0)

sigma_slow = np.arange(1, r + 1) ** (-0.1)


spectra = {
    "Exponential": sigma_exp,
    "Polynomial": sigma_poly,
    "Slow": sigma_slow
}


# ---------------------------------------------------------
# Run experiment
# ---------------------------------------------------------

for name, singular_values in spectra.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    # Generate same A for all q values
    A = generate_matrix(
        m,
        n,
        singular_values,
        seed=42
    )

    # Classical SVD
    U, S, Vt = np.linalg.svd(
        A,
        full_matrices=False
    )

    # Optimal rank-k approximation
    A_opt = (
        U[:, :k]
        @ np.diag(S[:k])
        @ Vt[:k, :]
    )

    _, optimal_relative_error = relative_frobenius_error(
        A,
        A_opt
    )

    print(
        f"Optimal rank-{k} relative error = "
        f"{optimal_relative_error:.6f}"
    )

    # RSVD for different power iterations
    for q in [0, 1, 2, 3]:

        A_rsvd, S_rsvd = rsvd(
            A,
            k=k,
            p=p,
            q=q,
            seed=123
        )

        error, relative_error = relative_frobenius_error(
            A,
            A_rsvd
        )

        ratio = relative_error / optimal_relative_error

        print(
            f"q = {q} | "
            f"RSVD Error = {error:.6f} | "
            f"Relative Error = {relative_error:.6f} | "
            f"RSVD/Optimal = {ratio:.6f}"
        )
