import numpy as np


# --------------------------------------------------
# Matrix dimensions
# --------------------------------------------------

m = 500
n = 300

k = 20
p = 10

ell = k + p


# --------------------------------------------------
# Reproducible random generator
# --------------------------------------------------

rng = np.random.default_rng(42)


# --------------------------------------------------
# Generate matrix A
# --------------------------------------------------

A = rng.standard_normal((m, n))


# --------------------------------------------------
# Generate Gaussian random matrix Omega
# --------------------------------------------------

Omega = rng.standard_normal((n, ell))


# --------------------------------------------------
# Random sketch
# --------------------------------------------------

Y = A @ Omega


# --------------------------------------------------
# QR factorization
# --------------------------------------------------

Q, R = np.linalg.qr(
    Y,
    mode="reduced"
)


# --------------------------------------------------
# Projection matrix
# --------------------------------------------------

P = Q @ Q.T


# --------------------------------------------------
# Test 1: Q^T Q = I
# --------------------------------------------------

orthogonality_error = np.linalg.norm(
    Q.T @ Q - np.eye(ell)
)


# --------------------------------------------------
# Test 2: P is symmetric
# --------------------------------------------------

symmetry_error = np.linalg.norm(
    P.T - P
)


# --------------------------------------------------
# Test 3: P is idempotent
# --------------------------------------------------

idempotence_error = np.linalg.norm(
    P @ P - P
)


# --------------------------------------------------
# Test 4: Projection residual is orthogonal
# --------------------------------------------------

residual = A - P @ A

residual_orthogonality = np.linalg.norm(
    Q.T @ residual
)


# --------------------------------------------------
# Test 5: Projection approximation error
# --------------------------------------------------

projection_error = np.linalg.norm(
    residual,
    "fro"
)

relative_error = (
    projection_error /
    np.linalg.norm(A, "fro")
)


# --------------------------------------------------
# Print table
# --------------------------------------------------

print("=" * 75)

print(
    f"{'Property':<45}"
    f"{'Error':>25}"
)

print("=" * 75)

print(
    f"{'||Q^T Q - I||':<45}"
    f"{orthogonality_error:>25.10e}"
)

print(
    f"{'||P^T - P||':<45}"
    f"{symmetry_error:>25.10e}"
)

print(
    f"{'||P^2 - P||':<45}"
    f"{idempotence_error:>25.10e}"
)

print(
    f"{'||Q^T(A - PA)||':<45}"
    f"{residual_orthogonality:>25.10e}"
)

print(
    f"{'Projection Frobenius error':<45}"
    f"{projection_error:>25.10f}"
)

print(
    f"{'Relative projection error':<45}"
    f"{relative_error:>25.10f}"
)

print("=" * 75)