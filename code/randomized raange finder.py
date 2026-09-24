import numpy as np

# Matrix dimensions
m = 500
n = 300

# Target rank
k = 20

# Reproducibility
np.random.seed(42)

# Generate matrix once
A = np.random.randn(m, n)

# Store norm of A
A_norm = np.linalg.norm(A, "fro")

# Print table heading
print("=" * 95)
print(f"{'p':>5} {'ell':>8} {'Frobenius Error':>20} "
      f"{'Relative Error':>20} {'Q Orthogonality Error':>25}")
print("=" * 95)

# Try different oversampling values
for p in [0, 5, 10, 20, 30]:

    # Sketch size
    ell = k + p

    # Generate Gaussian random matrix
    Omega = np.random.randn(n, ell)

    # Random sketch
    Y = A @ Omega

    # Orthonormal basis
    Q, R = np.linalg.qr(Y, mode="reduced")

    # Projection approximation
    A_projection = Q @ (Q.T @ A)

    # Frobenius error
    error = np.linalg.norm(A - A_projection, "fro")

    # Relative Frobenius error
    relative_error = error / A_norm

    # Check orthonormality of Q
    orthogonality_error = np.linalg.norm(
        Q.T @ Q - np.eye(ell)
    )

    # Print results
    print(f"{p:>5} {ell:>8} {error:>20.6f} "
          f"{relative_error:>20.6f} {orthogonality_error:>25.6e}")

print("=" * 95)