import numpy as np
import time

# Matrix dimensions
m = 500
n = 300

# Generate a reproducible random matrix
np.random.seed(42)
A = np.random.randn(m, n)

# Full/reduced classical SVD
start = time.perf_counter()

U, s, Vt = np.linalg.svd(A, full_matrices=False)

svd_time = time.perf_counter() - start

print("Matrix shape:", A.shape)
print("SVD time:", svd_time, "seconds")

# Test different approximation ranks
for k in [5, 10, 20, 50, 100]:

    # Truncated SVD
    Uk = U[:, :k]
    sk = s[:k]
    Vtk = Vt[:k, :]

    # Reconstruct A_k
    Ak = (Uk * sk) @ Vtk

    # Frobenius error
    error = np.linalg.norm(A - Ak, 'fro')

    # Relative Frobenius error
    relative_error = error / np.linalg.norm(A, 'fro')

    # Theoretical Eckart-Young error
    theoretical_error = np.sqrt(np.sum(s[k:]**2))

    print("\nRank k =", k)
    print("Frobenius error:", error)
    print("Theoretical error:", theoretical_error)
    print("Relative error:", relative_error)
