import numpy as np


# Matrix dimensions
m = 500
n = 300

rng = np.random.default_rng(42)

A = rng.standard_normal((m, n))

# SVD
U, s, Vt = np.linalg.svd(
    A,
    full_matrices=False
)


print("=" * 60)
print(
    f"{'Rank k':>10} "
    f"{'Frobenius Error':>25} "
    f"{'Relative Error':>20}"
)
print("=" * 60)


for k in [5, 10, 20, 30, 50]:

    # Rank-k approximation
    A_k = (
        U[:, :k]
        @ np.diag(s[:k])
        @ Vt[:k, :]
    )

    # Error
    error = np.linalg.norm(
        A - A_k,
        "fro"
    )

    relative_error = (
        error /
        np.linalg.norm(A, "fro")
    )

    print(
        f"{k:>10} "
        f"{error:>25.6f} "
        f"{relative_error:>20.6f}"
    )


print("=" * 60)