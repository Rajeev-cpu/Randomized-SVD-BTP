import numpy as np

m = 500
n = 300
k = 20

p_values = [0, 5, 10, 20, 30]
num_trials = 20

# Fixed matrix A
rng_A = np.random.default_rng(123)
A = rng_A.standard_normal((m, n))

A_norm = np.linalg.norm(A, "fro")

print("=" * 70)
print("Oversampling Experiment - Repeated Trials")
print("=" * 70)

for p in p_values:

    ell = k + p

    errors = []

    for trial in range(num_trials):

        rng = np.random.default_rng(trial)

        Omega = rng.standard_normal((n, ell))

        Y = A @ Omega

        Q, _ = np.linalg.qr(Y, mode="reduced")

        A_projection = Q @ (Q.T @ A)

        error = np.linalg.norm(A - A_projection, "fro")
        relative_error = error / A_norm

        errors.append(relative_error)

    mean_error = np.mean(errors)
    std_error = np.std(errors)

    print(
        f"p = {p:2d} | "
        f"ell = {ell:2d} | "
        f"Mean Error = {mean_error:.6f} | "
        f"Std = {std_error:.6f}"
    )