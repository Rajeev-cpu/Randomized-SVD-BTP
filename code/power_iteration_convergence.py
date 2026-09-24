import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time


# =========================================================
# 1. Generate matrix with a chosen singular-value spectrum
# =========================================================

def generate_matrix(m, n, singular_values, seed=42):

    rng = np.random.default_rng(seed)

    # Random orthogonal U
    G1 = rng.standard_normal((m, m))
    U, _ = np.linalg.qr(G1)

    # Random orthogonal V
    G2 = rng.standard_normal((n, n))
    V, _ = np.linalg.qr(G2)

    # Construct rectangular Sigma
    Sigma = np.zeros((m, n))

    r = min(m, n)

    Sigma[:r, :r] = np.diag(singular_values[:r])

    A = U @ Sigma @ V.T

    return A


# =========================================================
# 2. Stable RSVD with power iteration
# =========================================================

def rsvd(A, Omega, k, q):

    # -----------------------------------------------------
    # Step 1: Randomized sketch
    # -----------------------------------------------------

    Y = A @ Omega

    # Initial orthonormal basis
    Q, _ = np.linalg.qr(Y, mode="reduced")


    # -----------------------------------------------------
    # Step 2: Power iteration
    # -----------------------------------------------------

    for _ in range(q):

        # Apply A^T
        Z = A.T @ Q

        # Re-orthogonalize
        Z, _ = np.linalg.qr(Z, mode="reduced")

        # Apply A
        Y = A @ Z

        # Re-orthogonalize
        Q, _ = np.linalg.qr(Y, mode="reduced")


    # -----------------------------------------------------
    # Step 3: Reduced matrix
    # -----------------------------------------------------

    B = Q.T @ A


    # -----------------------------------------------------
    # Step 4: SVD of small matrix
    # -----------------------------------------------------

    U_tilde, S, Vt = np.linalg.svd(
        B,
        full_matrices=False
    )


    # -----------------------------------------------------
    # Step 5: Recover left singular vectors
    # -----------------------------------------------------

    U = Q @ U_tilde


    # -----------------------------------------------------
    # Step 6: Truncate to rank k
    # -----------------------------------------------------

    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]


    # -----------------------------------------------------
    # Step 7: Rank-k reconstruction
    # -----------------------------------------------------

    A_k = U_k @ np.diag(S_k) @ Vt_k

    return A_k


# =========================================================
# 3. Error functions
# =========================================================

def frobenius_error(A, A_k):

    return np.linalg.norm(A - A_k, "fro")


def relative_error(A, A_k):

    return (
        np.linalg.norm(A - A_k, "fro")
        / np.linalg.norm(A, "fro")
    )


# =========================================================
# 4. Experiment parameters
# =========================================================

m = 500
n = 300

k = 20
p = 10

ell = k + p

# Increase q to observe convergence
q_values = [0, 1, 2, 3, 4, 5, 6]


# =========================================================
# 5. Singular-value spectra
# =========================================================

r = min(m, n)

indices = np.arange(r)


# Fast / exponential decay
sigma_exp = np.exp(-0.15 * indices)


# Polynomial decay
sigma_poly = (indices + 1) ** (-1.0)


# Slow decay
sigma_slow = (indices + 1) ** (-0.1)


spectra = {

    "Exponential": sigma_exp,

    "Polynomial": sigma_poly,

    "Slow": sigma_slow
}


# =========================================================
# 6. Generate matrices
# =========================================================

matrices = {}

for name, sigma in spectra.items():

    matrices[name] = generate_matrix(
        m,
        n,
        sigma,
        seed=42
    )


# =========================================================
# 7. Generate ONE Omega
# =========================================================
#
# Same Omega is used for every q.
# This makes the comparison fair.
# =========================================================

rng = np.random.default_rng(123)

Omega = rng.standard_normal((n, ell))


# =========================================================
# 8. Storage for results
# =========================================================

all_results = {}


# =========================================================
# 9. Run convergence experiment
# =========================================================

for name, A in matrices.items():

    sigma = spectra[name]

    # -----------------------------------------------------
    # Optimal rank-k Frobenius error
    # Eckart-Young theorem
    # -----------------------------------------------------

    optimal_error = np.sqrt(
        np.sum(sigma[k:] ** 2)
    )

    optimal_relative_error = (
        optimal_error
        / np.linalg.norm(sigma)
    )


    results = []


    for q in q_values:

        start = time.perf_counter()


        # ---------------------------------------------
        # RSVD
        # ---------------------------------------------

        A_k = rsvd(
            A,
            Omega,
            k,
            q
        )


        runtime = time.perf_counter() - start


        # ---------------------------------------------
        # Errors
        # ---------------------------------------------

        error = frobenius_error(A, A_k)

        rel_error = relative_error(A, A_k)

        error_ratio = error / optimal_error


        # ---------------------------------------------
        # Theoretical spectral separation
        # ---------------------------------------------

        separation = (
            sigma[k - 1]
            / sigma[k]
        ) ** (2 * q + 1)


        results.append(
            (
                q,
                error,
                rel_error,
                error_ratio,
                runtime,
                separation
            )
        )


    all_results[name] = results


# =========================================================
# 10. Print results
# =========================================================

print("\n")
print("=" * 100)
print("POWER ITERATION CONVERGENCE EXPERIMENT")
print("=" * 100)


for name, results in all_results.items():

    print("\n")
    print("-" * 100)
    print(name)
    print("-" * 100)

    print(
        f"{'q':>4}"
        f"{'Frobenius Error':>20}"
        f"{'Relative Error':>20}"
        f"{'RSVD/Optimal':>20}"
        f"{'Runtime (s)':>15}"
        f"{'Separation':>25}"
    )

    for row in results:

        q = row[0]
        error = row[1]
        rel_error = row[2]
        ratio = row[3]
        runtime = row[4]
        separation = row[5]

        print(
            f"{q:>4}"
            f"{error:>20.8f}"
            f"{rel_error:>20.8f}"
            f"{ratio:>20.8f}"
            f"{runtime:>15.6f}"
            f"{separation:>25.4e}"
        )


# =========================================================
# 11. Plot relative error vs q
# =========================================================

plt.figure(figsize=(8, 5))

for name, results in all_results.items():

    q = [row[0] for row in results]

    errors = [row[2] for row in results]

    plt.plot(
        q,
        errors,
        marker="o",
        linewidth=2,
        label=name
    )

plt.xlabel("Number of Power Iterations (q)")
plt.ylabel("Relative Frobenius Error")

plt.title(
    "RSVD Convergence: Relative Error vs Power Iterations"
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "relative_error_vs_q.png",
    dpi=300
)

plt.show()


# =========================================================
# 12. Plot RSVD / Optimal error ratio
# =========================================================

plt.figure(figsize=(8, 5))

for name, results in all_results.items():

    q = [row[0] for row in results]

    ratios = [row[3] for row in results]

    plt.plot(
        q,
        ratios,
        marker="o",
        linewidth=2,
        label=name
    )


# Optimal rank-k error ratio = 1
plt.axhline(
    1.0,
    linestyle="--",
    linewidth=2,
    label="Optimal"
)


plt.xlabel("Number of Power Iterations (q)")

plt.ylabel(
    "RSVD Error / Optimal Rank-k Error"
)

plt.title(
    "Convergence Toward Optimal Rank-k Approximation"
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "error_ratio_vs_q.png",
    dpi=300
)

plt.show()


# =========================================================
# 13. Plot runtime vs q
# =========================================================

plt.figure(figsize=(8, 5))

for name, results in all_results.items():

    q = [row[0] for row in results]

    runtime = [row[4] for row in results]

    plt.plot(
        q,
        runtime,
        marker="o",
        linewidth=2,
        label=name
    )

plt.xlabel("Number of Power Iterations (q)")
plt.ylabel("Runtime (seconds)")

plt.title(
    "RSVD Runtime vs Power Iterations"
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "runtime_vs_q.png",
    dpi=300
)

plt.show()


print("\nExperiment completed.")
print("Three plots were saved:")
print("1. relative_error_vs_q.png")
print("2. error_ratio_vs_q.png")
print("3. runtime_vs_q.png")
