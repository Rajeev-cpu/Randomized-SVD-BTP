import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_DIR = Path(
    r"C:\Users\user\OneDrive\Desktop\RSVD_Project"
)

RESULTS_DIR = PROJECT_DIR / "Results"
PLOTS_DIR = PROJECT_DIR / "Plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. Generate a matrix with a chosen singular-value spectrum
# =========================================================

def generate_matrix(m, n, singular_values, seed=42):

    rng = np.random.default_rng(seed)

    # Random orthogonal matrix U
    G1 = rng.standard_normal((m, m))
    U, _ = np.linalg.qr(G1)

    # Random orthogonal matrix V
    G2 = rng.standard_normal((n, n))
    V, _ = np.linalg.qr(G2)

    # Rectangular singular-value matrix
    Sigma = np.zeros((m, n))

    r = min(m, n)

    Sigma[:r, :r] = np.diag(
        singular_values[:r]
    )

    # Construct A = U Sigma V^T
    A = U @ Sigma @ V.T

    return A


# =========================================================
# 3. Complete RSVD with stable power iteration
# =========================================================

def rsvd(A, Omega, k, q):

    # -----------------------------------------------------
    # Step 1: Randomized sketch
    # -----------------------------------------------------

    Y = A @ Omega

    # -----------------------------------------------------
    # Step 2: Initial orthonormal basis
    # -----------------------------------------------------

    Q, _ = np.linalg.qr(
        Y,
        mode="reduced"
    )

    # -----------------------------------------------------
    # Step 3: Stable power iteration
    # -----------------------------------------------------

    for _ in range(q):

        Z = A.T @ Q

        Z, _ = np.linalg.qr(
            Z,
            mode="reduced"
        )

        Y = A @ Z

        Q, _ = np.linalg.qr(
            Y,
            mode="reduced"
        )

    # -----------------------------------------------------
    # Step 4: Compressed matrix
    # -----------------------------------------------------

    B = Q.T @ A

    # -----------------------------------------------------
    # Step 5: SVD of compressed matrix
    # -----------------------------------------------------

    U_tilde, S, Vt = np.linalg.svd(
        B,
        full_matrices=False
    )

    # -----------------------------------------------------
    # Step 6: Recover approximate left singular vectors
    # -----------------------------------------------------

    U = Q @ U_tilde

    # -----------------------------------------------------
    # Step 7: Truncate to rank k
    # -----------------------------------------------------

    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    # -----------------------------------------------------
    # Step 8: Rank-k approximation
    # -----------------------------------------------------

    A_approx = (
        U_k
        @ np.diag(S_k)
        @ Vt_k
    )

    return A_approx


# =========================================================
# 4. Error functions
# =========================================================

def frobenius_error(A, A_approx):

    return np.linalg.norm(
        A - A_approx,
        "fro"
    )


def relative_frobenius_error(A, A_approx):

    return (
        frobenius_error(A, A_approx)
        /
        np.linalg.norm(A, "fro")
    )


# =========================================================
# 5. Parameters
# =========================================================

m = 500
n = 300

k = 20
p = 10

ell = k + p

q_values = [0, 1, 2, 3]

r = min(m, n)


# =========================================================
# 6. Singular-value spectra
# =========================================================

sigma_exp = np.exp(
    -0.15 * np.arange(r)
)

sigma_poly = (
    np.arange(1, r + 1) ** (-1.0)
)

sigma_slow = (
    np.arange(1, r + 1) ** (-0.1)
)


# =========================================================
# 7. Generate matrices
# =========================================================

A_exp = generate_matrix(
    m,
    n,
    sigma_exp,
    seed=42
)

A_poly = generate_matrix(
    m,
    n,
    sigma_poly,
    seed=42
)

A_slow = generate_matrix(
    m,
    n,
    sigma_slow,
    seed=42
)


matrices = {
    "Exponential": (A_exp, sigma_exp),
    "Polynomial": (A_poly, sigma_poly),
    "Slow": (A_slow, sigma_slow)
}


# =========================================================
# 8. Generate ONE random test matrix
# =========================================================

rng = np.random.default_rng(123)

Omega = rng.standard_normal(
    (n, ell)
)


# =========================================================
# 9. Run experiment
# =========================================================

all_results = {}


print("=" * 90)
print("DAY 18 - POWER ITERATION EXPERIMENT")
print("=" * 90)

print(
    f"Matrix size: {m} x {n}"
)

print(
    f"Target rank k = {k}"
)

print(
    f"Oversampling p = {p}"
)

print(
    f"Sample dimension ell = {ell}"
)

print(
    f"Power iterations q = {q_values}"
)


for name, (A, singular_values) in matrices.items():

    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)

    # -----------------------------------------------------
    # Optimal rank-k Frobenius error
    # -----------------------------------------------------

    optimal_error = np.sqrt(
        np.sum(
            singular_values[k:] ** 2
        )
    )

    optimal_relative_error = (
        optimal_error
        /
        np.linalg.norm(
            A,
            "fro"
        )
    )

    print(
        f"Optimal rank-{k} error = "
        f"{optimal_error:.6e}"
    )

    print(
        f"Optimal relative error = "
        f"{optimal_relative_error:.6e}"
    )

    results = []

    for q in q_values:

        # -------------------------------------------------
        # Measure runtime
        # -------------------------------------------------

        start_time = time.perf_counter()

        A_approx = rsvd(
            A,
            Omega,
            k,
            q
        )

        end_time = time.perf_counter()

        runtime = (
            end_time
            - start_time
        )

        # -------------------------------------------------
        # Calculate errors
        # -------------------------------------------------

        error = frobenius_error(
            A,
            A_approx
        )

        relative_error = (
            error
            /
            np.linalg.norm(
                A,
                "fro"
            )
        )

        # -------------------------------------------------
        # RSVD / optimal error ratio
        # -------------------------------------------------

        error_ratio = (
            error
            /
            optimal_error
        )

        results.append(
            (
                q,
                error,
                relative_error,
                error_ratio,
                runtime
            )
        )

        print(
            f"q = {q} | "
            f"Error = {error:.6e} | "
            f"Relative Error = {relative_error:.6e} | "
            f"Ratio = {error_ratio:.6f} | "
            f"Runtime = {runtime:.6f} s"
        )

    all_results[name] = {
        "results": results,
        "optimal_error": optimal_error,
        "optimal_relative_error": optimal_relative_error
    }


# =========================================================
# 10. Plot 1 - Relative Error vs q
# =========================================================

plt.figure(figsize=(8, 5))

for name, data in all_results.items():

    results = data["results"]

    q = [
        row[0]
        for row in results
    ]

    relative_errors = [
        row[2]
        for row in results
    ]

    plt.plot(
        q,
        relative_errors,
        marker="o",
        linewidth=2,
        label=name
    )

plt.xlabel(
    "Power Iterations (q)"
)

plt.ylabel(
    "Relative Frobenius Error"
)

plt.title(
    "Relative Frobenius Error vs Power Iterations"
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR /
    "relative_error_vs_q.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 11. Plot 2 - RSVD / Optimal Error Ratio vs q
# =========================================================

plt.figure(figsize=(8, 5))

for name, data in all_results.items():

    results = data["results"]

    q = [
        row[0]
        for row in results
    ]

    error_ratios = [
        row[3]
        for row in results
    ]

    plt.plot(
        q,
        error_ratios,
        marker="o",
        linewidth=2,
        label=name
    )

# Optimal reference
plt.axhline(
    y=1,
    linestyle="--",
    linewidth=1.5,
    label="Optimal ratio = 1"
)

plt.xlabel(
    "Power Iterations (q)"
)

plt.ylabel(
    "RSVD Error / Optimal Rank-k Error"
)

plt.title(
    "RSVD / Optimal Error Ratio vs Power Iterations"
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR /
    "error_ratio_vs_q.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 12. Plot 3 - Runtime vs q
# =========================================================

plt.figure(figsize=(8, 5))

for name, data in all_results.items():

    results = data["results"]

    q = [
        row[0]
        for row in results
    ]

    runtimes = [
        row[4]
        for row in results
    ]

    plt.plot(
        q,
        runtimes,
        marker="o",
        linewidth=2,
        label=name
    )

plt.xlabel(
    "Power Iterations (q)"
)

plt.ylabel(
    "Runtime (seconds)"
)

plt.title(
    "Runtime vs Power Iterations"
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR /
    "runtime_vs_q.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 13. Final summary
# =========================================================

print("\n")
print("=" * 100)
print("FINAL DAY-18 RESULTS")
print("=" * 100)

for name, data in all_results.items():

    print("\n" + name)

    print(
        f"{'q':<8}"
        f"{'Relative Error':<20}"
        f"{'Error Ratio':<20}"
        f"{'Runtime (s)':<15}"
    )

    print("-" * 65)

    for row in data["results"]:

        q = row[0]
        relative_error = row[2]
        error_ratio = row[3]
        runtime = row[4]

        print(
            f"{q:<8}"
            f"{relative_error:<20.6e}"
            f"{error_ratio:<20.6f}"
            f"{runtime:<15.6f}"
        )

print("\nAll plots saved in:")
print(PLOTS_DIR)