import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# SPECTRAL DECAY EXPERIMENT
# RSVD rank ell vs Optimal rank k vs RSVD rank k
# ============================================================

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
m = 500
n = 300

r = min(m, n)

k = 20
p = 10
ell = k + p

q = 1

seed = 123

# Output folder
PLOTS_DIR = Path("Plots")
PLOTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. GENERATE MATRIX WITH GIVEN SINGULAR VALUES
# ============================================================

def generate_matrix(m, n, singular_values, seed=42):

    rng = np.random.default_rng(seed)

    # Random orthogonal U
    G1 = rng.standard_normal((m, m))
    U, _ = np.linalg.qr(G1)

    # Random orthogonal V
    G2 = rng.standard_normal((n, n))
    V, _ = np.linalg.qr(G2)

    # Construct Sigma
    r = min(m, n)

    Sigma = np.zeros((m, n))
    Sigma[:r, :r] = np.diag(singular_values)

    # A = U Sigma V^T
    A = U @ Sigma @ V.T

    return A


# ============================================================
# 2. RSVD FUNCTION
# ============================================================

def rsvd(A, Omega, target_rank, q=1):

    m, n = A.shape

    # --------------------------------------------------------
    # Initial randomized sketch
    # Y = A Omega
    # --------------------------------------------------------
    Y = A @ Omega

    # QR
    Q, _ = np.linalg.qr(Y, mode="reduced")

    # --------------------------------------------------------
    # Stable power iteration
    # --------------------------------------------------------
    for _ in range(q):

        # Z = A^T Q
        Z = A.T @ Q

        # QR
        Z, _ = np.linalg.qr(Z, mode="reduced")

        # Y = A Z
        Y = A @ Z

        # QR
        Q, _ = np.linalg.qr(Y, mode="reduced")

    # --------------------------------------------------------
    # Small matrix
    # B = Q^T A
    # --------------------------------------------------------
    B = Q.T @ A

    # --------------------------------------------------------
    # SVD of B
    # --------------------------------------------------------
    U_tilde, S, Vt = np.linalg.svd(
        B,
        full_matrices=False
    )

    # Recover left singular vectors
    U = Q @ U_tilde

    # --------------------------------------------------------
    # Truncate to desired rank
    # --------------------------------------------------------
    U = U[:, :target_rank]
    S = S[:target_rank]
    Vt = Vt[:target_rank, :]

    # Reconstruction
    A_approx = (
        U @
        np.diag(S) @
        Vt
    )

    return A_approx, S


# ============================================================
# 3. ERROR FUNCTION
# ============================================================

def relative_error(A, A_approx):

    return (
        np.linalg.norm(A - A_approx, "fro")
        /
        np.linalg.norm(A, "fro")
    )


# ============================================================
# 4. SPECTRAL DECAY DEFINITIONS
# ============================================================

i = np.arange(1, r + 1)

# Fast / exponential decay
sigma_exp = np.exp(-0.15 * (i - 1))

# Polynomial decay
sigma_poly = i ** (-1.0)

# Slow decay
sigma_slow = i ** (-0.1)


spectra = {
    "Exponential": sigma_exp,
    "Polynomial": sigma_poly,
    "Slow": sigma_slow
}


# ============================================================
# 5. COMMON RANDOM OMEGA
# ============================================================

rng = np.random.default_rng(seed)

# We need ell columns because ell = k+p
Omega = rng.standard_normal((n, ell))


# ============================================================
# 6. STORE RESULTS
# ============================================================

results = {}


# ============================================================
# 7. RUN EXPERIMENT
# ============================================================

for name, sigma in spectra.items():

    print("\n" + "=" * 75)
    print(name.upper())
    print("=" * 75)

    # --------------------------------------------------------
    # Generate matrix
    # --------------------------------------------------------
    A = generate_matrix(
        m,
        n,
        sigma,
        seed=42
    )

    # --------------------------------------------------------
    # --------------------------------------------------------
    # A. OPTIMAL RANK-k APPROXIMATION
    # --------------------------------------------------------
    #
    # By Eckart-Young theorem:
    #
    # ||A - A_k||_F =
    # sqrt(sum_{i=k+1} sigma_i^2)
    #
    # --------------------------------------------------------

    optimal_error_k = np.sqrt(
        np.sum(sigma[k:] ** 2)
    )

    optimal_relative_k = (
        optimal_error_k /
        np.linalg.norm(sigma)
    )

    # --------------------------------------------------------
    # B. RSVD RANK ell
    # --------------------------------------------------------
    #
    # ell = k+p = 30
    #
    # Here target rank = ell
    # --------------------------------------------------------

    A_rsvd_ell, _ = rsvd(
        A,
        Omega,
        target_rank=ell,
        q=q
    )

    rsvd_ell_error = np.linalg.norm(
        A - A_rsvd_ell,
        "fro"
    )

    rsvd_ell_relative = (
        rsvd_ell_error /
        np.linalg.norm(A, "fro")
    )

    # --------------------------------------------------------
    # C. RSVD RANK k
    # --------------------------------------------------------
    #
    # The sketch dimension is ell = k+p,
    # but final approximation is truncated to k.
    # --------------------------------------------------------

    A_rsvd_k, _ = rsvd(
        A,
        Omega,
        target_rank=k,
        q=q
    )

    rsvd_k_error = np.linalg.norm(
        A - A_rsvd_k,
        "fro"
    )

    rsvd_k_relative = (
        rsvd_k_error /
        np.linalg.norm(A, "fro")
    )

    # --------------------------------------------------------
    # D. RSVD rank-k / Optimal rank-k
    # --------------------------------------------------------

    ratio = (
        rsvd_k_error /
        optimal_error_k
    )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    results[name] = {
        "optimal_error_k": optimal_error_k,
        "optimal_relative_k": optimal_relative_k,

        "rsvd_ell_error": rsvd_ell_error,
        "rsvd_ell_relative": rsvd_ell_relative,

        "rsvd_k_error": rsvd_k_error,
        "rsvd_k_relative": rsvd_k_relative,

        "ratio": ratio
    }

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(f"k = {k}")
    print(f"p = {p}")
    print(f"ell = k+p = {ell}")
    print(f"q = {q}")

    print("\nErrors:")
    print(
        f"Optimal rank-{k} error      : "
        f"{optimal_error_k:.8f}"
    )

    print(
        f"RSVD rank-{ell} error       : "
        f"{rsvd_ell_error:.8f}"
    )

    print(
        f"RSVD rank-{k} error         : "
        f"{rsvd_k_error:.8f}"
    )

    print("\nRelative errors:")

    print(
        f"Optimal rank-{k} relative   : "
        f"{optimal_relative_k:.8f}"
    )

    print(
        f"RSVD rank-{ell} relative    : "
        f"{rsvd_ell_relative:.8f}"
    )

    print(
        f"RSVD rank-{k} relative      : "
        f"{rsvd_k_relative:.8f}"
    )

    print("\nRSVD / Optimal:")
    print(
        f"RSVD rank-{k} / Optimal rank-{k} = "
        f"{ratio:.8f}"
    )


# ============================================================
# 8. PRINT SUMMARY TABLE
# ============================================================

print("\n\n")
print("=" * 110)
print("SUMMARY")
print("=" * 110)

print(
    f"{'Spectrum':<15}"
    f"{'Optimal k':>15}"
    f"{'RSVD ell':>15}"
    f"{'RSVD k':>15}"
    f"{'RSVD k / Optimal':>20}"
)

print("-" * 110)

for name, result in results.items():

    print(
        f"{name:<15}"
        f"{result['optimal_relative_k']:>15.6f}"
        f"{result['rsvd_ell_relative']:>15.6f}"
        f"{result['rsvd_k_relative']:>15.6f}"
        f"{result['ratio']:>20.6f}"
    )

# ============================================================
# 9. PLOT: SINGULAR VALUE DECAY
#    x-axis: 1 to 40
#    y-axis: 0 to 1.2 with step 0.2
# ============================================================

plt.figure(figsize=(8, 5))

# Plot only first 40 singular values
num_plot = 40

for name, sigma in spectra.items():

    plt.plot(
        i[:num_plot],
        sigma[:num_plot],
        marker="o",
        markersize=3,
        linewidth=1.8,
        label=name
    )

# X-axis: 1 to 40
plt.xlim(1, 40)
plt.xticks(np.arange(1, 41, 5))

# Y-axis: 0 to 1.2 with step 0.2
plt.ylim(0, 1.2)
plt.yticks(np.arange(0, 1.21, 0.2))

plt.xlabel(r"Index $i$")
plt.ylabel(r"Singular value $\sigma_i$")
plt.title("Singular Value Spectral Decay")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "singular_value_decay.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. PLOT: RELATIVE ERROR COMPARISON
# ============================================================

names = list(results.keys())

optimal_values = [
    results[name]["optimal_relative_k"]
    for name in names
]

rsvd_ell_values = [
    results[name]["rsvd_ell_relative"]
    for name in names
]

rsvd_k_values = [
    results[name]["rsvd_k_relative"]
    for name in names
]

x = np.arange(len(names))
width = 0.25

plt.figure(figsize=(9, 5))

plt.bar(
    x - width,
    optimal_values,
    width,
    label=f"Optimal rank-{k}"
)

plt.bar(
    x,
    rsvd_ell_values,
    width,
    label=f"RSVD rank-{ell}"
)

plt.bar(
    x + width,
    rsvd_k_values,
    width,
    label=f"RSVD rank-{k}"
)

plt.xticks(x, names)

plt.xlabel("Spectral decay")
plt.ylabel("Relative Frobenius Error")

plt.title("Rank and Spectral Decay Comparison")

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "spectral_decay_error_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. PLOT: RSVD RANK-k / OPTIMAL RANK-k
# ============================================================

ratio_values = [
    results[name]["ratio"]
    for name in names
]

plt.figure(figsize=(8, 5))

plt.bar(
    names,
    ratio_values
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1.5,
    label="Optimal = 1"
)

plt.xlabel("Spectral decay")
plt.ylabel("RSVD rank-k / Optimal rank-k")

plt.title("RSVD Rank-k Error Relative to Optimal Rank-k")

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "rsvd_optimal_error_ratio.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
