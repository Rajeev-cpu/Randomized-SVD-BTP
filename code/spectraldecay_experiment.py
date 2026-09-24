# =========================================================
# spectral_decay_experiment.py
# =========================================================
#
# SPECTRAL DECAY EXPERIMENT FOR RSVD
#
# Purpose:
#   Study how different singular-value decay patterns
#   affect low-rank approximation and RSVD accuracy.
#
# We compare:
#
#   1. RSVD rank-ell projection:
#          A_proj = Q Q^T A
#
#      where ell = k + p
#
#   2. Optimal rank-k approximation:
#          A_k = U_k Sigma_k V_k^T
#
#      obtained from the known SVD of A.
#
#   3. Strict rank-k RSVD approximation:
#          A_k_RSV D = U_k Sigma_k V_k^T
#
# IMPORTANT:
#   The rank-ell projection is NOT a rank-k approximation when
#   ell > k. Therefore it must not be directly compared with
#   the optimal rank-k approximation as an EYM comparison.
#
# =========================================================


import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. EXPERIMENT PARAMETERS
# =========================================================

m = 500
n = 300

# Target rank
k = 20

# Oversampling parameter
p = 10

# Sampling dimension
ell = k + p

# Random seed
seed = 42


print("=" * 70)
print("SPECTRAL DECAY EXPERIMENT")
print("=" * 70)

print(f"Matrix dimensions : {m} x {n}")
print(f"Target rank k     : {k}")
print(f"Oversampling p    : {p}")
print(f"Sampling dimension ell = k + p = {ell}")
print(f"Random seed       : {seed}")

print()


# =========================================================
# 2. RANDOM NUMBER GENERATOR
# =========================================================

rng = np.random.default_rng(seed)


# =========================================================
# 3. REDUCED SVD DIMENSION
# =========================================================
#
# r = min(m,n)
#
# For:
#
#       m = 500
#       n = 300
#
# we have:
#
#       r = 300
#
# Therefore:
#
#       U     : 500 x 300
#       Sigma : 300 x 300
#       V     : 300 x 300
#
# and:
#
#       A = U Sigma V^T
#
#       A : 500 x 300
#
# =========================================================

r = min(m, n)


# =========================================================
# 4. GENERATE RANDOM ORTHOGONAL MATRIX U
# =========================================================
#
# Generate:
#
#       G1 : m x r
#
#       G1 : 500 x 300
#
# QR decomposition:
#
#       G1 = U R
#
# gives:
#
#       U : 500 x 300
#
# with orthonormal columns.
#
# =========================================================

G1 = rng.standard_normal((m, r))

U, _ = np.linalg.qr(
    G1,
    mode="reduced"
)


# =========================================================
# 5. GENERATE RANDOM ORTHOGONAL MATRIX V
# =========================================================
#
# Generate:
#
#       G2 : n x r
#
#       G2 : 300 x 300
#
# QR decomposition gives:
#
#       V : 300 x 300
#
# =========================================================

G2 = rng.standard_normal((n, r))

V, _ = np.linalg.qr(
    G2,
    mode="reduced"
)


# =========================================================
# 6. VERIFY DIMENSIONS
# =========================================================

print("Dimensions of fixed orthogonal factors:")
print(f"U      = {U.shape}")
print(f"V      = {V.shape}")
print()


# =========================================================
# 7. CREATE SINGULAR-VALUE SPECTRA
# =========================================================
#
# i = 1,2,...,r
#
# Three spectral decay patterns:
#
#   Exponential:
#       sigma_i = exp(-0.15(i-1))
#
#   Polynomial:
#       sigma_i = i^(-1)
#
#   Slow:
#       sigma_i = i^(-0.1)
#
# =========================================================

i = np.arange(1, r + 1)


# Exponential decay
sigma_exp = np.exp(
    -0.15 * (i - 1)
)


# Polynomial decay
sigma_poly = i ** (-1.0)


# Slow decay
sigma_slow = i ** (-0.1)


# Store spectra
spectra = {
    "Exponential": sigma_exp,
    "Polynomial": sigma_poly,
    "Slow": sigma_slow
}


# =========================================================
# 8. RELATIVE FROBENIUS ERROR FUNCTION
# =========================================================

def relative_frobenius_error(A, A_approx):
    """
    Compute relative Frobenius error:

        ||A - A_approx||_F
        -------------------
              ||A||_F
    """

    numerator = np.linalg.norm(
        A - A_approx,
        ord="fro"
    )

    denominator = np.linalg.norm(
        A,
        ord="fro"
    )

    return numerator / denominator


# =========================================================
# 9. STORAGE FOR RESULTS
# =========================================================

results = {}


# =========================================================
# 10. RUN EXPERIMENT FOR EACH SPECTRUM
# =========================================================

for name, sigma in spectra.items():

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)


    # =====================================================
    # 10.1 CONSTRUCT Sigma
    # =====================================================
    #
    # sigma has length 300.
    #
    # Therefore:
    #
    # Sigma : 300 x 300
    #
    # =====================================================

    Sigma = np.diag(sigma)

    print(f"Sigma  = {Sigma.shape}")


    # =====================================================
    # 10.2 CONSTRUCT MATRIX A
    # =====================================================
    #
    # A = U Sigma V^T
    #
    # Dimensions:
    #
    # U     : 500 x 300
    # Sigma : 300 x 300
    # V^T   : 300 x 300
    #
    # Therefore:
    #
    # A     : 500 x 300
    #
    # =====================================================

    A = U @ Sigma @ V.T

    print(f"A      = {A.shape}")


    # =====================================================
    # 10.3 OPTIMAL RANK-k APPROXIMATION
    # =====================================================
    #
    # Since we constructed A using its SVD, the first k
    # singular components give the exact optimal rank-k
    # approximation.
    #
    # A_k = U_k Sigma_k V_k^T
    #
    # U_k     : 500 x 20
    # Sigma_k : 20 x 20
    # V_k^T   : 20 x 300
    #
    # A_k     : 500 x 300
    #
    # =====================================================

    U_k_opt = U[:, :k]

    sigma_k_opt = sigma[:k]

    V_k_opt = V[:, :k]


    A_opt_k = (
        U_k_opt
        @ np.diag(sigma_k_opt)
        @ V_k_opt.T
    )


    # Calculate optimal rank-k relative error

    optimal_error = relative_frobenius_error(
        A,
        A_opt_k
    )


    # =====================================================
    # 10.4 GENERATE RANDOM TEST MATRIX Omega
    # =====================================================
    #
    # Omega : n x ell
    #
    # Omega : 300 x 30
    #
    # =====================================================

    Omega = rng.standard_normal(
        (n, ell)
    )

    print(f"Omega  = {Omega.shape}")


    # =====================================================
    # 10.5 RANDOMIZED RANGE FINDER
    # =====================================================
    #
    # Y = A Omega
    #
    # A     : 500 x 300
    # Omega : 300 x 30
    #
    # Y     : 500 x 30
    #
    # =====================================================

    Y = A @ Omega

    print(f"Y      = {Y.shape}")


    # =====================================================
    # 10.6 QR DECOMPOSITION
    # =====================================================
    #
    # Y = Q R
    #
    # Q : 500 x 30
    # R : 30 x 30
    #
    # =====================================================

    Q, R = np.linalg.qr(
        Y,
        mode="reduced"
    )

    print(f"Q      = {Q.shape}")
    print(f"R      = {R.shape}")


    # =====================================================
    # 10.7 RSVD RANK-ell PROJECTION
    # =====================================================
    #
    # A_projection = Q Q^T A
    #
    # Q^T : 30 x 500
    # A   : 500 x 300
    #
    # Q^T A : 30 x 300
    #
    # Q(Q^T A) : 500 x 300
    #
    # This approximation has rank at most ell = 30.
    #
    # =====================================================

    A_projection = Q @ (Q.T @ A)


    projection_error = relative_frobenius_error(
        A,
        A_projection
    )


    # =====================================================
    # 10.8 FORM SMALL MATRIX B
    # =====================================================
    #
    # B = Q^T A
    #
    # Q^T : 30 x 500
    # A   : 500 x 300
    #
    # B   : 30 x 300
    #
    # =====================================================

    B = Q.T @ A

    print(f"B      = {B.shape}")


    # =====================================================
    # 10.9 SVD OF SMALL MATRIX
    # =====================================================
    #
    # B = U_tilde Sigma V^T
    #
    # U_tilde : 30 x 30
    # Sigma   : 30
    # V^T     : 30 x 300
    #
    # =====================================================

    U_tilde, S, Vt = np.linalg.svd(
        B,
        full_matrices=False
    )


    print(f"U_tilde = {U_tilde.shape}")
    print(f"S       = {S.shape}")
    print(f"Vt      = {Vt.shape}")


    # =====================================================
    # 10.10 RECOVER APPROXIMATE LEFT SINGULAR VECTORS
    # =====================================================
    #
    # U_R = Q U_tilde
    #
    # Q         : 500 x 30
    # U_tilde   : 30 x 30
    #
    # U_R       : 500 x 30
    #
    # =====================================================

    U_rsvd = Q @ U_tilde

    print(f"U_rsvd = {U_rsvd.shape}")


    # =====================================================
    # 10.11 STRICT RANK-k RSVD APPROXIMATION
    # =====================================================
    #
    # Keep only the first k = 20 components.
    #
    # U_k_RSV D : 500 x 20
    # S_k       : 20
    # V_k^T     : 20 x 300
    #
    # =====================================================

    U_k_rsvd = U_rsvd[:, :k]

    S_k_rsvd = S[:k]

    Vt_k_rsvd = Vt[:k, :]


    # Reconstruct rank-k RSVD approximation

    A_rsvd_k = (
        U_k_rsvd
        @ np.diag(S_k_rsvd)
        @ Vt_k_rsvd
    )


    # =====================================================
    # 10.12 STRICT RANK-k RSVD ERROR
    # =====================================================

    rsvd_rank_k_error = relative_frobenius_error(
        A,
        A_rsvd_k
    )


    # =====================================================
    # 10.13 RSVD / OPTIMAL RATIO
    # =====================================================
    #
    # This is the correct ratio for a strict rank-k
    # comparison.
    #
    # Since the optimal rank-k approximation is the minimum
    # possible rank-k error, this ratio should be >= 1
    # up to numerical roundoff.
    #
    # =====================================================

    rsvd_optimal_ratio = (
        rsvd_rank_k_error
        / optimal_error
    )


    # =====================================================
    # 10.14 STORE RESULTS
    # =====================================================

    results[name] = {

        "projection_error":
            projection_error,

        "optimal_rank_k_error":
            optimal_error,

        "rsvd_rank_k_error":
            rsvd_rank_k_error,

        "rsvd_optimal_ratio":
            rsvd_optimal_ratio
    }


    # =====================================================
    # 10.15 PRINT RESULTS
    # =====================================================

    print()

    print(
        f"RSVD rank-{ell} projection error : "
        f"{projection_error:.6f}"
    )

    print(
        f"Optimal rank-{k} error            : "
        f"{optimal_error:.6f}"
    )

    print(
        f"Strict rank-{k} RSVD error        : "
        f"{rsvd_rank_k_error:.6f}"
    )

    print(
        f"Strict RSVD / Optimal ratio       : "
        f"{rsvd_optimal_ratio:.6f}"
    )


# =========================================================
# 11. FINAL RESULTS TABLE
# =========================================================

print()
print()
print("=" * 90)
print("FINAL SPECTRAL DECAY RESULTS")
print("=" * 90)

print(
    f"{'Spectrum':<15}"
    f"{'RSVD rank-ell':>18}"
    f"{'Optimal rank-k':>18}"
    f"{'RSVD rank-k':>18}"
    f"{'RSVD/Optimal':>18}"
)

print("-" * 90)

for name in results:

    r = results[name]

    print(
        f"{name:<15}"
        f"{r['projection_error']:>18.5f}"
        f"{r['optimal_rank_k_error']:>18.5f}"
        f"{r['rsvd_rank_k_error']:>18.5f}"
        f"{r['rsvd_optimal_ratio']:>18.5f}"
    )


# =========================================================
# 12. PLOT 1: SINGULAR-VALUE SPECTRAL DECAY
# =========================================================

plt.figure(figsize=(8, 5))

for name, sigma in spectra.items():

    plt.semilogy(
        i,
        sigma,
        linewidth=2,
        label=name
    )

# Target rank
plt.axvline(
    k,
    linestyle="--",
    linewidth=1.5,
    label=f"Target rank k={k}"
)

plt.xlabel("Singular-value index $i$")

plt.ylabel(
    "Singular value $\\sigma_i$"
)

plt.title(
    "Singular-Value Spectral Decay"
)

plt.legend()

plt.grid(
    True,
    which="both"
)

plt.tight_layout()

plt.savefig(
    "spectral_decay.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 13. PLOT 2: APPROXIMATION ERROR COMPARISON
# =========================================================

names = list(results.keys())

projection_errors = [
    results[name]["projection_error"]
    for name in names
]

optimal_errors = [
    results[name]["optimal_rank_k_error"]
    for name in names
]

rsvd_rank_k_errors = [
    results[name]["rsvd_rank_k_error"]
    for name in names
]


x = np.arange(
    len(names)
)

width = 0.25


plt.figure(figsize=(9, 5))

plt.bar(
    x - width,
    projection_errors,
    width,
    label=f"RSVD rank-$\\ell$ projection ($\\ell={ell}$)"
)

plt.bar(
    x,
    optimal_errors,
    width,
    label=f"Optimal rank-$k$ ($k={k}$)"
)

plt.bar(
    x + width,
    rsvd_rank_k_errors,
    width,
    label="Strict rank-$k$ RSVD"
)

plt.xticks(
    x,
    names
)

plt.ylabel(
    "Relative Frobenius Error"
)

plt.title(
    "Spectral Decay vs Approximation Error"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "spectral_decay_error_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 14. PLOT 3: STRICT RSVD / OPTIMAL RATIO
# =========================================================

ratios = [
    results[name]["rsvd_optimal_ratio"]
    for name in names
]


plt.figure(figsize=(8, 5))

plt.bar(
    names,
    ratios
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1.5,
    label="Optimal ratio = 1"
)

plt.ylabel(
    "Strict Rank-$k$ RSVD / Optimal Error"
)

plt.xlabel(
    "Spectral decay"
)

plt.title(
    "Strict Rank-$k$ RSVD vs Optimal Rank-$k$"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "rsvd_optimal_ratio_spectral_decay.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 15. FINAL INTERPRETATION
# =========================================================

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

print()
print("1. Exponential decay:")
print("   Singular values decrease rapidly.")
print("   A small rank captures most of the matrix information.")

print()
print("2. Polynomial decay:")
print("   Singular values decrease more gradually.")
print("   More components are required for accurate approximation.")

print()
print("3. Slow decay:")
print("   Many singular values remain significant.")
print("   Low-rank approximation is intrinsically more difficult.")

print()
print("4. Important rank distinction:")
print(
    f"   RSVD projection has rank <= ell = {ell}, "
    f"while the EYM benchmark has rank k = {k}."
)

print()
print("5. Strict rank-k comparison:")
print(
    "   RSVD rank-k error should be >= optimal rank-k error "
    "up to numerical roundoff."
)

print()
print("Experiment completed successfully.")