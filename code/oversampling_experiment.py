import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# OVERSAMPLING EXPERIMENT FOR RSVD
# ============================================================

# Matrix dimensions
m = 500
n = 300

# Target rank
k = 20

# Oversampling values
p_values = [0, 5, 10, 20, 30]

# ------------------------------------------------------------
# Generate test matrix
# ------------------------------------------------------------
rng = np.random.default_rng(42)
A = rng.standard_normal((m, n))

A_norm = np.linalg.norm(A, "fro")

# ------------------------------------------------------------
# Generate ONE large Omega
# Then use first ell columns for each p
# This makes the experiment reproducible and controlled.
# ------------------------------------------------------------
max_p = max(p_values)
max_ell = k + max_p

Omega_max = rng.standard_normal((n, max_ell))

# Store results for plotting
relative_errors = []
frobenius_errors = []
orthogonality_errors = []
ell_values = []

# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------
print("OVERSAMPLING EXPERIMENT")
print("-" * 85)

print(
    f"{'p':>5} "
    f"{'ell':>8} "
    f"{'Frobenius Error':>20} "
    f"{'Relative Error':>18} "
    f"{'Q Orthogonality':>20}"
)

print("-" * 85)

for p in p_values:

    # --------------------------------------------------------
    # Oversampled dimension
    # ell = k + p
    # --------------------------------------------------------
    ell = k + p

    # Use first ell columns of the same Omega
    Omega = Omega_max[:, :ell]

    # --------------------------------------------------------
    # Randomized sketch
    # Y = A Omega
    # --------------------------------------------------------
    Y = A @ Omega

    # --------------------------------------------------------
    # QR factorization
    # Y = QR
    # --------------------------------------------------------
    Q, R = np.linalg.qr(Y, mode="reduced")

    # --------------------------------------------------------
    # Randomized projection
    # A_projection = Q Q^T A
    # --------------------------------------------------------
    A_projection = Q @ (Q.T @ A)

    # --------------------------------------------------------
    # Frobenius error
    # --------------------------------------------------------
    error = np.linalg.norm(A - A_projection, "fro")

    # --------------------------------------------------------
    # Relative Frobenius error
    # --------------------------------------------------------
    relative_error = error / A_norm

    # --------------------------------------------------------
    # Check orthogonality of Q
    # ||Q^T Q - I||
    # --------------------------------------------------------
    orthogonality_error = np.linalg.norm(
        Q.T @ Q - np.eye(ell),
        "fro"
    )

    # Store results
    ell_values.append(ell)
    frobenius_errors.append(error)
    relative_errors.append(relative_error)
    orthogonality_errors.append(orthogonality_error)

    # Print results
    print(
        f"{p:5d} "
        f"{ell:8d} "
        f"{error:20.6f} "
        f"{relative_error:18.6f} "
        f"{orthogonality_error:20.6e}"
    )

print("-" * 85)


# ============================================================
# PLOT: OVERSAMPLING vs RELATIVE ERROR
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    p_values,
    relative_errors,
    marker="o",
    linewidth=2
)

plt.xlabel("Oversampling parameter p")
plt.ylabel("Relative Frobenius Error")
plt.title("Effect of Oversampling on RSVD Error")

plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig(
    "oversampling_relative_error.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# OPTIONAL: PLOT ell vs RELATIVE ERROR
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    ell_values,
    relative_errors,
    marker="o",
    linewidth=2
)

plt.xlabel(r"Sketch dimension $\ell = k+p$")
plt.ylabel("Relative Frobenius Error")
plt.title("Effect of Sketch Dimension on RSVD Error")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "sketch_dimension_vs_error.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
