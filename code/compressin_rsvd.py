import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_DIR = Path(r"C:\Users\user\OneDrive\Desktop\RSVD_Project")

image_path = PROJECT_DIR / "Images" / "new.webp"

plots_dir = PROJECT_DIR / "Plots"
plots_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. Load grayscale image
# =========================================================

image = Image.open(image_path).convert("L")

A_uint8 = np.array(image)

A = A_uint8.astype(np.float64) / 255.0

m, n = A.shape

print("Image size:", m, "x", n)


# =========================================================
# 3. RSVD function
# =========================================================

def rsvd(A, k, p=10, q=1, seed=123):

    m, n = A.shape

    # Oversampled target dimension
    ell = k + p

    rng = np.random.default_rng(seed)

    # -----------------------------------------------------
    # Random test matrix
    # -----------------------------------------------------

    Omega = rng.standard_normal((n, ell))

    # -----------------------------------------------------
    # Randomized sketch
    # -----------------------------------------------------

    Y = A @ Omega

    # -----------------------------------------------------
    # Initial orthonormal basis
    # -----------------------------------------------------

    Q, _ = np.linalg.qr(
        Y,
        mode="reduced"
    )

    # -----------------------------------------------------
    # Power iteration
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
    # Small compressed matrix
    # -----------------------------------------------------

    B = Q.T @ A

    # -----------------------------------------------------
    # SVD of compressed matrix
    # -----------------------------------------------------

    U_tilde, S, Vt = np.linalg.svd(
        B,
        full_matrices=False
    )

    # -----------------------------------------------------
    # Recover left singular vectors
    # -----------------------------------------------------

    U = Q @ U_tilde

    # -----------------------------------------------------
    # Truncate to rank k
    # -----------------------------------------------------

    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    # -----------------------------------------------------
    # Reconstruct image
    # -----------------------------------------------------

    A_approx = (
        U_k
        @ np.diag(S_k)
        @ Vt_k
    )

    return A_approx


# =========================================================
# 4. Error metrics
# =========================================================

def relative_frobenius_error(A, A_approx):

    return (
        np.linalg.norm(
            A - A_approx,
            "fro"
        )
        /
        np.linalg.norm(
            A,
            "fro"
        )
    )


def calculate_mse(A, A_approx):

    return np.mean(
        (A - A_approx) ** 2
    )


def calculate_psnr(mse):

    if mse == 0:
        return float("inf")

    MAX = 1.0

    return 10 * np.log10(
        (MAX ** 2) / mse
    )


def compression_ratio(m, n, k):

    # Original image stores mn values.
    original_values = m * n

    # Low-rank representation stores:
    #
    # U_k       -> m*k
    # Sigma_k   -> k
    # V_k^T     -> k*n
    #
    # Total = mk + k + kn
    compressed_values = k * (m + n + 1)

    return (
        original_values
        /
        compressed_values
    )


# =========================================================
# 5. Test different ranks
# =========================================================

k_values = [10, 20, 50, 100,1000]

results = []


for k in k_values:

    print("\n" + "=" * 60)

    print("Rank k =", k)

    # -----------------------------------------------------
    # RSVD approximation
    # -----------------------------------------------------

    A_approx = rsvd(
        A,
        k=k,
        p=10,
        q=2,
        seed=123
    )

    # Clip numerical values to valid image range
    A_approx = np.clip(
        A_approx,
        0,
        1
    )

    # -----------------------------------------------------
    # Calculate metrics
    # -----------------------------------------------------

    rel_error = relative_frobenius_error(
        A,
        A_approx
    )

    mse = calculate_mse(
        A,
        A_approx
    )

    psnr = calculate_psnr(
        mse
    )

    cr = compression_ratio(
        m,
        n,
        k
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print(
        f"Relative Error       = {rel_error:.6f}"
    )

    print(
        f"MSE                  = {mse:.8f}"
    )

    print(
        f"PSNR                 = {psnr:.4f} dB"
    )

    print(
        f"Compression Ratio    = {cr:.2f} x"
    )

    # -----------------------------------------------------
    # Store results
    # -----------------------------------------------------

    results.append(
        (
            k,
            rel_error,
            mse,
            psnr,
            cr
        )
    )


# =========================================================
# 6. Convert results to arrays
# =========================================================

k_results = np.array(
    [row[0] for row in results]
)

error_results = np.array(
    [row[1] for row in results]
)

mse_results = np.array(
    [row[2] for row in results]
)

psnr_results = np.array(
    [row[3] for row in results]
)

compression_results = np.array(
    [row[4] for row in results]
)


# =========================================================
# 7. Display reconstructed images
# =========================================================

fig, axes = plt.subplots(
    1,
    len(k_values) + 1,
    figsize=(18, 4)
)

# Original image
axes[0].imshow(
    A,
    cmap="gray"
)

axes[0].set_title(
    "Original"
)

axes[0].axis("off")


# Reconstructed images
for i, k in enumerate(k_values):

    A_approx = rsvd(
        A,
        k=k,
        p=10,
        q=1,
        seed=123
    )

    A_approx = np.clip(
        A_approx,
        0,
        1
    )

    axes[i + 1].imshow(
        A_approx,
        cmap="gray"
    )

    axes[i + 1].set_title(
        f"Rank k={k}"
    )

    axes[i + 1].axis("off")


plt.tight_layout()

# Save reconstructed image comparison
plt.savefig(
    plots_dir / "reconstructed_images.png",
    dpi=300,
    bbox_inches="tight"
)

# IMPORTANT:
# No plt.show() here


# =========================================================
# 8. Graph 1 — PSNR vs Rank
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_results,
    psnr_results,
    marker="o",
    linewidth=2
)

plt.xlabel("Rank k")
plt.ylabel("PSNR (dB)")
plt.title("PSNR vs Rank")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    plots_dir / "psnr_vs_rank.png",
    dpi=300,
    bbox_inches="tight"
)

# No plt.show() here


# =========================================================
# 9. Graph 2 — Compression Ratio vs Rank
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_results,
    compression_results,
    marker="o",
    linewidth=2
)

plt.xlabel("Rank k")
plt.ylabel("Compression Ratio")
plt.title("Compression Ratio vs Rank")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    plots_dir / "compression_ratio_vs_rank.png",
    dpi=300,
    bbox_inches="tight"
)

# No plt.show() here


# =========================================================
# 10. Graph 3 — Relative Frobenius Error vs Rank
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_results,
    error_results,
    marker="o",
    linewidth=2
)

plt.xlabel("Rank k")
plt.ylabel("Relative Frobenius Error")
plt.title("Relative Frobenius Error vs Rank")

plt.grid(True)

plt.tight_layout()

# CORRECT filename
plt.savefig(
    plots_dir / "relative_error_vs_rank.png",
    dpi=300,
    bbox_inches="tight"
)

# =========================================================
# 11. Final results table
# =========================================================

print("\n")

print("=" * 80)

print("FINAL IMAGE COMPRESSION RESULTS")

print("=" * 80)

print(
    f"{'Rank':<10}"
    f"{'Relative Error':<20}"
    f"{'MSE':<18}"
    f"{'PSNR (dB)':<15}"
    f"{'Compression Ratio':<20}"
)

print("-" * 80)

for row in results:

    k, error, mse, psnr, cr = row

    print(
        f"{k:<10}"
        f"{error:<20.6f}"
        f"{mse:<18.8f}"
        f"{psnr:<15.4f}"
        f"{cr:<20.2f}"
    )

print("=" * 80)

print("\nAll plots saved in:")

print(plots_dir)


# =========================================================
# 12. Show all figures at once
# =========================================================

plt.show()