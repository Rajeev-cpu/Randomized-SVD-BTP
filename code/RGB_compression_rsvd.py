import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# =========================================================
# 1. Load RGB image
# =========================================================

image_path = r"C:\Users\user\OneDrive\Desktop\RSVD_Project\Images\some.jpeg"

image = Image.open(image_path).convert("RGB")

A_uint8 = np.array(image)

# Normalize pixel values to [0, 1]
A = A_uint8.astype(np.float64) / 255.0

m, n, channels = A.shape

print("Image size:", m, "x", n)
print("Channels:", channels)


# =========================================================
# 2. RSVD function for one 2-D matrix
# =========================================================

def rsvd(A, k, p=10, q=1, seed=123):

    m, n = A.shape

    ell = k + p

    rng = np.random.default_rng(seed)

    # Random test matrix
    Omega = rng.standard_normal((n, ell))

    # Randomized sketch
    Y = A @ Omega

    # Initial orthonormal basis
    Q, _ = np.linalg.qr(
        Y,
        mode="reduced"
    )

    # Power iteration
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

    # Small compressed matrix
    B = Q.T @ A

    # SVD of compressed matrix
    U_tilde, S, Vt = np.linalg.svd(
        B,
        full_matrices=False
    )

    # Recover left singular vectors
    U = Q @ U_tilde

    # Truncate to rank k
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    # Reconstruction
    A_approx = (
        U_k
        @ np.diag(S_k)
        @ Vt_k
    )

    return A_approx


# =========================================================
# 3. RGB RSVD function
# =========================================================

def rsvd_rgb(A, k, p=10, q=1, seed=123):

    # Separate RGB channels
    R = A[:, :, 0]
    G = A[:, :, 1]
    B = A[:, :, 2]

    # Apply RSVD independently to each channel
    R_approx = rsvd(
        R,
        k=k,
        p=p,
        q=q,
        seed=seed
    )

    G_approx = rsvd(
        G,
        k=k,
        p=p,
        q=q,
        seed=seed
    )

    B_approx = rsvd(
        B,
        k=k,
        p=p,
        q=q,
        seed=seed
    )

    # Combine reconstructed channels
    A_approx = np.stack(
        [
            R_approx,
            G_approx,
            B_approx
        ],
        axis=2
    )

    return A_approx


# =========================================================
# 4. Error metrics
# =========================================================

def relative_frobenius_error(A, A_approx):

    return (
        np.linalg.norm(
            A - A_approx
        )
        /
        np.linalg.norm(
            A
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


# =========================================================
# 5. Compression ratio
# =========================================================

def compression_ratio(m, n, k):

    # Three RGB channels
    original_values = 3 * m * n

    # Each channel stores:
    # U_k = m*k
    # Sigma_k = k
    # V_k^T = k*n

    compressed_values = (
        3 * k * (m + n + 1)
    )

    return (
        original_values
        /
        compressed_values
    )


# =========================================================
# 6. Test different ranks
# =========================================================

k_values = [10, 20, 50, 100,500]

results = []


for k in k_values:

    print("\n" + "=" * 60)

    print("Rank k =", k)

    # RGB RSVD approximation
    A_approx = rsvd_rgb(
        A,
        k=k,
        p=10,
        q=1,
        seed=123
    )

    # Clip values to valid image range
    A_approx = np.clip(
        A_approx,
        0,
        1
    )

    # Calculate metrics
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

    # Print results
    print(
        f"Relative Error      = {rel_error:.6f}"
    )

    print(
        f"MSE                 = {mse:.8f}"
    )

    print(
        f"PSNR                = {psnr:.4f} dB"
    )

    print(
        f"Compression Ratio   = {cr:.2f} x"
    )

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
# 7. Display reconstructed RGB images
# =========================================================

fig, axes = plt.subplots(
    1,
    len(k_values) + 1,
    figsize=(18, 4)
)


# Original
axes[0].imshow(A)

axes[0].set_title(
    "Original RGB"
)

axes[0].axis("off")


# Reconstructed images
for i, k in enumerate(k_values):

    A_approx = rsvd_rgb(
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
        A_approx
    )

    axes[i + 1].set_title(
        f"Rank k={k}"
    )

    axes[i + 1].axis("off")


plt.tight_layout()

plt.show()


# =========================================================
# 8. Final results table
# =========================================================

print("\n")

print("=" * 80)

print("FINAL RGB IMAGE COMPRESSION RESULTS")

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