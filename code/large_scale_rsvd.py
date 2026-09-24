import numpy as np
import time
import matplotlib.pyplot as plt


# =========================================================
# 1. Generate a large image-like matrix
# =========================================================

def generate_large_matrix(m, n, rank=50, seed=42):

    rng = np.random.default_rng(seed)

    # Low-dimensional factors
    U = rng.standard_normal((m, rank)).astype(np.float32)
    V = rng.standard_normal((n, rank)).astype(np.float32)

    # Construct approximately low-rank matrix
    A = (U @ V.T).astype(np.float32)

    # Normalize to [0, 1]
    A -= A.min()
    A /= A.max()

    return A


# =========================================================
# 2. RSVD
# =========================================================

def rsvd(A, k, p=10, q=1, seed=42):

    rng = np.random.default_rng(seed)

    m, n = A.shape
    ell = k + p

    # Random test matrix
    Omega = rng.standard_normal((n, ell)).astype(np.float32)

    # Initial sample
    Y = A @ Omega

    # Orthonormal basis
    Q, _ = np.linalg.qr(Y, mode="reduced")

    # Power iterations
    for _ in range(q):

        Z = A.T @ Q
        Qz, _ = np.linalg.qr(Z, mode="reduced")

        Y = A @ Qz
        Q, _ = np.linalg.qr(Y, mode="reduced")

    # Reduced matrix
    B = Q.T @ A

    # Small SVD
    U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)

    # Recover left singular vectors
    U = Q @ U_tilde

    # Rank-k approximation
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    A_approx = (U_k * S_k) @ Vt_k

    return A_approx


# =========================================================
# 3. Relative Frobenius Error
# =========================================================

def relative_error(A, A_approx):

    return np.linalg.norm(A - A_approx) / np.linalg.norm(A)


# =========================================================
# 4. Main Experiment
# =========================================================

if __name__ == "__main__":

    # Matrix sizes
    sizes = [
        1000,
        2000,
        4000,
        6000,
        8000,
        10000
    ]

    k = 50
    p = 10
    q = 1

    results = []

    print("\n============================================================")
    print("        LARGE-SCALE RSVD SCALABILITY EXPERIMENT")
    print("============================================================")

    print(f"Rank k              : {k}")
    print(f"Oversampling p      : {p}")
    print(f"Power iterations q  : {q}")
    print("============================================================\n")


    for N in sizes:

        print(f"Processing matrix: {N} × {N}")

        # -----------------------------------------------------
        # Generate matrix
        # -----------------------------------------------------

        start_generation = time.perf_counter()

        A = generate_large_matrix(
            N,
            N,
            rank=50,
            seed=42
        )

        generation_time = time.perf_counter() - start_generation

        print(
            f"Matrix generated in : "
            f"{generation_time:.3f} seconds"
        )

        # -----------------------------------------------------
        # RSVD
        # -----------------------------------------------------

        start = time.perf_counter()

        A_approx = rsvd(
            A,
            k=k,
            p=p,
            q=q,
            seed=42
        )

        runtime = time.perf_counter() - start

        # -----------------------------------------------------
        # Error
        # -----------------------------------------------------

        error = relative_error(A, A_approx)

        # -----------------------------------------------------
        # Memory of original matrix
        # -----------------------------------------------------

        memory_MB = A.nbytes / (1024 ** 2)

        results.append(
            (N, runtime, error, memory_MB)
        )

        print(f"RSVD runtime        : {runtime:.3f} seconds")
        print(f"Relative error      : {error:.6f}")
        print(f"Matrix memory       : {memory_MB:.2f} MB")
        print("------------------------------------------------------------")

        # Release large arrays before next experiment
        del A
        del A_approx


    # =========================================================
    # Print final table
    # =========================================================

    print("\n============================================================")
    print("FINAL RESULTS")
    print("============================================================")

    print(
        f"{'Size':>12}"
        f"{'Runtime(s)':>15}"
        f"{'Rel.Error':>15}"
        f"{'Memory(MB)':>15}"
    )

    print("-" * 60)

    for N, runtime, error, memory_MB in results:

        print(
            f"{N:>12}"
            f"{runtime:>15.3f}"
            f"{error:>15.6f}"
            f"{memory_MB:>15.2f}"
        )


    # =========================================================
    # Extract results
    # =========================================================

    sizes_plot = [x[0] for x in results]
    runtime_plot = [x[1] for x in results]
    error_plot = [x[2] for x in results]
    memory_plot = [x[3] for x in results]


    # =========================================================
    # Plot 1: Runtime vs Matrix Size
    # =========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        sizes_plot,
        runtime_plot,
        marker="o",
        linewidth=2
    )

    plt.xlabel("Matrix Size N (N × N)")
    plt.ylabel("RSVD Runtime (seconds)")
    plt.title("RSVD Runtime vs Matrix Size")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "../Plots/runtime_vs_matrix_size.png",
        dpi=300
    )


    # =========================================================
    # Plot 2: Relative Error vs Matrix Size
    # =========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        sizes_plot,
        error_plot,
        marker="o",
        linewidth=2
    )

    plt.xlabel("Matrix Size N (N × N)")
    plt.ylabel("Relative Frobenius Error")
    plt.title("RSVD Relative Error vs Matrix Size")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "../Plots/relative_error_vs_matrix_size.png",
        dpi=300
    )


    # =========================================================
    # Plot 3: Memory vs Matrix Size
    # =========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        sizes_plot,
        memory_plot,
        marker="o",
        linewidth=2
    )

    plt.xlabel("Matrix Size N (N × N)")
    plt.ylabel("Matrix Memory (MB)")
    plt.title("Matrix Memory vs Matrix Size")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "../Plots/memory_vs_matrix_size.png",
        dpi=300
    )


    # =========================================================
    # Show all plots
    # =========================================================

    plt.show()