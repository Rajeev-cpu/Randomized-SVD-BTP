import numpy as np

rng = np.random.default_rng(42)

n = 100
ell = 20

# Gaussian matrix
Omega = rng.standard_normal((n, ell))

# Generate a random orthogonal matrix
G = rng.standard_normal((n, n))
Q, R = np.linalg.qr(G)

# Rotate Omega
Omega_rotated = Q.T @ Omega

print("Original mean:", np.mean(Omega))
print("Rotated mean:", np.mean(Omega_rotated))

print("Original variance:", np.var(Omega))
print("Rotated variance:", np.var(Omega_rotated))

print("Original Frobenius norm:", np.linalg.norm(Omega, "fro"))
print("Rotated Frobenius norm:", np.linalg.norm(Omega_rotated, "fro"))