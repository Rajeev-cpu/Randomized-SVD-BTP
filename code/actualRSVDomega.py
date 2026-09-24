import numpy as np

m = 500
n = 300
k = 20
p = 10

ell = k + p

rng = np.random.default_rng(42)

Omega = rng.standard_normal((n, ell))

print("Shape of Omega:", Omega.shape)
print("Mean:", np.mean(Omega))
print("Variance:", np.var(Omega))
print("Rank:", np.linalg.matrix_rank(Omega))