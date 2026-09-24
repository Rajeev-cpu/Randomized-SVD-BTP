import numpy as np


m = 500
n = 300
k = 20
p = 10

ell = k + p

rng = np.random.default_rng(42)

Omega = rng.standard_normal((n, ell))
U, S, Vt = np.linalg.svd(Omega, full_matrices=False)

print("\nSingular values of Omega:")
print(S)

print("\nLargest singular value:")
print(S[0])

print("\nSmallest singular value:")
print(S[-1])

print("\nCondition number:")
print(S[0] / S[-1])