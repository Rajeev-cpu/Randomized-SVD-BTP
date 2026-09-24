import numpy as np

from controlled_matrices import create_matrix


# Dimensions
m = 500
n = 300
r = 100

i = np.arange(1, r + 1)

# Exponential singular values
alpha = 0.08
sigma_expected = np.exp(-alpha * i)

# Construct matrix
A = create_matrix(
    m,
    n,
    sigma_expected,
    seed=42
)

# Compute SVD
U, s, Vt = np.linalg.svd(
    A,
    full_matrices=False
)

print("Expected first 10 singular values:")
print(sigma_expected[:10])

print("\nComputed first 10 singular values:")
print(s[:10])

print("\nMaximum absolute difference:")

difference = np.max(
    np.abs(s[:r] - sigma_expected)
)

print(difference)