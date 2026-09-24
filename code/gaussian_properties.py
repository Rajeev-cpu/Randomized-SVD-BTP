import numpy as np

# Reproducibility
rng = np.random.default_rng(42)

# Generate Gaussian random numbers
x = rng.standard_normal(1000000)

print("Number of samples:", len(x))
print("Mean:", np.mean(x))
print("Variance:", np.var(x))
print("Standard deviation:", np.std(x))