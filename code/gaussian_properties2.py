import numpy as np

rng = np.random.default_rng(42)

sample_sizes = [10, 100, 1000, 10000, 100000, 1000000]

print("Sample Size       Mean          Variance")
print("-" * 45)

for n in sample_sizes:
    x = rng.standard_normal(n)

    mean = np.mean(x)
    variance = np.var(x)

    print(f"{n:10d}   {mean:12.6f}   {variance:12.6f}")