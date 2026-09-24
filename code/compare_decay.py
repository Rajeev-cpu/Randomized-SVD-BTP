import numpy as np

# Number of singular values
r = 100

i = np.arange(1, r + 1)

# Exponential decay
alpha_exp = 0.08
sigma_exp = np.exp(-alpha_exp * i)

# Polynomial decay
alpha_poly = 1.0
sigma_poly = i ** (-alpha_poly)

# Print selected values
print(" i       Exponential       Polynomial")
print("----------------------------------------")

for j in [0, 4, 9, 19, 39, 59, 79, 99]:

    print(
        f"{i[j]:3d}     "
        f"{sigma_exp[j]:12.6f}     "
        f"{sigma_poly[j]:12.6f}"
    )