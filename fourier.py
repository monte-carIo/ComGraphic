import numpy as np
import matplotlib.pyplot as plt

# Number of Fourier terms for the original function
n_terms_original = 20

# Number of Fourier terms for the approximation (fewer terms = less accuracy)
n_terms_approx = 10  # Reduce this to lower the accuracy

# Define the range for x values
x = np.linspace(-np.pi, np.pi, 1000)

# Generate a random target function (original function) as a sum of sinusoids
original_coeffs_sin = np.random.randn(n_terms_original)  # Random coefficients for sine terms
original_coeffs_cos = np.random.randn(n_terms_original)  # Random coefficients for cosine terms

# Define the original random function
def random_function(x, coeffs_cos, coeffs_sin, n_terms):
    result = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        result += coeffs_cos[n - 1] * np.cos(n * x) + coeffs_sin[n - 1] * np.sin(n * x)
    return result

# Calculate the original random function
original_function = random_function(x, original_coeffs_cos, original_coeffs_sin, n_terms_original)

# Fourier Series Approximation function
def fourier_series(x, coeffs_cos, coeffs_sin, terms):
    series_sum = np.zeros_like(x)
    for n in range(1, terms + 1):
        series_sum += coeffs_cos[n - 1] * np.cos(n * x) + coeffs_sin[n - 1] * np.sin(n * x)
    return series_sum

# Compute Fourier series approximation with fewer terms for lower accuracy
fourier_approximation = fourier_series(x, original_coeffs_cos, original_coeffs_sin, n_terms_approx)

# Plot the original function and its Fourier approximation with reduced accuracy
plt.figure(figsize=(10, 6))
plt.plot(x, original_function, label="Original Random Function", color="blue")
plt.plot(x, fourier_approximation, label=f"Fourier Series Approximation ({n_terms_approx} terms)", color="orange", linestyle="--")
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Original Random Function vs Low-Accuracy Fourier Series Approximation")
plt.grid(True)
plt.show()
