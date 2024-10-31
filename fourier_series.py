import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# argparser = argparse.ArgumentParser()
# argparser.add_argument('--n_terms', type=int, default=100)

# args = argparser.parse_args()

def create_fourier_mesh(equation_func):
    L = 5
    # Define the grid size and function domain
    N = 50  # Number of points along each axis
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    Z = equation_func(X, Y)   # Original function z = x^2 + y^2

    # Define the number of Fourier terms to use for reconstruction
    # Domain limit for the Fourier series (assuming -L to L in both x and y)
    dx = (x[1] - x[0])  # Increment for x
    dy = (y[1] - y[0])  # Increment for y
    area = (2 * L) ** 2  # Total area of the domain

    # Precompute Fourier coefficients
    coefficients = {}
    for m in range(-50, 50 + 1):
        for n in range(-50, 50 + 1):
            # Calculate Fourier coefficient C_mn (integral approximation with scaling)
            C_mn = (1 / area) * np.sum(Z * np.exp(-2j * np.pi * (m * X + n * Y) / (2 * L))) * dx * dy
            coefficients[(m, n)] = C_mn
    return coefficients

def generate_fourier_mesh(n_terms, coefficients):
    L=5
    # Define a function for Fourier approximation based on precomputed coefficients
    def fourier_approximation(x, y, coefficients=coefficients, n_terms=n_terms):
        Z_approx = np.zeros_like(x, dtype=complex)
        for (m, n), C_mn in coefficients.items():
            if abs(m) <= n_terms and abs(n) <= n_terms:
                Z_approx += C_mn * np.exp(2j * np.pi * (m * x + n * y) / (2 * L))
        return np.real(Z_approx)

    # Calculate the Fourier series approximation
    t = 50
    x_test = np.linspace(-L/2, L/2, t)
    y_test = np.linspace(-L/2, L/2, t)
    X_test, Y_test = np.meshgrid(x_test, y_test)
    Z_approx_real = fourier_approximation(X_test, Y_test)
    
    vertices = []
    indices = []
    colors = []
    normals = []
    
    for i in range(t):
        for j in range(t):
            x = x_test[j]
            z = y_test[i]
            y = Z_approx_real[i, j]
            vertices.append([x, y, z])
            normal = np.array([2 * x, -1, 2 * z])
            normal = normal / np.linalg.norm(normal)
            normals.append(normal)
            colors.append([0.5 + 0.5 * np.sin(x), 0.5 + 0.5 * np.cos(z), 0.5])
            
    for i in range(t - 1):
        for j in range(t - 1):
            bottom_left = i * t + j
            bottom_right = bottom_left + 1
            top_left = bottom_left + t
            top_right = top_left + 1
            indices.append(bottom_left)
            indices.append(top_left)
            indices.append(bottom_right)
            indices.append(top_left)
            indices.append(top_right)
            indices.append(bottom_right)
        indices.append(top_right)
        for j in range(t - 1):
            indices.append(top_right - j - 1)
            indices.append(top_right - j - 1)
            
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    colors = np.array(colors, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    
    return vertices, indices, colors, normals

# # Plot the original and reconstructed functions in 3D
# fig = plt.figure(figsize=(14, 20))

# # Plot original function
# ax1 = fig.add_subplot(221, projection='3d')
# ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
# ax1.set_title('Original Function $z = x^2 + y^2$')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')

# # Plot Fourier approximation
# ax2 = fig.add_subplot(222, projection='3d')
# ax2.plot_surface(X_test, Y_test, Z_approx_real, cmap='viridis', edgecolor='none')
# ax2.set_title(f'Fourier Approximation with {n_terms} Terms')
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('Z')

# # Plot Fourier approximation for scatter
# ax3 = fig.add_subplot(223, projection='3d')
# ax3.scatter(X_test, Y_test, Z_approx_real, c='r', marker='o')
# ax3.set_title(f'Scatter Plot of Fourier Approximation with {n_terms} Terms')
# ax3.set_xlabel('X')
# ax3.set_ylabel('Y')
# ax3.set_zlabel('Z')


# plt.tight_layout()
# plt.show()

