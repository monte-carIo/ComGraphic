import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import c_module.fourier_module as fourier_module

L = 5
t = 50
x_test = np.linspace(-L/2, L/2, t)
y_test = np.linspace(-L/2, L/2, t)
X_test, Y_test = np.meshgrid(x_test, y_test)

def fourier_approximation(coefficients, n_terms):
    # Z_approx = np.zeros_like(x, dtype=complex)
    Z_list = [0] * n_terms
    for i, items in enumerate(coefficients.items()):
        (m, n), C_mn = items
        # if abs(m) <= n_terms and abs(n) <= n_terms:
        #     Z_approx += C_mn * np.exp(2j * np.pi * (m * x + n * y) / (2 * L))
        Z_tmp = C_mn * np.exp(2j * np.pi * (m * X_test + n * Y_test) / (2 * L))
        for j in range(n_terms):
            if abs(m) <= j and abs(n) <= j:
                Z_list[j] += Z_tmp
    return [np.real(Z) for Z in Z_list]

def create_fourier_mesh(equation_func):
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
    return fourier_module.calculate_coefficients(Z, X, Y, dx, dy, area, L)

def pre_cofficients(Z, X, Y, dx, dy, area):
    # Precompute Fourier coefficients
    coefficients = {}
    for m in range(-100, 100 + 1):
        for n in range(-100, 100 + 1):
            # Calculate Fourier coefficient C_mn (integral approximation with scaling)
            C_mn = (1 / area) * np.sum(Z * np.exp(-2j * np.pi * (m * X + n * Y) / (2 * L))) * dx * dy
            coefficients[(m, n)] = C_mn
    return coefficients

def generate_fourier_mesh(Z_approx_real):
    # Calculate the Fourier series approximation   
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


