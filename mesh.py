import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np

from libs.shader import *
from libs import transform as T
from libs.buffer import *
import ctypes
from fourier_series import create_fourier_mesh, generate_fourier_mesh, fourier_approximation

import numpy as np

def mountain_function(x, z):
    """
    Default mountain-like function: y = exp(-(x^2 + z^2)/a) * sin(bx) * sin(cz)
    Creates a mountain-like surface with peaks and valleys.
    """
    a = 5.0  # Controls the width of the mountain
    b = 2.0  # Controls the frequency of sine wave along the x-axis
    c = 2.0  # Controls the frequency of sine wave along the z-axis

    # Exponential decay (mountain peak at the center)
    height = np.exp(-(x**2 + z**2) / a)

    # Adding sine wave patterns for complexity
    wave_pattern = np.sin(b * x) * np.sin(c * z)

    # Combine the height and wave pattern
    y = height * wave_pattern

    return y

def get_user_input_equation():
    """
    Prompts the user to input a mathematical equation for the surface.
    The equation should be in terms of variables 'x' and 'z'.
    
    Returns:
        equation_func (function): A function that calculates y from x and z.
    """
    # Prompt the user for input
    equation_str = input("Enter the equation for y in terms of x and z (e.g., x**2 + y**2 - 1.5): ")
    
    if equation_str.strip() == "":
        print("Using default mountain equation.")
        return mountain_function

    # Define a safe environment for eval to avoid security risks
    safe_globals = {
        "__builtins__": None,  # Disable built-in functions for safety
        "np": np,  # Allow numpy functions for advanced math operations
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "x": 0,  # Default values, to be overridden during eval
        "y": 0,
    }

    # Define the equation as a function using eval
    def equation_func(x, z):
        # Update x and z in the safe globals
        safe_globals["x"] = x
        safe_globals["y"] = z

        # Evaluate the equation and return the result
        return eval(equation_str, safe_globals)

    return equation_func

def generate_sphere(radius=1.0, lat_segments=20, lon_segments=20):
    """Generates a sphere mesh with vertices, indices, normals, and colors."""

    vertices = []
    indices = []
    colors = []
    normals = []

    # Latitude (phi) and longitude (theta) generation
    for i in range(lat_segments + 1):
        lat = np.pi * i / lat_segments  # from 0 to pi (latitude angle)
        for j in range(lon_segments + 1):
            lon = 2 * np.pi * j / lon_segments  # from 0 to 2pi (longitude angle)

            # Parametric equations for the sphere
            x = radius * np.sin(lat) * np.cos(lon)
            y = radius * np.cos(lat)
            z = radius * np.sin(lat) * np.sin(lon)

            # Append the vertex
            vertices.append([x, y, z])

            # Compute the normal (same as the vertex for a sphere centered at the origin)
            normals.append([x / radius, y / radius, z / radius])  # Unit normals

            # Assign a color (for now, use a simple gradient based on the y-coordinate)
            colors.append([1,1,0] if i%2==0 else [0,1,1])

    # Generate indices (two triangles per quad)
    for i in range(lat_segments):
        for j in range(lon_segments):
            # Get the indices of the four corners of the quad
            bottom_left = i * (lon_segments + 1) + j
            bottom_right = bottom_left + 1
            top_left = bottom_left + (lon_segments + 1)
            top_right = top_left + 1

            # First triangle of the quad
            indices.append(bottom_left)
            indices.append(bottom_right)
            indices.append(top_right)

            # Second triangle of the quad
            indices.append(bottom_left)
            indices.append(top_right)
            indices.append(top_left)

    # Convert lists to numpy arrays for efficient OpenGL processing
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    colors = np.array(colors, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)

    return vertices, indices, colors, normals


def generate_grid_mesh(equation_func, grid_size=10, spacing=0.3):
    """Generates a grid of vertices, indices, and normals for a paraboloid surface."""
    vertices = []
    indices = []
    colors = []
    normals = []

    # Generate vertices and normals for a paraboloid surface
    for i in range(int(-grid_size/2), int(grid_size/2 + 1)):
        for j in range(int(-grid_size/2), int(grid_size/2 + 1)):
            x = j * spacing
            z = i * spacing
            # y = (x ** 2 + z ** 2) - (grid_size * spacing)/2 # Paraboloid equation: y = x^2 + z^2
            y = equation_func(x, z)
            # Normalize the vertices to stay within a reasonable range
            vertices.append([x / (grid_size * spacing),
                            y / (grid_size * spacing),
                            z / (grid_size * spacing)])

            # Calculate the normal vector for a paraboloid surface (approximation)
            # The normal is the gradient of the surface, which is [2x, -1, 2z] (from the paraboloid equation)
            normal = np.array([2 * x, -1, 2 * z])
            normal = normal / np.linalg.norm(normal)  # Normalize the normal vector
            normals.append(normal)

            # Assign a color (gradient color based on the y value)
            colors.append([0.5 + 0.5 * np.sin(x), 0.5 + 0.5 * np.cos(z), 0.5])

    # Generate indices for the grid, where each quad is divided into two triangles
    for i in range(grid_size):
        for j in range(grid_size):
            # Indices of the four corners of a quad
            bottom_left = i * (grid_size + 1) + j
            bottom_right = bottom_left + 1
            top_left = bottom_left + (grid_size + 1)
            top_right = top_left + 1
            # print(bottom_left, bottom_right, top_left, top_right)

            # Two triangles for the quad
            # Ensure proper winding order: counterclockwise for OpenGL's default front face
            indices.append(bottom_left)
            indices.append(top_left)
            indices.append(bottom_right)
            
            indices.append(top_left)
            indices.append(top_right)
            indices.append(bottom_right)
        indices.append(top_right)
        for j in range(grid_size):
            indices.append(top_right - j - 1)
            indices.append(top_right - j - 1)
            

    # Convert lists to numpy arrays for efficient OpenGL processing
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    colors = np.array(colors, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)

    return vertices, indices, colors, normals



class Mesh:
    def __init__(self, vert_shader, frag_shader, type):
        """
        Mesh class that takes a function as input to generate the geometry.
        The function should return vertices, indices, (optionally) colors, and normals.
        
        Args:
            generate_func (function): A function that generates the vertices, indices,
                                    colors, and normals for the mesh.
        """
        if type == 'sphere':
            self.vertices, self.indices, self.colors, self.normals = generate_sphere()
        elif type == 'mesh':
            equation_func = get_user_input_equation()
            self.vertices, self.indices, self.colors, self.normals = generate_grid_mesh(equation_func)
        elif type == 'fourier':
            equation_func = get_user_input_equation()
            self.equation_func = equation_func
            self.coeffs = create_fourier_mesh(equation_func)
            self.Z_list = fourier_approximation(self.coeffs, 100)
            self.vertices, self.indices, self.colors, self.normals = generate_fourier_mesh(self.Z_list[10])
        else:
            raise ValueError(f"Invalid mesh type: {type}")

        # Set up the Vertex Array Object (VAO)
        self.vao = VAO()

        # Set up shaders
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    def setup(self):
        """ Set up the VAO, VBOs, and shaders """
        # Add VBOs for vertices, colors, and normals
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        if self.colors is not None:
            self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        if self.normals is not None:
            self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        # Add index buffer object for drawing the mesh with glDrawElements
        self.vao.add_ebo(self.indices)

        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        modelview = np.dot(view, model)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_matrix4fv(view, 'normalMat', True)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                        self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
    def update(self,arg):
        new_n_terms = arg['n_terms']
        if new_n_terms < 0:
            new_n_terms = 0
        if new_n_terms >= len(self.Z_list):
            new_n_terms = len(self.Z_list) - 1
        
        del self.vao
        self.vao = VAO()
        
        self.vertices, self.indices, self.colors, self.normals = generate_fourier_mesh(self.Z_list[new_n_terms])
        
        self.setup()

