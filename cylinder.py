import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np

from libs.shader import *
from libs import transform as T
from libs.buffer import *
import ctypes

class Cylinder:
    def __init__(self, vert_shader, frag_shader, radius=1.0, height=2.0, segments=32):
        """
        Initialize a cylinder with a given radius, height, and number of segments.
        
        Args:
            vert_shader (str): Path to the vertex shader.
            frag_shader (str): Path to the fragment shader.
            radius (float): Radius of the cylinder.
            height (float): Height of the cylinder.
            segments (int): Number of segments around the circumference.
        """
        self.radius = radius
        self.height = height
        self.segments = segments
        self.vertices, self.indices_top, self.indices_bottom, self.indices_sides = self.generate_cylinder()

        # Set up the Vertex Array Object (VAO)
        self.vao = VAO()

        # Set up shaders
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    def generate_circle(self, y, is_top=True):
        """
        Generates the vertices for a circle (cap of the cylinder).
        
        Args:
            y (float): The height at which the circle is placed (either top or bottom).
            is_top (bool): Whether this is the top circle (used for normal direction).
            
        Returns:
            vertices (list): Vertices of the circle including the center.
            indices (list): Indices for the triangles forming the cap.
        """
        vertices = []
        indices = []

        # Center point of the circle (top or bottom cap)
        center = [0, y, 0]  # x, y, z
        vertices.append(center)

        # Angle between each segment (in radians)
        angle_step = 2 * np.pi / self.segments

        # Generate vertices around the circumference
        for i in range(self.segments):
            angle = i * angle_step
            x = self.radius * np.cos(angle)
            z = self.radius * np.sin(angle)
            vertices.append([x, y, z])

        # Generate indices for the cap
        for i in range(1, self.segments + 1):
            next_idx = i + 1 if i < self.segments else 1
            if is_top:
                indices.append([0, i, next_idx])  # Counter-clockwise for top cap
            else:
                indices.append([0, next_idx, i])  # Clockwise for bottom cap

        return vertices, indices

    def generate_cylinder(self):
        """
        Generates the vertices and indices for a cylinder.
        
        Returns:
            vertices (np.array): List of vertices (x, y, z) for the cylinder.
            indices_top (np.array): List of indices for the top cap.
            indices_bottom (np.array): List of indices for the bottom cap.
            indices_sides (np.array): List of indices for the side surface.
        """
        vertices = []
        indices_top = []
        indices_bottom = []
        indices_sides = []

        top_y = self.height / 2
        bottom_y = -self.height / 2

        # Generate top circle (cap)
        top_vertices, top_indices = self.generate_circle(top_y, is_top=True)
        top_offset = len(vertices)  # Offset for the top vertices
        vertices.extend(top_vertices)
        top_indices = [[idx + top_offset for idx in tri] for tri in top_indices]
        indices_top.extend(top_indices)

        # Generate bottom circle (cap)
        bottom_vertices, bottom_indices = self.generate_circle(bottom_y, is_top=False)
        bottom_offset = len(vertices)  # Offset for the bottom vertices
        vertices.extend(bottom_vertices)
        bottom_indices = [[idx + bottom_offset for idx in tri] for tri in bottom_indices]
        indices_bottom.extend(bottom_indices)

        # Generate the side surface (triangle strip)
        for i in range(self.segments):
            next_idx = (i + 1) % self.segments

            top_curr = top_offset + i + 1
            top_next = top_offset + next_idx + 1

            bottom_curr = bottom_offset + i + 1
            bottom_next = bottom_offset + next_idx + 1

            # Add indices for the side using triangle strip pattern
            indices_sides.append([top_curr, bottom_curr])
            indices_sides.append([top_next, bottom_next])

        return np.array(vertices, dtype=np.float32), np.array(indices_top), np.array(indices_bottom), np.array(indices_sides)

    def setup(self):
        """ Set up the VAO, VBOs and shaders """
        self.vao = VAO()

        # Add VBOs for vertices
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        # Add index buffer object for drawing the cylinder
        self.vao.add_ebo(self.indices_sides)

        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        modelview = np.dot(view, model)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_matrix4fv(view, 'normalMat', True)

        self.vao.activate()

        # Draw top cap using GL_TRIANGLE_FAN
        GL.glDrawElements(GL.GL_TRIANGLE_FAN, len(self.indices_top) * 3, GL.GL_UNSIGNED_INT, self.indices_top)

        # Draw bottom cap using GL_TRIANGLE_FAN
        GL.glDrawElements(GL.GL_TRIANGLE_FAN, len(self.indices_bottom) * 3, GL.GL_UNSIGNED_INT, self.indices_bottom)

        # Draw sides using GL_TRIANGLE_STRIP
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, len(self.indices_sides) * 2, GL.GL_UNSIGNED_INT, None)

        self.vao.deactivate()
