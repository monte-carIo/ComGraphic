import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np

from libs.shader import *
from libs import transform as T
from libs.buffer import *
import ctypes

import numpy as np
import OpenGL.GL as GL


class Cylinder:
    def __init__(self, vert_shader, frag_shader):
        
        def create_circle(radius, z, num_segments):
            theta = 2 * np.pi / num_segments
            circle_vertices = []
            for i in range(num_segments):
                x = radius * np.cos(i * theta)
                y = radius * np.sin(i * theta)
                circle_vertices.append([x, y, z])
            return circle_vertices

        num_segments = 36
        self.num_segments = num_segments
        radius = 1.0
        height = 2.0

        top_circle = create_circle(radius, height / 2, num_segments)
        bottom_circle = create_circle(radius, -height / 2, num_segments)

        self.vertices = np.array(
            [[0, 0, height / 2]] + top_circle + [[0, 0, - height / 2]] + bottom_circle,
            dtype=np.float32
        ) / 2

        self.indices = []

        # Top circle (triangle fan)
        self.indices.append(0)  # Center of the top circle
        for i in range(1, num_segments + 1):
            self.indices.append(i)
        self.indices.append(1)  # Closing the fan

        # Bottom circle (triangle fan)
        self.indices.append(num_segments + 1)  # Center of the bottom circle
        for i in range(num_segments + 2, 2 * num_segments + 2):
            self.indices.append(i)
        self.indices.append(num_segments + 2)  # Closing the fan

        # Side surface (triangle strip)
        for i in range(1, num_segments + 1):
            self.indices.append(i)
            self.indices.append(i + num_segments + 1)
        self.indices.append(1)  # Closing the strip
        self.indices.append(num_segments + 2)

        self.indices = np.array(self.indices, dtype=np.uint32)

        # Colors for each vertex
        self.colors = np.array(
            [[1.0, 0.0, 0.0]] * (num_segments + 1) +  # Red for top circle
            [[0.0, 1.0, 0.0]] * (num_segments + 1),   # Green for bottom circle
            dtype=np.float32
        )

        # Normals (pointing out from each face)
        self.normals = np.array(
            [[0, 0, 1]] * (num_segments + 1) +  # Normals for top circle
            [[0, 0, -1]] * (num_segments + 1),  # Normals for bottom circle
            dtype=np.float32
        )

        # Set up the Vertex Array Object (VAO)
        self.vao = VAO()

        # Set up shaders
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    def setup(self):
        """ Set up the VAO, VBOs and shaders """
        # Add VBOs for vertices, colors, and normals
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        # Add index buffer object for drawing the cube with glDrawElements
        self.vao.add_ebo(self.indices)

        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        modelview = np.dot(view, model)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_matrix4fv(view, 'normalMat', True)

        self.vao.activate()
        # Draw the top circle using a triangle fan
        GL.glDrawElements(GL.GL_TRIANGLE_FAN, self.num_segments + 2, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))

        # Draw the bottom circle using a triangle fan
        offset = (self.num_segments + 2) * ctypes.sizeof(ctypes.c_uint)
        GL.glDrawElements(GL.GL_TRIANGLE_FAN, self.num_segments + 2, GL.GL_UNSIGNED_INT, ctypes.c_void_p(offset))

        # Draw the side surface using a triangle strip
        offset = 2 * (self.num_segments + 2) * ctypes.sizeof(ctypes.c_uint)
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, 2 * (self.num_segments + 1) + 2, GL.GL_UNSIGNED_INT, ctypes.c_void_p(offset))