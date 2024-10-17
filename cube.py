import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np

from libs.shader import *
from libs import transform as T
from libs.buffer import *
import ctypes

import numpy as np
import OpenGL.GL as GL


class Cube:
    def __init__(self, vert_shader, frag_shader):
        # Define the vertices for a cube (8 unique vertices, 6 faces)
        self.vertices = np.array([
            # Front face (z = 1)
            [-1, -1,  1],  # Bottom-left
            [ 1, -1,  1],  # Bottom-right
            [ 1,  1,  1],  # Top-right
            [-1,  1,  1],  # Top-left
            # Back face (z = -1)
            [-1, -1, -1],  # Bottom-left
            [ 1, -1, -1],  # Bottom-right
            [ 1,  1, -1],  # Top-right
            [-1,  1, -1],  # Top-left
        ], dtype=np.float32) / 2

        # Define the indices for the 6 faces (2 triangles per face)
        self.indices = np.array([
            0, 1, 3, 2,
            1, 5, 2, 6,
            5, 4, 6, 7,
            4, 0, 7, 3,
            3, 2, 7, 6,
            0, 1, 4, 5
        ], dtype=np.uint32)

        # Colors for each vertex
        self.colors = np.array([
            [1.0, 0.0, 0.0],  # Red (front-bottom-left)
            [0.0, 1.0, 0.0],  # Green (front-bottom-right)
            [0.0, 0.0, 1.0],  # Blue (front-top-right)
            [1.0, 1.0, 0.0],  # Yellow (front-top-left)
            [1.0, 0.0, 1.0],  # Magenta (back-bottom-left)
            [0.0, 1.0, 1.0],  # Cyan (back-bottom-right)
            [1.0, 1.0, 1.0],  # White (back-top-right)
            [0.5, 0.5, 0.5],  # Gray (back-top-left)
        ], dtype=np.float32)

        # Normals (pointing out from each face)
        self.normals = np.array([
            [0, 0, 1],  # Front face
            [0, 0, -1],  # Back face
            [-1, 0, 0],  # Left face
            [1, 0, 0],  # Right face
            [0, 1, 0],  # Top face
            [0, -1, 0],  # Bottom face
        ], dtype=np.float32)

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
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                        self.indices.shape[0], GL.GL_UNSIGNED_INT, None)