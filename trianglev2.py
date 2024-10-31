import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np

from libs.shader import *
from libs import transform as T
from libs.buffer import *
import ctypes

import numpy as np
import OpenGL.GL as GL


class Triangle:
    def __init__(self, vert_shader, frag_shader):
        
        self.vertices = np.array([
            [-1, -1, 1],  # Point 1
            [1, -1, 1],  # Point 2
            [0.5, 1.0, 1],  # Point 3
            [0, 0, -1]   # Point 4
        ], dtype=np.float32) / 2

        # Define the indices for the 6 faces (2 triangles per face)
        self.indices = np.array([
            0, 1, 2, 3, 1, 0, 3, 2
        ], dtype=np.uint32)

        # Colors for each vertex
        self.colors = np.array([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0]   # Yellow
        ], dtype=np.float32)

        # Normals (pointing out from each face)
        self.normals = np.array([
            [0, 0, 1],  # Front face
            [0, 0, 1],  # Front face
            [0, 0, 1],  # Front face
            [0, 0, 1]   # Front face
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