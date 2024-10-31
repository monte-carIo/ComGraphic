import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np

from libs.shader import *
from libs import transform as T
from libs.buffer import *
import ctypes

import numpy as np
import OpenGL.GL as GL


class Sphere:
    def __init__(self, vert_shader, frag_shader):
        
        def create_sphere(radius, subdivisions):
            def add_vertex(v):
                length = np.linalg.norm(v)
                return (v / length * radius).tolist()

            def midpoint(v1, v2):
                return add_vertex((np.array(v1) + np.array(v2)) / 2)

            t = (1.0 + np.sqrt(5.0)) / 2.0

            # Create 12 vertices of a icosahedron
            vertices = [
            add_vertex([-1,  t,  0]),
            add_vertex([ 1,  t,  0]),
            add_vertex([-1, -t,  0]),
            add_vertex([ 1, -t,  0]),
            add_vertex([ 0, -1,  t]),
            add_vertex([ 0,  1,  t]),
            add_vertex([ 0, -1, -t]),
            add_vertex([ 0,  1, -t]),
            add_vertex([ t,  0, -1]),
            add_vertex([ t,  0,  1]),
            add_vertex([-t,  0, -1]),
            add_vertex([-t,  0,  1]),
            ]

            # Create 20 triangles of the icosahedron
            faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
            ]

            # Refine triangles
            for _ in range(subdivisions):
                faces_subdiv = []
                midpoints = {}
                for tri in faces:
                    v1, v2, v3 = tri
                    a = tuple(midpoint(vertices[v1], vertices[v2]))
                    b = tuple(midpoint(vertices[v2], vertices[v3]))
                    c = tuple(midpoint(vertices[v3], vertices[v1]))

                    if a not in midpoints:
                        midpoints[a] = len(vertices)
                        vertices.append(a)
                    if b not in midpoints:
                        midpoints[b] = len(vertices)
                        vertices.append(b)
                    if c not in midpoints:
                        midpoints[c] = len(vertices)
                        vertices.append(c)

                    faces_subdiv.append([v1, midpoints[a], midpoints[c]])
                    faces_subdiv.append([v2, midpoints[b], midpoints[a]])
                    faces_subdiv.append([v3, midpoints[c], midpoints[b]])
                    faces_subdiv.append([midpoints[a], midpoints[b], midpoints[c]])

            faces = faces_subdiv

            return vertices, faces

        radius = 1.0
        subdivisions = 3
        self.vertices, faces = create_sphere(radius, subdivisions)
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.indices = []
        for face in faces:
            self.indices.extend(face)

        # Colors for each vertex
        self.colors = np.array(
            [[1.0, 0.0, 0.0]] * len(self.vertices),  # Red for all vertices
            dtype=np.float32
        )

        # Normals (pointing out from each face)
        self.normals = np.array(
            self.vertices,  # Normals are the same as the vertices for a sphere
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
        # Draw the sphere using triangles
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))
