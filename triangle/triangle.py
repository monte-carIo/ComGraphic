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
        ], dtype=np.float32)

        # Define the indices for the 6 faces (2 triangles per face)
        self.indices = np.array([
            # Front face
            0, 1, 3, 2,
            # Right face
            1, 5, 2, 6,
            # Back face
            5, 4, 6, 7,
            # Left face
            4, 0, 7, 3,
            # Top face
            3, 2, 7, 6,
            # Bottom face
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



class Cylinder:
    def __init__(self, vert_shader, frag_shader, num_segments=32):
        # Create vertices for a cylinder
        self.num_segments = num_segments
        angle_step = 2 * np.pi / num_segments

        self.vertices = []
        for i in range(num_segments + 1):
            angle = i * angle_step
            x = np.cos(angle)
            y = np.sin(angle)
            self.vertices.append([x, y, 1])  # Top circle (z = 1)
            self.vertices.append([x, y, -1])  # Bottom circle (z = -1)

        self.vertices = np.array(self.vertices, dtype=np.float32)

        # Normals (pointing outward for each vertex)
        self.normals = []
        for i in range(num_segments + 1):
            angle = i * angle_step
            x = np.cos(angle)
            y = np.sin(angle)
            self.normals.append([x, y, 0])
            self.normals.append([x, y, 0])
        
        self.normals = np.array(self.normals, dtype=np.float32)

        # Colors for each vertex (could vary or be constant)
        self.colors = np.array([[0.0, 1.0, 0.0]] * len(self.vertices), dtype=np.float32)  # Green

        # Set up VAO
        self.vao = VAO()

        # Set up shaders
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    def setup(self):
        # Add VBOs for the cylinder
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        # Set up projection and modelview matrices (as in Triangle)
        GL.glUseProgram(self.shader.render_idx)
        projection = np.identity(4, dtype=np.float32)  # Placeholder (to be updated by Viewer)
        modelview = np.identity(4, dtype=np.float32)  # Placeholder (to be updated by Viewer)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        return self

    def draw(self, projection, view, model):
        self.vao.activate()
        GL.glUseProgram(self.shader.render_idx)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(view, 'view', True)
        self.uma.upload_uniform_matrix4fv(model, 'modelview', True)

        # Draw the cylinder as a triangle strip
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, len(self.vertices))


class Triangle:
    def __init__(self, vert_shader, frag_shader):
        self.vertices = np.array([
            [-1, -1, 0],
            [+1, -1, 0],
            [ 0, +1, 0]
        ], dtype=np.float32)
        normals = np.random.normal(0, 3, (3, 3)).astype(np.float32)
        normals[:, 2] = np.abs(normals[:, 2])
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        self.colors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        GL.glUseProgram(self.shader.render_idx)
        projection = T.ortho(-1, 1, -1, 1, -1, 1)
        modelview = np.identity(4, 'f')
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        # Light
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6]  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        # Materials
        K_materials = np.array([
            [0.6, 0.4, 0.7],  # diffuse
            [0.6, 0.4, 0.7],  # specular
            [0.6, 0.4, 0.7]  # ambient
        ], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)

        shininess = 100.0
        mode = 1

        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')
        return self

    def draw(self, projection, view, model):
        self.vao.activate()
        GL.glUseProgram(self.shader.render_idx)
        
        # Upload the projection, view, and model matrices to the shader
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(view, 'view', True)
        self.uma.upload_uniform_matrix4fv(model, 'modelview', True)
        
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 3)


class TriangleEx:
    def __init__(self, vert_shader, frag_shader):
        """
        self.vertex_attrib:
        each row: v.x, v.y, v.z, c.r, c.g, c.b, n.x, n.y, n.z
                  v.x, v.y, v.z, c.r, c.g, c.b, n.x, n.y, n.z
        =>  (a) stride = nbytes(v0.x -> v1.x) = 9*4 = 36
            (b) offset(vertex) = ctypes.c_void_p(0); can use "None"
                offset(color) = ctypes.c_void_p(3*4)
                offset(normal) = ctypes.c_void_p(6*4)
        """
        vertex_color = np.array([
            [-1, -1, 0, 1.0, 0.0, 0.0],
            [+1, -1, 0, 0.0, 1.0, 0.0],
            [ 0, +1, 0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        normals = np.random.normal(0, 3, (3, 3)).astype(np.float32)
        normals[:, 2] = np.abs(normals[:, 2])
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        self.vertex_attrib = np.concatenate([vertex_color, normals], axis=1)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #

    def setup(self):
        stride = 9*4
        offset_v = ctypes.c_void_p(0)  # None
        offset_c = ctypes.c_void_p(3*4)
        offset_n = ctypes.c_void_p(6*4)
        self.vao.add_vbo(0, self.vertex_attrib, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=stride, offset=offset_v)
        self.vao.add_vbo(1, self.vertex_attrib, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=stride, offset=offset_c)
        self.vao.add_vbo(2, self.vertex_attrib, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=stride, offset=offset_n)

        GL.glUseProgram(self.shader.render_idx)
        normalMat = np.identity(4, 'f')
        projection = T.ortho(-1, 1, -1, 1, -1, 1)
        modelview = np.identity(4, 'f')
        self.uma.upload_uniform_matrix4fv(normalMat, 'normalMat', True)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        # Light
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6]  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        # Materials
        K_materials = np.array([
            [0.6, 0.4, 0.7],  # diffuse
            [0.6, 0.4, 0.7],  # specular
            [0.6, 0.4, 0.7]  # ambient
        ], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)

        shininess = 100.0
        mode = 1

        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')
        return self

    def draw(self, projection, view, model):
        self.vao.activate()
        GL.glUseProgram(self.shader.render_idx)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)


class AMesh:
    def __init__(self, vert_shader, frag_shader):
        self.vertices = np.array([
            [-1,     0,   0], #T1
            [-0.5,   1,   0],
            [ 0,     0,   0],
            [ 0,     0,   0], #T2
            [ 0.5,   1,   0],
            [ 1,     0,   0],
            [-0.5,  -1,   0], #T3
            [ 0,     0,   0],
            [ 0.5,  -1,   0],
        ], dtype=np.float32)
        normals = np.random.normal(0, 3, (3, 3)).astype(np.float32)
        normals[:, 2] = np.abs(normals[:, 2])
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        self.colors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        GL.glUseProgram(self.shader.render_idx)
        projection = T.ortho(-1, 1, -1, 1, -1, 1)
        modelview = np.identity(4, 'f')
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        # Light
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6]  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        # Materials
        K_materials = np.array([
            [0.6, 0.4, 0.7],  # diffuse
            [0.6, 0.4, 0.7],  # specular
            [0.6, 0.4, 0.7]  # ambient
        ], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)

        shininess = 100.0
        mode = 1

        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')
        return self

    def draw(self, projection, view, model):
        self.vao.activate()
        GL.glUseProgram(self.shader.render_idx)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 9)

class AMeshIDX:
    def __init__(self, vert_shader, frag_shader):
        self.vertices = np.array([
            [-0.5,   1,   0],
            [-1,     0,   0],
            [ 0,     0,   0],
            [ 0.5,   1,   0],
            [ 1,     0,   0],
            [-0.5,  -1,   0],
            [ 0.5,  -1,   0]
        ], dtype=np.float32)
        normals = np.random.normal(0, 3, (3, 3)).astype(np.float32)
        normals[:, 2] = np.abs(normals[:, 2])
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        self.indices = np.array(
            [1, 0, 2, 2, 3, 4, 5, 2, 6],
            dtype=np.uint32
        )
        self.colors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        #setup array of indices
        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)
        projection = T.ortho(-1, 1, -1, 1, -1, 1)
        modelview = np.identity(4, 'f')
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        # Light
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6]  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        # Materials
        K_materials = np.array([
            [0.6, 0.4, 0.7],  # diffuse
            [0.6, 0.4, 0.7],  # specular
            [0.6, 0.4, 0.7]  # ambient
        ], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)

        shininess = 100.0
        mode = 1

        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')
        return self

    def draw(self, projection, view, model):
        self.vao.activate()
        GL.glUseProgram(self.shader.render_idx)
        GL.glDrawElements(GL.GL_TRIANGLES, 9, GL.GL_UNSIGNED_INT, None)

class AMeshStrip:
    def __init__(self, vert_shader, frag_shader):
        self.vertices = np.array([
            [-1, -1, 0],
            [-1, +1, 0],
            [-0.5, -1, 0],
            [-0.5, +1, 0],
            [+0.5, -1, 0],
            [+0.5, +1, 0],
            [+1, -1, 0],
            [+1, +1, 0],
        ], dtype=np.float32)
        normals = np.random.normal(0, 3, (3, 3)).astype(np.float32)
        normals[:, 2] = np.abs(normals[:, 2])
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        self.indices = np.arange(len(self.vertices),
                                 dtype=np.uint32)
        self.colors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float32)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        #setup array of indices
        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)
        projection = T.ortho(-1, 1, -1, 1, -1, 1)
        modelview = np.identity(4, 'f')
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        # Light
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6]  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        # Materials
        K_materials = np.array([
            [0.6, 0.4, 0.7],  # diffuse
            [0.6, 0.4, 0.7],  # specular
            [0.6, 0.4, 0.7]  # ambient
        ], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)

        shininess = 100.0
        mode = 1

        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')
        return self

    def draw(self, projection, view, model):
        self.vao.activate()
        GL.glUseProgram(self.shader.render_idx)
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          len(self.indices),
                          GL.GL_UNSIGNED_INT,
                          None)

class Circle:
    def __init__(self, vert_shader, frag_shader, N=50, R=1):
        alpha = np.linspace(0, 2*np.pi, N)
        x, y, z = R*np.cos(alpha), R*np.sin(alpha), np.zeros_like(alpha)
        x, y, z = [v.reshape(-1, 1).astype(np.float32) for v in [x, y, z]]
        center = np.array([0, 0, 0], dtype=np.float32).reshape(1, -1)
        V = np.hstack([x, y, z])
        self.vertices = np.vstack([center, V])

        normals = np.random.normal(0, 3, (3, 3)).astype(np.float32)
        normals[:, 2] = np.abs(normals[:, 2])
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        self.indices = np.arange(len(self.vertices),
                                 dtype=np.uint32)

        color_cen = np.array([1.0, 0.0, 0.0], dtype=np.float32).reshape(1, -1)
        color_pts = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        num_pts = len(self.vertices) - 1
        color_pts = color_pts + np.zeros((num_pts, 1), dtype=np.float32)
        self.colors = np.vstack([color_cen, color_pts])


        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        #setup array of indices
        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)
        projection = T.ortho(-1, 1, -1, 1, -1, 1)
        modelview = np.identity(4, 'f')
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        # Light
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6]  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        # Materials
        K_materials = np.array([
            [0.6, 0.4, 0.7],  # diffuse
            [0.6, 0.4, 0.7],  # specular
            [0.6, 0.4, 0.7]  # ambient
        ], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)

        shininess = 100.0
        mode = 1

        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')
        return self

    def draw(self, projection, view, model):
        self.vao.activate()
        GL.glUseProgram(self.shader.render_idx)
        GL.glDrawElements(GL.GL_TRIANGLE_FAN,
                          len(self.indices),
                          GL.GL_UNSIGNED_INT,
                          None)
