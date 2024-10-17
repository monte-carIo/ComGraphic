# ------------ Package Import ------------

import OpenGL.GL as GL
import glfw
import numpy as np
import argparse

# ------------ Library Import ------------

from libs.transform import Trackball
from itertools import cycle

# ------------ Shape Import ------------
from cube import Cube
from mesh import Mesh
from cylinder import Cylinder

def parse_args():
    parser = argparse.ArgumentParser(description="Viewer")
    parser.add_argument("--rotation", action="store_true", help="Enable rotation")
    parser.add_argument("--type", type=str)
    return parser.parse_args()

class Viewer:

    def __init__(self, width=800, height=800, rotation=False):
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

        # version hints: create GL windows with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        glfw.window_hint(glfw.DEPTH_BITS, 16)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # initialize trackball
        self.trackball = Trackball()
        self.mouse = (0, 0)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
                GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
                ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.5, 0.5, 0.5, 0.1)

        GL.glEnable(GL.GL_DEPTH_TEST)  
        GL.glDepthFunc(GL.GL_LESS)

        # initially empty list of object to draw
        self.drawables = []
        self.angle = 0.0
        self.rotation = rotation

    def run(self):
        """ Main render loop for this OpenGL windows """
        while not glfw.window_should_close(self.win):
            # clear draw buffer
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            win_size = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(win_size)
            model = np.identity(4)
            if self.rotation:
                self.angle += 0.001
                model = np.identity(4)
                rotation_matrix = np.array([
                    [np.cos(self.angle), 0, np.sin(self.angle), 0],
                    [0, 1, 0, 0],
                    [-np.sin(self.angle), 0, np.cos(self.angle), 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)

                model = np.dot(model, rotation_matrix) 

            # draw our scene objects
            for drawable in self.drawables:
                drawable.draw(projection, view, model)

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables):
        """ add objects to draw in this windows """
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)

            if key == glfw.KEY_S:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))
            
            if key == glfw.KEY_R:
                self.rotation = not self.rotation

            for drawable in self.drawables:
                if hasattr(drawable, 'key_handler'):
                    drawable.key_handler(key)

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.trackball.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.trackball.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.trackball.zoom(deltay, glfw.get_window_size(win)[1])

def main():
    args = parse_args()
    viewer = Viewer(rotation=args.rotation)
    if args.type == 'cube':
        model = Cube("./triangle/gouraud.vert", "./triangle/gouraud.frag").setup()
    # elif args.type == "cylinder":
    #     model = Cylinder("./triangle/gouraud.vert", "./triangle/gouraud.frag").setup()
    else:
        model = Mesh("./triangle/gouraud.vert", "./triangle/gouraud.frag", type=args.type).setup()
    viewer.add(model)
    viewer.run()


if __name__ == "__main__":
    glfw.init()
    main()
    glfw.terminate()
