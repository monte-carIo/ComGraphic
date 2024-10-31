import sys
import os
sys.path.append(os.getcwd())
import OpenGL.GL as GL
import glfw
import numpy as np
from libs.transform import Trackball, ortho
from itertools import cycle
import imgui
from imgui.integrations.glfw import GlfwRenderer
from cube import Cube
from sphere import Sphere
# from icosphere import Icosphere
# from cubesphere import Cubesphere
# from pyramid import Pyramid
from cylinder import Cylinder
# from ray import Ray
from mesh import Mesh
import math
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Viewer")
    parser.add_argument("--rotation", action="store_true", help="Enable rotation")
    parser.add_argument("--type", type=str)
    return parser.parse_args()

value_ratio = 0.65
space_between = 10
label_text_space = 5

#Sphere
default_radius = 1.0
default_sector = 36
default_stack = 18

#IcoSphere
default_sub_div = 5

#CubeSphere
default_smooth = False

#Cube
default_size = 0.5

#Pyramid
default_base_radius = math.sqrt(3) / 6
default_height = math.sqrt(6) / 6
default_base_sector = 3

#Cylinder
default_top_radius = default_base_radius

#Ray
default_start_point = [0,0,0]
default_end_point = [0,0.5,0]

#Mesh
default_function = ("sin(0.5*x)*cos(0.5*y) + 0.01 / (1+x**2+y**2)")
default_x_range = [-0.5,0.5]
default_y_range = [-0.5,0.5]
default_resolution = 50

class Viewer:

    def __init__(self, width=900, height=700, rotation = False):
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])
        
        #rotation
        self.rotation = rotation
        self.angle = None
        self.model_state = np.identity(4, dtype=np.float32)

        
        # version hints: create GL windows with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.DEPTH_BITS, 16)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)
        
        # initialize ImGui context and renderer for GLFW
        imgui.create_context()
        self.impl = GlfwRenderer(self.win)
        
        # initialize trackball
        self.trackball = Trackball()
        self.mouse = (0, 0)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)
        # glfw.set_mouse_button_callback(self.win,self.on_mouse_click)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
                GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
                ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.5, 0.5, 0.5, 1)

        # OpenGL settings
        GL.glEnable(GL.GL_DEPTH_TEST)  
        GL.glDepthFunc(GL.GL_LESS)

        # Initialize UI state variables
        self.model_dict = {}
        
        self.dropdown_items = None
        self.selected_item = None  # Index for dropdown
        
        self.smooth = None
        
        self.change_size = None
        self.update = None
        self.reset_state = None
        
        # initially empty list of object to draw
        self.drawables = []
    
    def reset_model(self):
        model = self.model_dict[self.dropdown_items[self.selected_item]]
        if self.dropdown_items[self.selected_item] in ['Ray']:
            for arg_name in model:
                model[arg_name][0] = model[arg_name][-1]
            argument = {key: (value[0]) for key, value in model.items()}
        else:
            for arg_name in model:
                model[arg_name][1] = model[arg_name][-1]
            argument = {key: (value[1] if key not in ['smooth','x_range','y_range','function'] else value[0]) for key, value in model.items()}
        self.drawables[self.selected_item].update(argument) 
    def run(self):
        """ Main render loop for this OpenGL windows """
        self.dropdown_items = list(self.model_dict.keys())
        self.selected_item = 0  # Index for dropdown
        while not glfw.window_should_close(self.win):
            self.args = {}
            self.update = False
            self.reset_state = False
            self.angle = 0.0005 * 4
            
            
            # Poll for and process events
            glfw.poll_events()
            
            # Process ImGui inputs
            self.impl.process_inputs()
            
            # Start a new frame for ImGui
            imgui.new_frame()
        
            # Render the UI window
            self.render_ui()
            
            if self.update:
                current_model = self.model_dict[self.dropdown_items[self.selected_item]]
                self.args = {key: (value[1]) for key, value in current_model.items() if len(value) > 1}
                self.drawables[self.selected_item].update(self.args)
            
            # clear draw buffer
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            
            win_size = glfw.get_window_size(self.win)
            if self.reset_state:
                self.trackball.reset_trackball()
                self.model_state = np.identity(4, 'f')
                self.rotation = False
                self.reset_model()
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(win_size)
            # model = np.identity(4)
            if self.rotation:
                # model = np.identity(4)
                rotation_matrix = np.array([
                    [np.cos(self.angle), 0, np.sin(self.angle), 0],
                    [0, 1, 0, 0],
                    [-np.sin(self.angle), 0, np.cos(self.angle), 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)

                self.model_state = np.dot(rotation_matrix, self.model_state)
            
            # draw our scene objects
            # for drawable in self.drawables:
            #     drawable.draw(projection, view, self.model_state)
            
            self.drawables[self.selected_item].draw(projection, view, self.model_state)
            
            imgui.render()
            self.impl.render(imgui.get_draw_data())
            
            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)
        
        self.impl.shutdown()
        
    def render_ui(self):
        # Get the number of parameters for the selected model
        num_params = len(list(self.model_dict[self.dropdown_items[self.selected_item]].keys()))
        # print(num_params)
        # Calculate window height dynamically (you can adjust these values for better layout)
        base_height = 90  # Base height for the window header
        param_height = 36  # Approximate height per slider
        dynamic_height = base_height + (num_params * param_height)

        if (self.change_size):
            self.reset_state = True
            imgui.set_next_window_size(280, dynamic_height)
        
        # Start a new ImGui window for the UI
        imgui.begin("Settings", closable=False)
        
        imgui.text("Configure the viewer window")
        
        imgui.push_item_width(imgui.get_window_width() * value_ratio)
        
        imgui.spacing()
        imgui.spacing()
        imgui.separator()
        
        self.change_size, self.selected_item = imgui.combo("##Models", self.selected_item, self.dropdown_items, height_in_items = 4)
        imgui.same_line(spacing = 10)
        imgui.text("Models")
        
        
        
        for item in self.model_dict[self.dropdown_items[self.selected_item]]:
            tmp = self.model_dict[self.dropdown_items[self.selected_item]][item]
            
            imgui.spacing()  # Adds vertical spacing
            imgui.spacing()
            imgui.spacing()
            
            temp = tmp[1]
            
            changed, tmp[1] = imgui.slider_int(f"##{item}",tmp[1], min_value = tmp[0], max_value = tmp[-2], format='%d')
            
            # self.args.append(tmp[1])
            if str(item).endswith("point") or str(item).endswith("range") or item in ['function']:
                if not self.update and temp != tmp[0]:
                    self.update = True
            elif item not in ['smooth']:
                if not self.update and temp != tmp[1]:
                    self.update = True
            else:
                if not self.update and temp != tmp[0]:
                    self.update = True

            imgui.same_line(spacing = 10)
            imgui.text(f'{item}')
        
        
        # End the ImGui window
        imgui.end()
        
    
    def add(self, *drawables):
        """ add objects to draw in this windows """
        for model in drawables:
            if isinstance(model, Sphere):
                self.model_dict['Sphere'] = {
                    'radius': [0.05,default_radius,5.0,default_radius],
                    'sector': [2,default_sector,50,default_sector], 
                    'stack': [2,default_stack,50,default_stack]
                }
            elif isinstance(model, Cube):
                self.model_dict['Cube'] = {
                    'size': [0.1,default_size,1.0,default_size]
                }
            elif isinstance(model,Cylinder):
                self.model_dict['Cylinder'] = {
                    'base_radius': [0.05,default_base_radius,5.0,default_base_radius],
                    'top_radius': [0.05,default_top_radius,5.0,default_top_radius],
                    'height': [0.05,default_height,5.0,default_height],
                    'sector': [2,default_base_sector,32,default_base_sector]
                }
            elif isinstance(model,Mesh):
                self.model_dict['Mesh'] = {
                    'n_terms': [1,10,100,10],
                }
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        if imgui.get_io().want_capture_keyboard:
            return  # Ignore GLFW keyboard input if ImGui is handling it
        
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)

            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))
            
            if key == glfw.KEY_R:
                self.rotation = not self.rotation
            
            for drawable in self.drawables:
                if hasattr(drawable, 'key_handler'):
                    drawable.key_handler(key)
    
    def on_mouse_move(self, win, xpos, ypos):
        if imgui.get_io().want_capture_mouse:
            return
        
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
    # elif args.type == "triangle":
    #     model = Triangle("./triangle/gouraud.vert", "./triangle/gouraud.frag").setup()
    elif args.type == "cylinder":
        model = Cylinder("./triangle/gouraud.vert", "./triangle/gouraud.frag").setup()
    else:
        model = Mesh("./triangle/gouraud.vert", "./triangle/gouraud.frag", type=args.type).setup()
    viewer.add(model)
    viewer.run()


if __name__ == "__main__":
    glfw.init()
    main()
    glfw.terminate()
