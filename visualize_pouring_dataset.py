import numpy as np
import os, pickle
import time
import threading
from datetime import datetime

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from copy import deepcopy

from vis_utils.open3d_utils import (
    get_mesh_bottle, 
    get_mesh_mug, 
)

# color template (2023 pantone top 10 colors)
rgb = np.zeros((10, 3))
rgb[0, :] = [208, 28, 31] # fiery red
rgb[1, :] = [207, 45, 113] # beetroot purple
rgb[2, :] = [249, 77, 0] # tangelo
rgb[3, :] = [250, 154, 133] # peach pink
rgb[4, :] = [247, 208, 0] # empire yellow
rgb[5, :] = [253, 195, 198] # crystal rose
rgb[6, :] = [57, 168, 69] # classic green
rgb[7, :] = [193, 219, 60] # love bird 
rgb[8, :] = [75, 129, 191] # blue perennial 
rgb[9, :] = [161, 195, 218] # summer song
rgb = rgb / 255  
        
class AppWindow:
    def __init__(self):
        # initialize
        self.root = 'datasets/pouring_data'
        self.dir_idx = 1
        self.skip_size = 5
        self.pour_style = 'water'
        self.pour_amount = '200'
        self.thread_finished = True

        # parameters
        image_size = [1024, 768]
        
        # mesh table
        self.mesh_box = o3d.geometry.TriangleMesh.create_box(
            width=2, height=2, depth=0.03
        )
        self.mesh_box.translate([-1, -1, -0.03])
        self.mesh_box.paint_uniform_color([222/255, 184/255, 135/255])
        self.mesh_box.compute_vertex_normals()

        # bottle label
        self.draw_bottle_label = True
        self.bottle_label_angle = 0

        # frame
        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # object material
        self.mat = rendering.MaterialRecord()
        self.mat.shader = 'defaultLit'
        self.mat.base_color = [1.0, 1.0, 1.0, 0.9]

        ######################################################
        ################# STARTS FROM HERE ###################
        ######################################################

        # set window
        self.window = gui.Application.instance.create_window(
            str(datetime.now().strftime('%H%M%S')), width=image_size[0], height=image_size[1]
        )
        w = self.window
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        # camera viewpoint
        self._scene.scene.camera.look_at(
            [0, 0, 0], # camera lookat
            [0.7, 0, 0.9], # camera position
            [0, 0, 1] # fixed
        )

        # other settings
        self._scene.scene.set_lighting(self._scene.scene.LightingProfile.DARK_SHADOWS, (-0.3, 0.3, -0.9))
        self._scene.scene.set_background([1.0, 1.0, 1.0, 1.0], image=None)

        ############################################################
        ######################### MENU BAR #########################
        ############################################################
        
        # menu bar initialize
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # initialize collapsable vert
        dataset_config = gui.CollapsableVert("Dataset config", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        # Visualize type
        self._video_button = gui.Button("Video")
        self._video_button.horizontal_padding_em = 0.5
        self._video_button.vertical_padding_em = 0
        self._video_button.set_on_clicked(self._set_vis_mode_video)
        
        self._afterimage_button = gui.Button("Afterimage")
        self._afterimage_button.horizontal_padding_em = 0.5
        self._afterimage_button.vertical_padding_em = 0
        self._afterimage_button.set_on_clicked(self._set_vis_mode_afterimage)
        
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._video_button)
        h.add_child(self._afterimage_button)
        h.add_stretch()

        # add
        dataset_config.add_child(gui.Label("Visualize type"))
        dataset_config.add_child(h)

        # style
        self._water_button = gui.Button("Water")
        self._water_button.horizontal_padding_em = 0.5
        self._water_button.vertical_padding_em = 0
        self._water_button.set_on_clicked(self._set_pour_mode_water)
        
        self._wine_button = gui.Button("Wine")
        self._wine_button.horizontal_padding_em = 0.5
        self._wine_button.vertical_padding_em = 0
        self._wine_button.set_on_clicked(self._set_pour_mode_wine)
        
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._water_button)
        h.add_child(self._wine_button)
        h.add_stretch()

        # add
        dataset_config.add_fixed(separation_height)
        dataset_config.add_child(gui.Label("Pouring style"))
        dataset_config.add_child(h)

        # direction
        self._direction_silder = gui.Slider(gui.Slider.INT)
        self._direction_silder.set_limits(1, 5)
        self._direction_silder.set_on_value_changed(self._set_dir_idx)
        
        # add
        dataset_config.add_fixed(separation_height)
        dataset_config.add_child(gui.Label("Direction"))
        dataset_config.add_child(self._direction_silder)
        
        # skip_size
        self._skip_size_silder = gui.Slider(gui.Slider.INT)
        self._skip_size_silder.set_limits(5, 100)
        self._skip_size_silder.set_on_value_changed(self._set_skip_size)
        
        # add
        dataset_config.add_fixed(separation_height)
        dataset_config.add_child(gui.Label("Skip size"))
        dataset_config.add_child(self._skip_size_silder)

        # add 
        self._settings_panel.add_child(dataset_config)

        # add scene
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # initial scene
        self._scene.scene.add_geometry('frame_init', self.frame, self.mat)
        self._scene.scene.add_geometry('box_init', self.mesh_box, self.mat)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

    def _set_vis_mode_video(self):
        if self.thread_finished:
            threading.Thread(target=self.update_trajectory_video).start()

    def _set_vis_mode_afterimage(self):
        if self.thread_finished:
            self.update_trajectory()
    
    def _set_pour_mode_water(self):
        self.remove_trajectory()
        self.pour_style = 'water'
        
    def _set_pour_mode_wine(self):
        self.remove_trajectory()
        self.pour_style = 'wine'

    def _set_dir_idx(self, value):
        self.remove_trajectory()
        self.dir_idx = int(value)

    def _set_skip_size(self, value):
        self.remove_trajectory()
        self.skip_size = int(value)    

    def load_data(self):
        file_ = f'{self.dir_idx}_{self.pour_style}_{self.pour_amount}.pkl'
        print(f'The name of the file is "{file_}"')
        with open(os.path.join(self.root, file_), 'rb') as f:
            data = pickle.load(f)
        
        # trajectory of bottle
        self.traj = data['traj']
        self.traj = self.traj@np.array(
                [[
                    [1., 0., 0., data['offset'][0]], 
                    [0., 1., 0., data['offset'][1]], 
                    [0., 0., 1., data['offset'][2]], 
                    [0., 0., 0., 1.]]])

        # mesh list
        self.bottle_idx = data['bottle_idx']
        self.mug_idx = data['mug_idx']

        # load bottle
        self.mesh_bottle = get_mesh_bottle(
            root='./3dmodels/bottles', 
            bottle_idx=self.bottle_idx
        )
        self.mesh_bottle.compute_vertex_normals()

        # new bottle coloring
        bottle_normals = np.asarray(self.mesh_bottle.vertex_normals)
        bottle_colors = np.ones_like(bottle_normals)
        bottle_colors[:, :3] = rgb[8]
        self.mesh_bottle.vertex_colors = o3d.utility.Vector3dVector(bottle_colors)

        if self.draw_bottle_label:
            self.mesh_bottle_label = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0355, height=0.07, resolution=30, create_uv_map=True, split=20)
            self.mesh_bottle_label.paint_uniform_color([0.8, 0.8, 0.8])
            self.mesh_bottle_label.compute_vertex_normals()

            # initialize
            bottle_cylinder_vertices = np.asarray(self.mesh_bottle_label.vertices)
            bottle_cylinder_normals = np.asarray(self.mesh_bottle_label.vertex_normals)
            bottle_cylinder_colors = np.ones_like(bottle_cylinder_normals)
            # print(np.min(bottle_cylinder_vertices), np.max(bottle_cylinder_vertices))
            
            # band
            n = bottle_cylinder_normals[:, :2]
            n = n/np.linalg.norm(n, axis=1, keepdims=True)
            bottle_cylinder_colors[
                np.logical_and(np.logical_and(n[:, 0] > 0.85, bottle_cylinder_vertices[:, 2] > -0.025), bottle_cylinder_vertices[:, 2] < 0.025)
            ] = rgb[0]
            
            self.mesh_bottle_label.vertex_colors = o3d.utility.Vector3dVector(
                bottle_cylinder_colors
            )

            self.mesh_bottle_label.translate([0.0, 0, 0.155])
            R = self.mesh_bottle_label.get_rotation_matrix_from_xyz((0, 0, self.bottle_label_angle))
            self.mesh_bottle_label.rotate(R, center=(0, 0, 0))

        # load mug
        self.mesh_mug = get_mesh_mug(
            root='./3dmodels/mugs', 
            mug_idx=self.mug_idx
        )
        self.mesh_mug.paint_uniform_color(rgb[2] * 0.6)
        self.mesh_mug.compute_vertex_normals()

    def update_trajectory(self):
        # load data
        self.load_data()
            
        # update initials
        self._scene.scene.add_geometry('frame_init', self.frame, self.mat)
        self._scene.scene.add_geometry('mug_init', self.mesh_mug, self.mat)
        self._scene.scene.add_geometry('box_init', self.mesh_box, self.mat)

        # update trajectory
        for idx in range(0, len(self.traj), self.skip_size):
            mesh_bottle_ = deepcopy(self.mesh_bottle)
            frame_ = deepcopy(self.frame)
            T = self.traj[idx]
            mesh_bottle_.transform(T)
            frame_.transform(T)
            if self.draw_bottle_label:
                mesh_bottle_label_ = deepcopy(self.mesh_bottle_label)
                mesh_bottle_label_.transform(T)

            self._scene.scene.add_geometry(f'bottle_{idx}', mesh_bottle_, self.mat)
            # self._scene.scene.add_geometry(f'coord_{idx}', frame_, self.mat)
            if self.draw_bottle_label:
                self._scene.scene.add_geometry(f'bottle_label_{idx}', mesh_bottle_label_, self.mat)

    def update_trajectory_video(self):
        # load data
        self.load_data()

        # update trajectory
        self.thread_finished = False
        end_idx = range(0, len(self.traj), 5)[-1]
        for idx in range(0, len(self.traj), 5):
            self.mesh_bottle_ = deepcopy(self.mesh_bottle)
            self.frame_ = deepcopy(self.frame)
            T = self.traj[idx]
            self.mesh_bottle_.transform(T)
            self.frame_.transform(T)
            if self.draw_bottle_label:
                self.mesh_bottle_label_ = deepcopy(self.mesh_bottle_label)
                self.mesh_bottle_label_.transform(T)
            self.idx = idx
            if idx == end_idx:
                self.mesh_box.paint_uniform_color([222/255,184/255,135/255])
            else:
                color_scale = 4
                self.mesh_box.paint_uniform_color(
                    [222/255/color_scale,184/255/color_scale,135/255/color_scale]
                )
            # Update geometry
            gui.Application.instance.post_to_main_thread(self.window, self.update_bottle_coord)
            time.sleep(0.05)
        self.thread_finished = True

    def update_bottle_coord(self):
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry('frame_init', self.frame, self.mat)
        self._scene.scene.add_geometry('mug_init', self.mesh_mug, self.mat)
        self._scene.scene.add_geometry('box_init', self.mesh_box, self.mat)
        self._scene.scene.add_geometry(
            f'bottle_{self.idx}', self.mesh_bottle_, self.mat
        )
        self._scene.scene.add_geometry(
            f'coord_{self.idx}', self.frame_, self.mat)
        if self.draw_bottle_label:
            self._scene.scene.add_geometry(
                f'bottle_label_{self.idx}', self.mesh_bottle_label_, self.mat
            )            
        
    def remove_trajectory(self):
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry('frame_init', self.frame, self.mat)
        self._scene.scene.add_geometry('box_init', self.mesh_box, self.mat)

if __name__ == "__main__":
    gui.Application.instance.initialize()
    w = AppWindow()
    
    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()