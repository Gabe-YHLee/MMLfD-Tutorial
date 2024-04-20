import torch
import numpy as np

from sklearn.mixture import GaussianMixture

import time
import threading
from datetime import datetime

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from copy import deepcopy
import copy

from vis_utils.open3d_utils import (
    get_mesh_bottle, 
    get_mesh_mug, 
)

from loader.Pouring_dataset import Pouring

from models.rfm import SE3FlowMatching
from models.modules import vf_FC_se3

from utils.LieGroup_torch import log_SO3, skew, exp_so3
from utils.utils import SE3smoothing

from models.mmp import MMP, NRMMP
from models.modules import FC_SE32vec, FC_vec2SE3, vf_FC_vec, FC_vec
from models.lfm import FlowMatching

import scipy

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
        self.skip_size = 5
        self.thread_finished = True
        self.text = ["Give me anything to drink, please."]
        self.device = f'cuda:0'
        
        encoder2 = FC_SE32vec(
            in_chan=480*12,
            out_chan=2,
            l_hidden=[2048, 1024, 512, 256, ],
            activation=['gelu', 'gelu', 'gelu', 'gelu',],
            out_activation='linear'
        )
        decoder2 = FC_vec2SE3(
            in_chan=2,
            out_chan=480*6,
            l_hidden=[256, 512, 1024, 2048, ],
            activation=['gelu', 'gelu', 'gelu', 'gelu',],
            out_activation='linear'
        )
        nrmmp = NRMMP(
            encoder2, 
            decoder2, 
            approx_order=1, 
            kernel={'type': 'binary', 'lambda':0.1})

        velocity_field = vf_FC_vec(
            in_chan=4+2+1, 
            out_chan=2, 
            l_hidden=[1024, 1024, ],
            activation=['gelu', 'gelu', ],
            out_activation='linear'
        )
        text_embedder = FC_vec(
            in_chan=768, 
            out_chan=4, 
            l_hidden=[1024, 1024, ],
            activation=['gelu', 'gelu', ],
            out_activation='linear'
        )
        lfm = FlowMatching(
            velocity_field,
            text_embedder,
            z_dim=2,
            mmp=nrmmp
        )
        
        load_dict = torch.load("results/nrmmfp.pkl", map_location='cpu')
        ckpt = load_dict["model_state"]
        lfm.load_state_dict(ckpt)
        self.lfm = copy.copy(lfm).to(self.device)
        self.kwargs = {'dt': 0.1, 'guidance': 1.5}

        # parameters 
        image_size = [1024, 768]
        
        # mesh table
        self.mesh_box = o3d.geometry.TriangleMesh.create_box(
            width=2, height=2, depth=0.03
        )
        self.mesh_box.translate([-1, -1, -0.03])
        self.mesh_box.paint_uniform_color([222/255,184/255,135/255])
        self.mesh_box.compute_vertex_normals()

        # bottle label
        self.draw_bottle_label = True
        self.bottle_label_angle = 0

        # frame
        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1
        )

        # object material
        self.mat = rendering.MaterialRecord()
        self.mat.shader = 'defaultLit'
        self.mat.base_color = [1.0, 1.0, 1.0, 0.9]
        mat_prev = rendering.MaterialRecord()
        mat_prev.shader = 'defaultLitTransparency'
        mat_prev.base_color = [1.0, 1.0, 1.0, 0.7]
        mat_coord = rendering.MaterialRecord()
        mat_coord.shader = 'defaultLitTransparency'
        mat_coord.base_color = [1.0, 1.0, 1.0, 0.87]

        ######################################################
        ################# STARTS FROM HERE ###################
        ######################################################

        # set window
        self.window = gui.Application.instance.create_window(str(datetime.now().strftime('%H%M%S')), width=image_size[0], height=image_size[1])
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
        inference_config = gui.CollapsableVert("Inference config", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        # # sample
        # self._sample_button = gui.Button("Sample!")
        # self._sample_button.horizontal_padding_em = 0.5
        # self._sample_button.vertical_padding_em = 0
        # self._sample_button.set_on_clicked(self._set_sampler)
        # h = gui.Horiz(0.25 * em)  # row 1
        # h.add_stretch()
        # h.add_child(self._sample_button)
        # h.add_stretch()

        # # add
        # inference_config.add_child(gui.Label("Sample robot trajectory"))
        # inference_config.add_child(h)

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
        inference_config.add_child(gui.Label("Sample visualize type"))
        inference_config.add_child(h)

        # text
        self._text_editor = gui.TextEdit()
        self._text_editor.set_on_value_changed(self._set_text)
        
        # add
        inference_config.add_fixed(separation_height)
        inference_config.add_child(gui.Label("Text editor (press enter)"))
        inference_config.add_child(self._text_editor)

        # text
        self._current_text = gui.TextEdit()
        
        # add
        inference_config.add_fixed(separation_height)
        inference_config.add_child(gui.Label("Current text"))
        inference_config.add_child(self._current_text)

        # direction
        self._skip_size_silder = gui.Slider(gui.Slider.INT)
        self._skip_size_silder.set_limits(5, 100)
        self._skip_size_silder.set_on_value_changed(self._set_skip_size)
        
        # add
        inference_config.add_fixed(separation_height)
        inference_config.add_child(gui.Label("Skip size"))
        inference_config.add_child(self._skip_size_silder)

        # bottle label angle
        self._bottle_label_angle_silder = gui.Slider(gui.Slider.DOUBLE)
        self._bottle_label_angle_silder.set_limits(0, 2 * np.pi)
        self._bottle_label_angle_silder.set_on_value_changed(self._set_bottle_label_angle)
        
        # add
        inference_config.add_fixed(separation_height)
        inference_config.add_child(gui.Label("Bottle label angle"))
        inference_config.add_child(self._bottle_label_angle_silder)

        # add 
        self._settings_panel.add_child(inference_config)

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
            self.remove_trajectory()
            self.update_trajectory()

    def _set_sampler(self):
        self.remove_trajectory()

    def _set_text(self, value):
        self.remove_trajectory()
        self.text = [value]
        self._current_text.text_value = value

    def _set_skip_size(self, value):
        self.remove_trajectory()
        self.skip_size = int(value)

    def _set_bottle_label_angle(self, value):
        self.remove_trajectory()
        self.bottle_label_angle = value     

    def sample_trajectories(self):
        x_gen = self.lfm.sample(
            self.text, 
            self.device, 
            smoothing=False, 
            **self.kwargs
        )
        x_gen = SE3smoothing(x_gen, mode='savgol')
  
        self.traj = x_gen[0].detach().cpu()

        # mesh list
        self.bottle_idx = 1
        self.mug_idx = 4

        # load bottle
        self.mesh_bottle = get_mesh_bottle(
            root='./3dmodels/bottles', 
            bottle_idx=self.bottle_idx
        )
        self.mesh_bottle.compute_vertex_normals()

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

        # # bottle coloring
        # bottle_vertices = np.asarray(self.mesh_bottle.vertices) 
        # bottle_normals = np.asarray(self.mesh_bottle.vertex_normals)
        # # bottle_colors = np.sqrt(np.ones_like(bottle_normals) - (bottle_normals - 1) ** 2)
        # bottle_colors = 0.8*np.ones_like(bottle_normals)
        # n = bottle_normals[:, :2]
        # n = n/np.linalg.norm(n, axis=1, keepdims=True)
        # bottle_colors[n[:, 0] > 0.99] = np.array([1, 0, 0])
        # n = bottle_normals[:, [0, 2]]
        # n = n/np.linalg.norm(n, axis=1, keepdims=True)
        # angle = np.arctan2(n[:, 1], n[:, 0])
        # h = (angle/np.pi/2).reshape(-1, 1)
        # s = 80/100 * np.ones_like(h)
        # v = 80/100 * np.ones_like(h)

        # new bottle coloring
        bottle_vertices = np.asarray(self.mesh_bottle.vertices) 
        bottle_normals = np.asarray(self.mesh_bottle.vertex_normals)
        bottle_colors = np.ones_like(bottle_normals)
        bottle_colors[:, :3] = rgb[8]
        # z_values = bottle_vertices[:, 2]
        # print(np.max(z_values), np.min(z_values))
        # bottle_colors[np.logical_and(z_values > 0.17, z_values < 0.2)] = rgb[6]
        self.mesh_bottle.vertex_colors = o3d.utility.Vector3dVector(
            bottle_colors
        )

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
            
            # # gradation
            # bottle_cylinder_colors = np.zeros_like(bottle_cylinder_normals)
            # bottle_cylinder_thetas = np.arctan2(bottle_cylinder_vertices[:, 1], bottle_cylinder_vertices[:, 0])
            # bottle_cylinder_colors[:, 0] = (1 + np.cos(3 * bottle_cylinder_thetas)) / 2 * 0.7 
            # bottle_cylinder_colors[:, 1] = (1 + np.sin(3 * bottle_cylinder_thetas)) / 2 * 0.7
    
            self.mesh_bottle_label.vertex_colors = o3d.utility.Vector3dVector(
                bottle_cylinder_colors
            )

            self.mesh_bottle_label.translate([0.0, 0, 0.155])
            R = self.mesh_bottle_label.get_rotation_matrix_from_xyz((0, 0, self.bottle_label_angle))
            self.mesh_bottle_label.rotate(R, center=(0, 0, 0))

        # # # combine
        # self.mesh_bottle += self.mesh_bottle_label

        # load mug
        self.mesh_mug = get_mesh_mug(
            root='./3dmodels/mugs', 
            mug_idx=self.mug_idx
        )
        self.mesh_mug.paint_uniform_color(rgb[2] * 0.6)
        self.mesh_mug.compute_vertex_normals()

    def update_trajectory(self):
            
        # load data
        self.sample_trajectories()
            
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
        self.sample_trajectories()

        # update trajectory
        self.thread_finished = False
        skip_size = 5
        end_idx = range(0, len(self.traj), skip_size)[-1]
        for idx in range(0, len(self.traj), skip_size):
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