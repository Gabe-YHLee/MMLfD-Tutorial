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
from models.modules import FC_SE32vec, FC_vec2SE3
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

def reparametrize(ts, ts_coordinates, encoded_data):
    # ts : (Len, 2)
    # ts_coordinates : (bs, 2)
    # encoded_data : (bs, 2)
    z1_val = scipy.interpolate.griddata(
        ts_coordinates, 
        encoded_data[:, 0], 
        ts, 
        method='cubic', 
        rescale=False)
    z2_val = scipy.interpolate.griddata(
        ts_coordinates, 
        encoded_data[:, 1], 
        ts, 
        method='cubic', 
        rescale=False)
    interpolated_points = torch.cat(
        [torch.tensor(z1_val).view(-1, 1), torch.tensor(z2_val).view(-1, 1)], dim=1)
    return interpolated_points
        
class AppWindow:
    def __init__(self):
        # initialize
        self.root = 'datasets/pouring_data'
        self.dir_coordinates = 0
        self.style_coordinates = 0
        self.skip_size = 5
        self.thread_finished = True
        self.model_type = 'mmp'
        
        ds = Pouring(root=self.root)
        
        ##################################
        ########### Load Model ###########
        ##################################
        
        encoder1 = FC_SE32vec(
            in_chan=480*12,
            out_chan=2,
            l_hidden=[2048, 1024, 512, 256, ],
            activation=['gelu', 'gelu', 'gelu', 'gelu',],
            out_activation='linear'
        )
        decoder1 = FC_vec2SE3(
            in_chan=2,
            out_chan=480*6,
            l_hidden=[256, 512, 1024, 2048, ],
            activation=['gelu', 'gelu', 'gelu', 'gelu',],
            out_activation='linear'
        )
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
        self.mmp = MMP(encoder1, decoder1)
        self.nrmmp = NRMMP(encoder2, decoder2, approx_order=1, kernel={'type': 'binary', 'lambda':0.1})

        load_dict = torch.load("results/mmp.pkl", map_location='cpu')
        ckpt = load_dict["model_state"]
        self.mmp.load_state_dict(ckpt)

        load_dict = torch.load("results/nrmmp.pkl", map_location='cpu')
        ckpt = load_dict["model_state"]
        self.nrmmp.load_state_dict(ckpt)
        
        self.mmp_encoded_data = self.mmp.encode(ds.traj_data_).detach().cpu()
        self.nrmmp_encoded_data = self.nrmmp.encode(ds.traj_data_).detach().cpu()
        self.ts_coordinates = ds.labels_[:, [0, 2]].to(torch.float32)
        self.ts_coordinates[:, -1] = (self.ts_coordinates[:, 1] - 1)/4

        ##################################
        ##################################
        ############################### 
        
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

        # model
        self._mmp_button = gui.Button("MMP")
        self._mmp_button.horizontal_padding_em = 0.5
        self._mmp_button.vertical_padding_em = 0
        self._mmp_button.set_on_clicked(self._set_mmp)
        
        self._nrmmp_button = gui.Button("NRMMP")
        self._nrmmp_button.horizontal_padding_em = 0.5
        self._nrmmp_button.vertical_padding_em = 0
        self._nrmmp_button.set_on_clicked(self._set_nrmmp)
        
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._mmp_button)
        h.add_child(self._nrmmp_button)
        h.add_stretch()

        # add
        dataset_config.add_fixed(separation_height)
        dataset_config.add_child(gui.Label("Model type"))
        dataset_config.add_child(h)
        
        # skip_size
        self._dir_coor_silder = gui.Slider(gui.Slider.DOUBLE)
        self._dir_coor_silder.set_limits(0, 1)
        self._dir_coor_silder.set_on_value_changed(self._set_dir_coor)
        
        # add
        dataset_config.add_fixed(separation_height)
        dataset_config.add_child(gui.Label("Direction coordinates"))
        dataset_config.add_child(self._dir_coor_silder)
        
        # skip_size
        self._style_coor_silder = gui.Slider(gui.Slider.DOUBLE)
        self._style_coor_silder.set_limits(0, 1)
        self._style_coor_silder.set_on_value_changed(self._set_style_coor)
        
        # add
        dataset_config.add_fixed(separation_height)
        dataset_config.add_child(gui.Label("Style coordinates"))
        dataset_config.add_child(self._style_coor_silder)
        
        # skip_size
        self._skip_size_silder = gui.Slider(gui.Slider.INT)
        self._skip_size_silder.set_limits(5, 30)
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
    
    def _set_mmp(self):
        self.remove_trajectory()
        self.model_type = 'mmp'
        
    def _set_nrmmp(self):
        self.remove_trajectory()
        self.model_type = 'nrmmp'
        
    def _set_dir_coor(self, value):
        # self.remove_trajectory()
        self.dir_coordinates = float(value)
        self._set_vis_mode_afterimage()
        
    def _set_style_coor(self, value):
        # self.remove_trajectory()
        self.style_coordinates = float(value)
        self._set_vis_mode_afterimage()
        
    def _set_skip_size(self, value):
        self.remove_trajectory()
        self.skip_size = int(value)   

    def load_data(self):
        # trajectory of bottle
        ts = torch.tensor([self.style_coordinates, self.dir_coordinates]).view(1, 2)
        if self.model_type == 'mmp':
            z1_val = scipy.interpolate.griddata(
                self.ts_coordinates, 
                self.mmp_encoded_data[:, 0], 
                ts, 
                method='cubic', 
                rescale=False)
            z2_val = scipy.interpolate.griddata(
                self.ts_coordinates, 
                self.mmp_encoded_data[:, 1], 
                ts, 
                method='cubic', 
                rescale=False)
            z = torch.cat(
            [torch.tensor(z1_val).view(-1, 1), 
                torch.tensor(z2_val).view(-1, 1)],
            dim=1).to(torch.float32)
            self.traj = SE3smoothing(self.mmp.decode(z).detach().cpu())[0]
        elif self.model_type == 'nrmmp':
            z1_val = scipy.interpolate.griddata(
                self.ts_coordinates, 
                self.nrmmp_encoded_data[:, 0], 
                ts, 
                method='cubic', 
                rescale=False)
            z2_val = scipy.interpolate.griddata(
                self.ts_coordinates, 
                self.nrmmp_encoded_data[:, 1], 
                ts, 
                method='cubic', 
                rescale=False)
            z = torch.cat(
            [torch.tensor(z1_val).view(-1, 1), 
                torch.tensor(z2_val).view(-1, 1)],
            dim=1).to(torch.float32)
            self.traj = SE3smoothing(self.nrmmp.decode(z).detach().cpu())[0]
        # mesh list
        self.bottle_idx = 1
        self.mug_idx = 4

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
        self.remove_trajectory()
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
            if self.draw_bottle_label:
                self._scene.scene.add_geometry(f'bottle_label_{idx}', mesh_bottle_label_, self.mat)

    def update_trajectory_video(self):
        # load data
        self.remove_trajectory()
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