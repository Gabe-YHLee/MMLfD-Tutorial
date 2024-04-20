import torch
import utils.LieGroup_torch as lie
from robot.groups import PouringGroup
import numpy as np
from models import load_pretrained

import pickle
import os
from robot.franka_sub import Franka as Franka_sub


from robot import Kinematics_numpy
from utils.utils import SE3smoothing
from robot.sq_collision import check_collision
from scipy import signal

def modify_traj(traj_table:torch.Tensor, cup_pos:torch.Tensor, theta1_trans, theta2_rot):
    """_summary_
    Args:
        traj_table (torch.Tensor): trajectory from bottle at (0.6, 0) to cup at (0, 0)
        cup_pos (torch.Tensor): new cup position
        theta1_trans (_type_): rotation angle of the bottle around the cup (translation.)
        theta2_rot (_type_): rotation angle of the bottle around itself
    """
    group = PouringGroup()
    # Shaping
    original_traj_shape = traj_table.shape
    if len(traj_table.shape) == 3:
        traj_table = traj_table.unsqueeze(0)
    if len(cup_pos.shape) == 1:
        cup_pos = cup_pos.unsqueeze(0)
    if type(theta1_trans) != torch.Tensor:
        theta1_trans = torch.tensor(theta1_trans).to(cup_pos)
        theta2_rot = torch.tensor(theta2_rot).to(cup_pos)
    if len(theta1_trans.shape) == 0:
        theta1_trans = theta1_trans.unsqueeze(0).unsqueeze(0)
    elif len(theta1_trans.shape) == 1:
        theta1_trans = theta1_trans.unsqueeze(1)
    if len(theta2_rot.shape) == 0:
        theta2_rot = theta2_rot.unsqueeze(0).unsqueeze(0)
    elif len(theta2_rot.shape) == 1:
        theta2_rot = theta2_rot.unsqueeze(1)
    num_traj = len(traj_table)
    num_pos = len(cup_pos)
    num_theta1_onlytrans = len(theta1_trans)
    if num_pos == 1 and num_traj != 1:
        cup_pos = cup_pos.repeat(num_traj, 1)
    if num_pos == 1 and num_theta1_onlytrans != 1:
        theta1_trans = theta1_trans.repeat(num_traj, 1)
        theta2_rot = theta2_rot.repeat(num_traj, 1)
    if len(cup_pos) != num_traj:
        raise ValueError("cup_pos.shape != traj_table.shape")
    
    # get group element
    zero_ = torch.zeros(num_traj, 1).to(traj_table)
    h = torch.cat([cup_pos, theta1_trans, theta2_rot - theta1_trans], dim=1)
    
    # task parameter for group action
    tau = torch.zeros(num_traj, 6).to(traj_table)
    tau[:, 2] = 0.6
    h_tau, h_traj_table = group.action_traj(h, tau, traj_table)
    
    return h_traj_table.reshape(*original_traj_shape)

def adjust_grasp_point(traj:torch.Tensor, grasp_point_height):
    T = torch.eye(4).unsqueeze(0)
    T[0, 2, 3] += grasp_point_height
    traj_transformed = traj@T
    return traj_transformed
    
def traj_tablebase_to_robotbase(traj:torch.Tensor, alpha=0):
    T_tablebase = torch.eye(4).to(traj)
    T_tablebase[:3, :3] = torch.tensor([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
        ], dtype=torch.float).to(traj)
    trans = torch.tensor([0.95, -0.2656, 0.255])
    T_tablebase[:3, 3] = trans.to(traj)
    traj_out = T_tablebase.unsqueeze(0) @ traj
    return traj_out

def traj_robotbase_to_tablebase(traj:torch.Tensor, alpha=0):
    T_tablebase = torch.eye(4).to(traj)
    T_tablebase[:3, :3] = torch.tensor([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
        ], dtype=torch.float).to(traj)
    trans = torch.tensor([0.95, -0.2656, 0.255])
    T_tablebase[:3, 3] = trans.to(traj)
    T_robotbase = torch.inverse(T_tablebase)
    traj_out = T_robotbase.unsqueeze(0) @ traj
    return traj_out

def traj_robotbase_bottle_to_EE(traj):
    w1 = torch.zeros(1, 3)
    w1[0, 2] = torch.pi/4
    R1 = lie.exp_so3(w1)
    p1 = torch.tensor([0.0, 0.0, 0.0]).reshape(1, 3, 1)
    eye = torch.eye(4)[-1].reshape(1, 1, 4)
    T_ee1 = torch.cat([torch.cat([R1, p1], dim=2), eye], dim=1).to(traj)
    w2 = torch.zeros(1, 3)
    w2[0, 0] =  torch.pi/2
    R2 = lie.exp_so3(w2)
    p2 = torch.tensor([0.0, 0.0, 0.0]).reshape(1, 3, 1)
    eye = torch.eye(4)[-1].reshape(1, 1, 4)
    T_ee2 = torch.cat([torch.cat([R2, p2], dim=2), eye], dim=1).to(traj)
    traj_out = traj @ T_ee2 @ T_ee1 
    return traj_out

def traj_robotbase_EE_to_bottle(traj):
    w1 = torch.zeros(1, 3)
    w1[0, 2] = torch.pi/4
    R1 = lie.exp_so3(w1)
    p1 = torch.tensor([0.0, 0.0, 0.0]).reshape(1, 3, 1)
    eye = torch.eye(4)[-1].reshape(1, 1, 4)
    T_ee1 = torch.cat([torch.cat([R1, p1], dim=2), eye], dim=1).to(traj)
    T_bottle1 = torch.inverse(T_ee1)
    w2 = torch.zeros(1, 3)
    w2[0, 0] =  torch.pi/2
    R2 = lie.exp_so3(w2)
    p2 = torch.tensor([0.0, 0.0, 0.0]).reshape(1, 3, 1)
    eye = torch.eye(4)[-1].reshape(1, 1, 4)
    T_ee2 = torch.cat([torch.cat([R2, p2], dim=2), eye], dim=1).to(traj)
    T_bottle2 = torch.inverse(T_ee2)
    traj_out = traj @ T_bottle1 @ T_bottle2  
    return traj_out

def traj_robotbase_EE_to_bottle_numpy_single(T):
    T_torch = torch.from_numpy(T).to(float).unsqueeze(0)
    T_torch_out = traj_robotbase_EE_to_bottle(T_torch)
    return T_torch_out.squeeze(0).numpy()

def SE3_EE_to_bottle(traj):
    w2 = torch.zeros(1, 3)
    # w2[0, 2] = torch.pi/4
    w2[0, 2] = -3 * torch.pi/4
    R2 = lie.exp_so3(w2)
    p2 = torch.tensor([0.0, 0.0, 0.0]).reshape(1, 3, 1)
    eye = torch.eye(4)[-1].reshape(1, 1, 4)
    T_ee_to_bottle = torch.cat([torch.cat([R2, p2], dim=2), eye], dim=1).detach()
    T_ee_to_bottle = T_ee_to_bottle.numpy()

    w = torch.zeros(1, 3)
    w[0, 1] = -torch.pi/2
    R = lie.exp_so3(w)
    p = torch.tensor([0.0, 0.0, 0.0]).reshape(1, 3, 1)
    eye = torch.eye(4)[-1].reshape(1, 1, 4)
    T_ee = torch.cat([torch.cat([R, p], dim=2), eye], dim=1)
    T_ee = T_ee.numpy()
    traj_bottle = traj @ T_ee_to_bottle @ T_ee
    return traj_bottle

def get_grid_search_args(
    cup_trans_x_range=[-0.2, 0.2], 
    cup_trans_y_range=[-0.2, 0.2],
    theta1_range=[-0.4, 0.4],
    theta2_range=[-0.4, 0.4],
    grasp_point_range=[0.05, 0.2],
    num_grid = [5, 5, 5, 5, 4],
    ):
    # num_grid = [cup_trans_x, cup_trans_y, theta1_trans, theta2_rot]
    
    transf_grid = torch.zeros(*num_grid, 5)
    cup_trans_x_linspace = torch.linspace(*cup_trans_x_range, num_grid[0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    cup_trans_y_linspace = torch.linspace(*cup_trans_y_range, num_grid[1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    theta1_linspace = torch.linspace(*theta1_range, num_grid[2]).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    theta2_linspace = torch.linspace(*theta2_range, num_grid[3]).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    grasp_point_linspace = torch.linspace(*grasp_point_range, num_grid[4]).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    transf_grid[..., 0] = cup_trans_x_linspace
    transf_grid[..., 1] = cup_trans_y_linspace
    transf_grid[..., 2] = theta1_linspace
    transf_grid[..., 3] = theta2_linspace
    transf_grid[..., 4] = grasp_point_linspace
    cup_pos_grid = transf_grid[..., 0:2].reshape(-1, 2)
    theta1_trans_grid = transf_grid[..., 2].reshape(-1)
    theta2_rot_grid = transf_grid[..., 3].reshape(-1)
    grasp_point_grid = transf_grid[..., 4].reshape(-1)
    return cup_pos_grid, theta1_trans_grid, theta2_rot_grid, grasp_point_grid

def flip_traj_norot(traj):
    original_shape = traj.shape
    if len(original_shape) == 3:
        traj = traj.unsqueeze(0)
    T_flip = torch.tensor([[[-1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]], dtype=torch.float32)
    T_orientation = torch.tensor([[[-1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]], dtype=torch.float32)
    traj_fliped = (T_flip @ traj @ T_orientation).reshape(*original_shape)
    return traj_fliped

def flip_T_norot_np(T:np.ndarray):
    T_flip = np.array([[-1, 0, 0, 0.8],
                            [0, -1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
    T_orientation = np.array([[-1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
    traj_fliped = (T_flip @ T @ T_orientation)
    return traj_fliped

def load_traj_in_dataset(
    root='../datasets/bigbottle_pkl_subdataset/pouring_data', 
    dir_idx=1, 
    pour_style='wine', 
    pour_amount=200
    ):
    file_ = f'{dir_idx}_{pour_style}_{pour_amount}.pkl'
    print(f'The name of the file is "{file_}"')
    with open(os.path.join(root, file_), 'rb') as f:
        data = pickle.load(f)
    
    # trajectory of bottle
    traj = data['traj']
    traj = traj@np.array(
            [[
                [1., 0., 0., data['offset'][0]], 
                [0., 1., 0., data['offset'][1]], 
                [0., 0., 1., data['offset'][2]], 
                [0., 0., 0., 1.]]])
    return traj

def sample_traj_from_trained_model(text, text2motion, device='cuda:0'):
    x_gen = text2motion.sample(
            text, 
            device, 
            smoothing=False, 
            dt=0.1,
        )
    x_gen = SE3smoothing(x_gen, mode='savgol')
    return x_gen

def transform_traj(traj, cup_x, cup_y, theta1_trans, theta2_grasping, grasp_height=0.12, flip=False):
    traj_torch = torch.from_numpy(traj).to(torch.float)
    if flip == True:
        traj_torch = flip_traj_norot(traj_torch)
    traj_torch = adjust_grasp_point(traj_torch, grasp_height)
    traj_torch = modify_traj(
        traj_table=traj_torch,
        cup_pos=torch.tensor([cup_x, cup_y]).to(torch.float),
        theta1_trans=theta1_trans,
        theta2_rot=theta2_grasping,
    )
    traj_torch = traj_tablebase_to_robotbase(traj_torch)
    traj_EE_torch = traj_robotbase_bottle_to_EE(traj_torch)
    return traj_EE_torch.numpy()

def solve_IK(
    traj_EE, franka_model, 
    show=False, print_=False, 
    q_final = [1.17701699, 1.11877165, -1.58112058, -1.5430262, 1.46616503, 2.62377974, 0.2775677 ],
    return_q=False,
    tolerance=0.001,
):
    q_traj = []
    q = q_final.copy()
    flag = True
    initial_flag = True
    for idx in range(len(traj_EE)-1, -1, -1):
        T_ee = traj_EE[idx]
        q_prev = q.copy()
        try:
            q, dict_infos = Kinematics_numpy.inverse_kinematics(
                franka_model.initialEEFrame,
                T_ee.copy(),
                q_prev.copy(),
                franka_model.S_screw,
                max_iter=5000,
                tolerance=tolerance,
                step_size=0.01,
                step_size2=0.1,
                step_size3=0,
                joint_limit=franka_model.joint_limit,
                singular_avoidance=False,
                debug=True,
                show=show,
            )
            error_continuous = np.linalg.norm(q - q_prev)
            if initial_flag:
                error_continuous = 0
                initial_flag = False
            error = dict_infos['final_error']
            jointlimit_bool = dict_infos['jointlimit_bool']
            q_traj.append(q)
        except:
            error = 10000000
            jointlimit_bool = 0  
            error_continuous = 10000000  
        if print_:
            print(f'traj idx: {idx} solved!, error: {error}, jointlimit_bool: {jointlimit_bool}, error_continuous: {error_continuous}')
        flag = flag and (error < 0.001) and (jointlimit_bool == 1) and (error_continuous < 0.3)
        
        if not flag:
            print(f'traj idx: {idx} failed!, error: {error}, jointlimit_bool: {jointlimit_bool}, error_continuous: {error_continuous}')
            break
        
    # reverse q_traj
    q_traj.reverse()
    if print_:
        print(f"flag : {flag}")
    
    if return_q:
        return q_traj, flag    
    else:
        return flag



def get_feasible_IK_sol(
    trajs, device='cpu', base_dir='./', 
    cup_x=0.2, cup_y=0.15, 
    tolerance=0.001, grasp_height=0.12,
    check_collision_function=None,
    theta1_grid=[-20*np.pi/180, -10*np.pi/180, 0*np.pi/180, 10*np.pi/180, 20*np.pi/180],
    theta2_grid=[-60*np.pi/180, -30*np.pi/180, 0*np.pi/180, 30*np.pi/180, 60*np.pi/180],
    flip_grid=[True, False]
    ):
    # xml_root = os.path.join(base_dir, 'robot_vis')
    franka_xml_path = os.path.join(base_dir, 'robot_vis/franka_panda_robot_only_cat.xml')
    franka_model = Franka_sub(franka_xml_path)
    NIK_classifier, _ = load_pretrained(
        identifier='NIK_classifier/n512',
        config_file='NIK_classifier.yml',
        ckpt_file='model_best.pkl',
        root=os.path.join(base_dir, 'results/exp1/')
    )
    NIK_classifier = NIK_classifier.to(device)

    print(f"2.1. Symmetry transformations are applied.")
    trajs_grid_list = []
    setting_list = []
    for traj in trajs:
            for theta1_trans in theta1_grid:
                for theta2_grasping in theta2_grid:
                    for flip in flip_grid:
                        traj_EE = transform_traj(
                                traj, 
                                cup_x=cup_x, 
                                cup_y=cup_y, 
                                theta1_trans=theta1_trans, 
                                theta2_grasping=theta2_grasping,
                                flip=flip,
                                grasp_height=grasp_height
                                )
                        setting_list.append([theta1_trans, theta2_grasping, flip])
                        trajs_grid_list.append(torch.tensor(traj_EE, dtype=torch.float32).unsqueeze(0))
    trajdata = torch.cat(trajs_grid_list, dim=0).to(device)
    settings_numpy = np.array(setting_list)
    print(f"The number of augmented trajectories is {len(trajdata)}.")
    print(f"2.2. Apply Nueral IK Classifier")
    flags = torch.round(NIK_classifier(trajdata)).to(torch.long).view(-1)
    feasible_SE3_trajs = trajdata[flags==1]
    feasible_setting_list = settings_numpy[flags==1]
    # feasible_SE3_trajs = trajdata
    # feasible_setting_list = settings_numpy
    flag = False
    i = 0
    
    if len(feasible_SE3_trajs) == 0:
        print (f"no feasible traj exists")
        return None, None
    else:
        print(f"The number of candidate trajectories is {len(feasible_SE3_trajs)}.")
        done = False
        while (not done) and (i < len(feasible_SE3_trajs)):
            SE3traj = feasible_SE3_trajs[i].detach().cpu().numpy()
            setting = feasible_setting_list[i]
            q_traj, flag = solve_IK(SE3traj, franka_model, show=False, print_=False, return_q=True, tolerance=tolerance)
            if not flag:
                print(f"{i+1}-th attempt: IK fail")
                i += 1
            elif check_collision_function is not None:
                collision_score = check_collision_function(q_traj)
                if sum(collision_score) > 0:
                    print(f"{i+1}-th attempt: collision fail")
                    # print(collision_score)
                    i += 1
                    flag = False
                else:
                    done = True
            else:
                done = True
                
        if flag:
            print(f"{i+1}-th attempt: IK success")
            return np.array(q_traj), setting
        else:
            print(f"no sol. exists")
            return None, None

def interpolate_traj(traj, init_fps=60, target_fps=1000):
    len_init_traj = len(traj)
    t_f = int(len_init_traj/init_fps)
    len_target_traj = target_fps * t_f
    target_traj = np.zeros([len_target_traj, *traj.shape[1:]])
    target_traj[0] = traj[0]
    idx_list = [0]
    for i in range(len_init_traj):
        idx = int(target_fps/init_fps*(i+1)) - 1
        idx_list.append(idx)
        target_traj[idx] = traj[i]
    for j in range(len_init_traj):
        start_idx = idx_list[j]
        end_idx = idx_list[j + 1]
        len_current_stage = end_idx - start_idx + 1
        start_point = target_traj[idx_list[j]]
        end_point = target_traj[idx_list[j+1]]
        for k in range(1, len_current_stage-1):
            target_traj[idx_list[j] + k] = start_point + (k*(end_point - start_point))/len_current_stage
    return target_traj

def filter_traj(traj, window_length=50, polyorder=3):
    traj_smoothed = traj.copy()
    for i in range(traj.shape[1]):
        traj_smoothed[:, i] = signal.savgol_filter(traj[:, i], window_length=window_length, polyorder=polyorder)
    return traj_smoothed