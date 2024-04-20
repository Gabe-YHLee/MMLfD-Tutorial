import yaml
import numpy as np
from matplotlib.patches import Ellipse, Rectangle, Polygon
import torch
import copy
from scipy import signal
import re
import os
from utils.LieGroup_torch import log_SO3, skew, exp_so3

def save_yaml(filename, text):
    """parse string as yaml then dump as a file"""
    with open(filename, "w") as f:
        yaml.dump(yaml.safe_load(text), f, default_flow_style=False)

def label_to_color(label):
    
    n_points = label.shape[0]
    color = np.zeros((n_points, 3))

    # color template (2021 pantone color: orbital)
    rgb = np.zeros((11, 3))
    rgb[0, :] = [253, 134, 18]
    rgb[1, :] = [106, 194, 217]
    rgb[2, :] = [111, 146, 110]
    rgb[3, :] = [153, 0, 17]
    rgb[4, :] = [179, 173, 151]
    rgb[5, :] = [245, 228, 0]
    rgb[6, :] = [255, 0, 0]
    rgb[7, :] = [0, 255, 0]
    rgb[8, :] = [0, 0, 255]
    rgb[9, :] = [18, 134, 253]
    rgb[10, :] = [155, 155, 155] # grey

    for idx_color in range(10):
        color[label == idx_color, :] = rgb[idx_color, :]
    return color

def figure_to_array(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

def PD_metric_to_ellipse(G, center, scale, **kwargs):
    
    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(G)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # find angle of ellipse
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # draw ellipse
    width, height = 2 * scale * np.sqrt(eigvals)
    return Ellipse(xy=center, width=width, height=height, angle=np.degrees(theta), **kwargs)

def rectangle_scatter(size, center, color):

    return Rectangle(xy=(center[0]-size[0]/2, center[1]-size[1]/2) ,width=size[0], height=size[1], facecolor=color)

def triangle_scatter(size, center, color):
    
    return Polygon(((center[0], center[1] + size[1]/2), (center[0] - size[0]/2, center[1] - size[1]/2), (center[0] + size[0]/2, center[1] - size[1]/2)), fc=color)


def parse_arg_type(val):
    if val.isnumeric():
        return int(val)
    if (val == 'True') or (val == 'true'):
        return True
    if (val == 'False') or (val == 'false'):
        return False
    try:
        return float(val)
    except:
        return str(val)

def parse_unknown_args(l_args):
    """convert the list of unknown args into dict
    this does similar stuff to OmegaConf.from_cli()
    I may have invented the wheel again..."""
    n_args = len(l_args) // 2
    kwargs = {}
    for i_args in range(n_args):
        key = l_args[i_args*2]
        val = l_args[i_args*2 + 1]
        assert '=' not in key, 'optional arguments should be separated by space'
        kwargs[key.strip('-')] = parse_arg_type(val)
    return kwargs

def parse_nested_args(d_cmd_cfg):
    """produce a nested dictionary by parsing dot-separated keys
    e.g. {key1.key2 : 1}  --> {key1: {key2: 1}}"""
    d_new_cfg = {}
    for key, val in d_cmd_cfg.items():
        l_key = key.split('.')
        d = d_new_cfg
        for i_key, each_key in enumerate(l_key):
            if i_key == len(l_key) - 1:
                d[each_key] = val
            else:
                if each_key not in d:
                    d[each_key] = {}
                d = d[each_key]
    return d_new_cfg

def encode_data(encoder, dataset, device='cuda:0'):
    Z = []
    for data in dataset.split(100):
        x = data.to(device)
        z = copy.copy(encoder(x))
        Z.append(z.detach().cpu())  
    Z = torch.cat(Z, dim=0)
    return Z

def conditional_ode_integrator(model, t0, t1, rand_init, text, method, device, output_traj=False, **kwargs):
    bs = len(rand_init)
    dt = kwargs['dt']
    T = t1 - t0
    traj = [copy.deepcopy(rand_init.unsqueeze(1))]
    if method == 'euler':
        z = rand_init
        t = torch.tensor(t0, dtype=torch.float32).unsqueeze(0).repeat(bs, 1).to(device)
        for _ in range(int(T/dt)):
            zdot = copy.copy(model(t, z, text, device).detach())
            z += zdot*dt
            t += torch.tensor(dt, dtype=torch.float32).to(device)
            traj.append(copy.deepcopy(z.unsqueeze(1)))
        if output_traj:
            return z, torch.cat(traj, dim=1)
        else:
            return z
    elif method == 'se3-euler':
        SE3traj = rand_init # bs x len x 4 x 4 
        t = torch.tensor(t0, dtype=torch.float32).unsqueeze(0).repeat(bs, 1).to(device)
        for _ in range(int(T/dt)):
            se3trajdot = copy.copy(model(t, SE3traj, text, device).detach()) # bs x len x 6 
            so3trajdot = se3trajdot[:, :, :3] # bs x len x 3 
            p3trajdot = se3trajdot[:, :, 3:] # bs x len x 3 
            
            deltaSO3traj = exp_so3(skew(so3trajdot.view(-1, 3))*dt).view(bs, -1, 3, 3)
            SE3traj[:, :, :3, :3] = SE3traj[:, :, :3, :3]@deltaSO3traj
            SE3traj[:, :, :3, 3] += p3trajdot*dt
            t += torch.tensor(dt, dtype=torch.float32).to(device)
        return SE3traj
    
def ode_integrator(model, t0, t1, rand_init, method, device, output_traj=False, **kwargs):
    bs = len(rand_init)
    dt = kwargs['dt']
    T = t1 - t0
    traj = [copy.deepcopy(rand_init.unsqueeze(1))]
    if method == 'euler':
        z = rand_init
        t = torch.tensor(t0, dtype=torch.float32).unsqueeze(0).repeat(bs, 1).to(device)
        for _ in range(int(T/dt)):
            zdot = copy.copy(model(t, z, device).detach())
            z += zdot*dt
            t += torch.tensor(dt, dtype=torch.float32).to(device)
            traj.append(copy.deepcopy(z.unsqueeze(1)))
        if output_traj:
            return z, torch.cat(traj, dim=1)
        else:
            return z
    elif method == 'se3-euler':
        SE3traj = rand_init # bs x len x 4 x 4 
        t = torch.tensor(t0, dtype=torch.float32).unsqueeze(0).repeat(bs, 1).to(device)
        for _ in range(int(T/dt)):
            se3trajdot = copy.copy(model(t, SE3traj, device).detach()) # bs x len x 6 
            so3trajdot = se3trajdot[:, :, :3] # bs x len x 3 
            p3trajdot = se3trajdot[:, :, 3:] # bs x len x 3 
            
            deltaSO3traj = exp_so3(skew(so3trajdot.view(-1, 3))*dt).view(bs, -1, 3, 3)
            SE3traj[:, :, :3, :3] = SE3traj[:, :, :3, :3]@deltaSO3traj
            SE3traj[:, :, :3, 3] += p3trajdot*dt
            t += torch.tensor(dt, dtype=torch.float32).to(device)
        return SE3traj
    
class MMDCalculator:
    def __init__(self, type_, num_episodes=100, kernel_mul=1, kernel_num=1, bandwidth_base=None):
        self.num_episodes = num_episodes
        self.type = type_
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.bandwidth_base = bandwidth_base

    def calculate_squared_distance(self, x, y):
        if self.type is None:
            if (x.size(-1) == 4) and (x.size(-2) == 4):
                self.type == 'SE3_traj'
            else:
                self.type == 'L2_traj' 
        
        # if self.type == 'SE3':
        #     batch_size = len(x)

        #     T_x = x.reshape(-1, 4, 4)
        #     T_y = y.reshape(-1, 4, 4)

        #     dist_R = skew(log_SO3(torch.einsum('bij,bjk->bik', T_x[:, :3, :3].permute(0, 2, 1), T_y[:, :3, :3])))
        #     dist_p = T_x[:, :3, 3] - T_y[:, :3, 3]
        #     return torch.sum(dist_R ** 2 + dist_p ** 2, dim=1).reshape(batch_size, batch_size)
        # if self.type == 'L2':
        #     return ((x - y)**2).sum(dim=2)
        if self.type == 'L2_traj':
            return torch.sum((x-y)**2, dim=-1).mean(dim=-1)
        elif self.type == 'SE3_traj':
            batch_size = len(x)
            T_x = x.reshape(-1, 4, 4)
            T_y = y.reshape(-1, 4, 4)

            dist_R = skew(log_SO3(torch.einsum('bij,bjk->bik', T_x[:, :3, :3].permute(0, 2, 1), T_y[:, :3, :3])))
            dist_p = T_x[:, :3, 3] - T_y[:, :3, 3]
            return torch.sum(10*dist_R ** 2 + dist_p ** 2, dim=1).view(batch_size, batch_size, -1).mean(dim=-1)
        else:
            raise NotImplementedError(f"Type {self.type} is not implemented. Choose type between 'SE3_traj' and 'L2_traj'.")

    def guassian_kernel(self, source, target):
        total = torch.cat([source, target], dim=0)
        total0 = torch.repeat_interleave(total.unsqueeze(1), len(total), dim=1) # bs x bs x ...
        total1 = torch.repeat_interleave(total.unsqueeze(0), len(total), dim=0) # bs x bs x ...

        distance_squared = self.calculate_squared_distance(total0, total1)

        if self.bandwidth_base == None:
            self.bandwidth_base = torch.sum(distance_squared) / (len(total) ** 2 - len(total))

        self.bandwidth_base /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [self.bandwidth_base * (self.kernel_mul ** i) for i in range(self.kernel_num)]

        kernel_val = [torch.exp(-distance_squared / bandwidth) for bandwidth in bandwidth_list]

        return sum(kernel_val)

    def __call__(self, source, target):
        assert len(source) <= len(target), f"The number of samples in source {len(source)} must be less than or equal to the number of samples in target {len(target)}."

        batch_size = len(source)

        mmd_list = []

        for _ in range(self.num_episodes):
            target_ = target[np.random.choice(range(len(target)), len(source), replace=False)]

            kernels = self.guassian_kernel(source, target_)

            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]
            YX = kernels[batch_size:, :batch_size]

            mmd = torch.mean(XX + YY - XY - YX).item()

            mmd_list += [mmd]

        mmd_avg = sum(mmd_list) / len(mmd_list)

        return mmd_avg
    
def Rn_smoothing_single(traj, mode='savgol'):
    if type(traj) == torch.Tensor:
        tensor_mode = True
        traj_device = traj.device
        traj = traj.detach().cpu()
    else:
        tensor_mode = False
        traj = torch.from_numpy(traj).to(torch.float)
    window_length = 50
    polyorder = 3
    traj = signal.savgol_filter(traj, window_length=window_length, polyorder=polyorder, mode="nearest", axis=0)
    if tensor_mode:
        traj = torch.from_numpy(traj).to(traj_device)
    return traj
        
def SE3smoothing(traj, mode='savgol'):
    # input size = (bs, n, 4, 4)
    bs = len(traj)
    n = traj.shape[1]
    if mode == 'moving_average':
        R1 = traj[:, :-1, :3, :3].reshape(-1, 3, 3)
        R2 = traj[:, 1:, :3, :3].reshape(-1, 3, 3)
        p1 = traj[:, :-1, :3, 3:]
        p2 = traj[:, 1:, :3, 3:]
        
        R = R1@exp_so3(0.5*log_SO3(R1.permute(0,2,1)@R2))
        R = R.view(bs, -1, 3, 3)
        p = (p1+p2)/2
        
        traj = torch.cat([
                torch.cat([
                    traj[:, 0:1, :3, :],
                    torch.cat([R, p], dim=-1)  
                ], dim=1),
            traj[:, :, 3:4, :]
            ], dim=2)
    elif mode == 'savgol':
        traj_device = traj.device
        traj = traj.detach().cpu()
        window_length = 50
        polyorder = 3
        R = (traj[:, :, :3, :3]) # size = (bs, n, 3, 3)
        w = skew(log_SO3(R.reshape(-1, 3, 3))).reshape(bs, n, 3)
        w = signal.savgol_filter(w, window_length=window_length, polyorder=polyorder, mode="nearest", axis=1)
        w = torch.from_numpy(w).to(traj)
        R = exp_so3(w.reshape(-1, 3)).reshape(bs, n, 3, 3)
        p = (traj[:, :, :3, 3:]) # size = (bs, n, 3, 1)
        p = signal.savgol_filter(p, window_length=window_length, polyorder=polyorder, mode="nearest", axis=1)
        p = torch.from_numpy(p).to(traj)
        traj = torch.cat(
            [torch.cat([R, p], dim=-1), 
            torch.zeros(bs, n, 1, 4).to(traj)]
            , dim=2)
        traj[..., -1, -1] = 1
        traj = traj.to(traj_device)
    return traj

def approximate_pinv(A, eta=0.000001):
    return np.linalg.pinv(A + eta * np.eye(len(A)))
    # return np.linalg.pinv(A + eta*len(A))
    
def process_existing_name(name):
    # add a number after the name. If exists, change it to the next number. 
    idx_num = name.find('_')
    while True:
        idx_temp = name[idx_num + 1:].find('_')
        if idx_temp == -1:
            break
        else:
            idx_num += idx_temp + 1
    if idx_num == -1:
        return name + '_0'
    else:
        num = name[idx_num + 1:]
        if num.isnumeric():
            num_new = int(num) + 1
            return(name[:idx_num + 1] + str(num_new))
        else:
            return name + '_0'
    