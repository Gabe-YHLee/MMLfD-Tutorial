import torch
import numpy as np
import os, sys
from omegaconf import OmegaConf
from matplotlib.patches import Circle

pallete = [
    'tab:red',#C5A494',
    'tab:green',#'#844C32',
    'tab:blue',#'#845253',
    'tab:purple',#'#91343D',
    '#b9e2f8', 
    '#23a696', 
    '#f05133', 
    '#D4D1C3', 
    '#8B877C',
    '#A79182', 
    '#3D3D32', 
]

class Toy(torch.utils.data.Dataset):
    def __init__(self, 
        root='datasets/EXP2',
        split='training',
        **kwargs):

        super(Toy, self).__init__()

        data = []
        targets = []
        for file_ in os.listdir(root):
            if file_.endswith('.npy'):
                traj_data = np.load(os.path.join(root, file_))
                data.append(torch.tensor(traj_data[0], dtype=torch.float32).unsqueeze(0))
                targets.append(torch.tensor(int(file_[5]), dtype=torch.long).view(-1))
            elif file_.endswith('yaml'):
                cfg = OmegaConf.load(os.path.join(root, file_))
        data = torch.cat(data, dim=0)
        targets = torch.cat(targets, dim=0) - 1
        
        data.size(), targets.size()

        xlim = cfg['xlim']
        ylim = cfg['ylim']
        start_point_pos = cfg['start_point_pos']
        final_point_pos = cfg['final_point_pos']
        obstacles = cfg['obstacles']

        env = {
            'xlim': xlim,
            'ylim': ylim,
            'start_point_pos': start_point_pos,
            'final_point_pos': final_point_pos,
            'obstacles': obstacles
        }
        
        self.env = env
        
        self.data = data
        self.targets = targets

        print(f"Toy split {split} | {self.data.size()}")
  
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y
    
    def check_coliision(self, traj):
        pos = self.env['obstacles']['pos']
        rad =  self.env['obstacles']['rad']
        no = len(rad)
        pos = torch.tensor(pos, dtype=torch.float32).view(no, 2).to(traj) # (no, 2)
        rad = torch.tensor(rad, dtype=torch.float32).view(no, 1).to(traj) # (no, )
        bs = len(traj)
        
        # traj : (bs, L, dof)
        dist2center = ((pos.view(1, 1, no, 2) - traj.view(bs, -1, 1, 2))**2).sum(dim=-1) # (bs, L, no)
        temp = (dist2center - rad.view(1, 1, -1)**2).view(bs, -1)
        outputs = torch.min(temp, dim=1).values # (bs, no)
        return outputs < 0
        
        
def toy_visualizer(dict_env, ax, traj=None, label=None, alpha=1):
    workspace = [dict_env['xlim'], dict_env['ylim']]
    init = dict_env['start_point_pos']
    goal = dict_env['final_point_pos']
    obstacles = dict_env['obstacles']
    
    ax.set_xlim(workspace[0])
    ax.set_ylim(workspace[1])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.scatter(init[0], init[1], marker='s', s=100, c='k')
    ax.scatter(goal[0], goal[1], marker='*', s=100, c='k')
    
    for xy, rad in zip(obstacles['pos'], obstacles['rad']):
        Obs = Circle(xy=xy, radius=rad, color='tab:orange')
        ax.add_patch(Obs)
    
    if traj is not None:
        if label is not None:
            for data, target in zip(traj, label):
                ax.plot(
                    data[:, 0], 
                    data[:, 1],
                    c=pallete[target],
                    alpha=alpha)
        else:
            for data in traj:
                ax.plot(data[:, 0], data[:, 1], c=pallete[0], alpha=alpha)