import numpy as np
import torch
import torch.nn as nn
import copy

from utils.utils import ode_integrator
from utils.utils import MMDCalculator
from utils.LieGroup_torch import exp_so3, skew, log_SO3

from tqdm import tqdm
mmd_func = MMDCalculator(type_='SE3_traj')
from utils.utils import SE3smoothing

class SE3FlowMatching(nn.Module):
    def __init__(
        self, 
        velocity_field,
        traj_len=480,
        prob_path='OT',
        sigma_1=0.01,
        ):
        super(SE3FlowMatching, self).__init__()
        
        self.velocity_field = velocity_field
        self.prob_path = prob_path
        self.sigma_1 = sigma_1
        self.traj_len = traj_len
    
    def forward(self, t, x):
        """_summary_
        Args:
            t (torch.tensor): bs, 1
            x (torch.tensor): bs, traj_len, 4, 4
        """
        return self.velocity_field(t, x) # se3 * traj_len
    
    def sample_from_p0(self, num_samples, device):
        p_samples = 0.1*torch.randn(num_samples*self.traj_len, 3).to(device)
        R_samples = exp_so3(0.1*skew(torch.randn(num_samples*self.traj_len, 3))).to(device)
        T_samples = torch.cat(
            [
                R_samples.view(num_samples, self.traj_len, 3, 3),
                p_samples.view(num_samples, self.traj_len, 3, 1)
            ], dim=-1)
        T_samples = torch.cat(
            [
                T_samples,
                torch.tensor([[[[0, 0, 0, 1]]]], dtype=torch.float32, device=device).repeat(
                    num_samples, self.traj_len, 1, 1
                )
            ], dim=2
        )
        return T_samples # n_samples, traj_len, 4, 4
    
    def sample(
            self, 
            n_samples=100,
            device='cuda:0',
            method='se3-euler',
            smoothing=False,
            **kwargs
        ):
        rand_init = self.sample_from_p0(n_samples, device)
        
        def func(t, z, device):
            return self(t, z)
        
        gen_x = ode_integrator(
            func, 
            t0=0, 
            t1=1, 
            rand_init=rand_init, 
            method=method, 
            device=device, 
            **kwargs
        )
        if smoothing:
            gen_x = SE3smoothing(gen_x)
        return gen_x
    
    def x_t_and_u_t_xbarx1(self, t, x, x1):
        bs, traj_len, _, _ = x.size()
        
        R = x[:, :, :3, :3]
        p = x[:, :, :3, 3]
        R1 = x1[:, :, :3, :3]
        p1 = x1[:, :, :3, 3]
        
        w = skew(log_SO3((R.permute(0, 1, 3, 2)@R1).view(bs*traj_len, 3, 3))).view(bs, traj_len, 3)
        pdot = p1 - p
        
        Rt = R.permute(0, 1, 3, 2)@exp_so3(
            skew(w.view(-1, 3)*t.repeat(traj_len, 1))).view(bs, traj_len, 3, 3)
        pt = p + t.unsqueeze(-1)*pdot
        
        Tt = torch.cat([
            Rt,
            pt.unsqueeze(-1)
        ], dim=-1)
        
        Tt = torch.cat([
            Tt,
            torch.tensor([[[[0, 0, 0, 1]]]], dtype=torch.float32, device=x.device).repeat(
                    bs, traj_len, 1, 1
                )
            ], dim=2)
        
        return Tt, torch.cat([w, pdot], dim=-1)
        
    def train_step(self, x, optimizer=None, *args, **kwargs):
        bs = len(x)
        
        optimizer.zero_grad()
        t = torch.rand(bs, 1).to(x) # (bs, 1)
        x1 = x # (bs, len, 4, 4)
        x0 = self.sample_from_p0(len(x1), device=x1.device) # (bs, len, 4, 4)
        x_t, u_t = self.x_t_and_u_t_xbarx1(t, x0, x1) # (bs, len, 6)
        v_t = self(t, x_t) # (bs, len, 6)
        loss = ((u_t.detach() - v_t)**2).mean()
        
        loss.backward()
        optimizer.step()
        return {
            'loss': loss.detach().cpu().item()
        }
        
    def eval_step(self, x):
        samples = self.sample(
            100, 
            dt=0.1, 
            device=x.device)
        mmd = mmd_func(x, samples)
        
        return {
            'mmd': mmd,
        }