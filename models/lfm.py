import numpy as np
import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer

from utils.utils import conditional_ode_integrator
from utils.utils import MMDCalculator
import copy

from tqdm import tqdm
from utils.utils import SE3smoothing
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

class FlowMatching(nn.Module):
    def __init__(
        self, 
        velocity_field,
        text_embedder,
        prob_path='OT',
        sigma_1=0.01,
        z_dim=1,
        mmp=None,
        core_text='give me a drink, anything please.'
        ):
        super(FlowMatching, self).__init__()
        self.velocity_field = velocity_field
        self.text_embedder = text_embedder
        self.prob_path = prob_path
        self.sigma_1 = sigma_1
        self.z_dim = z_dim
        
        assert mmp is not None
        self.mmp = mmp

        if self.mmp.type == 'SE3':
            self.mmd_func = MMDCalculator(type_='SE3_traj')
        else:
            self.mmd_func = MMDCalculator(type_='L2_traj')

        for param in self.mmp.parameters():
            param.requires_grad = False
            
        self.core_text = core_text # this text is true for every motion 

        self.register_buffer('mean', torch.zeros(1, z_dim))
        self.register_buffer('std', torch.ones(1, z_dim))
        
    def normalize(self, dl, device):
        zs = []
        for traj, _, _ in dl:
            z = self.mmp.encode(traj.to(device))
            zs.append(z)
        zs = torch.cat(zs, dim=0)
        self.mean = zs.mean(dim=0, keepdim=True)
        self.std = zs.std(dim=0, keepdim=True)
                    
    def text_embedding(self, query_vec):
        q = self.text_embedder(query_vec) 
        return q
    
    def forward(self, t, z, text, device):
        query_vec = torch.tensor(
            sbert_model.encode(text), 
            dtype=torch.float32
        ).to(device)
        q = self.text_embedding(query_vec) 
        return self.velocity_field(t, z, q) 
    
    def velocity(self, t, z ,text, device, guidance=None):
        if guidance is None:
            return self(t, z, text, device)
        else:
            v0 = self(t, z, [self.core_text]*len(text), device)
            vc = self(t, z, text, device)
            return v0 + guidance*(vc - v0)
    
    def sample(
            self, 
            text, 
            device='cuda:0',
            method='euler',
            sample_z=False,
            guidance=None,
            smoothing=False,
            output_traj=False,
            **kwargs
        ):
        rand_init = torch.randn(len(text), self.z_dim).to(device)
        
        def func(t, z, text, device):
            return self.velocity(t, z, text, device, guidance=guidance)
        
        ode_results = conditional_ode_integrator(
            func, 
            t0=0, 
            t1=1, 
            rand_init=rand_init, 
            text=text, 
            method=method, 
            device=device, 
            output_traj=output_traj,
            **kwargs
        )
        if output_traj:
            gen_z = ode_results[0]
            traj = ode_results[1]
        else:
            gen_z = ode_results 
        
        gen_z = self.std*gen_z + self.mean
        gen_x = self.mmp.decode(gen_z)
        if smoothing:
            gen_x = SE3smoothing(gen_x)
        
        if output_traj:
            traj = self.std.unsqueeze(1)*traj + self.mean.unsqueeze(1)
            return traj
            # trajs = self.mmp.decode(
            #     traj.view(-1, traj.size(-1)))
            # return trajs.view(
            #     len(traj), 
            #     -1, 
            #     trajs.size(1), 
            #     trajs.size(2)
            # )
        elif sample_z:
            return gen_z, gen_x
        else:    
            return gen_x
        
    def Gaussian_t_xbar_x1(self, t, z1, **kwargs):
        if self.prob_path == 'OT':
            mu_t = t*z1
            sigma_t = torch.ones_like(t) - (1-self.sigma_1) * t
        elif self.prob_path == 'VE':
            mu_t = z1
            sigma_t = 3*(1-t)
        elif self.prob_path == 'VP':
            alpha_1mt = torch.exp(-5*(1-t)**2)
            mu_t = alpha_1mt*z1 
            sigma_t = torch.sqrt((1-alpha_1mt**2))       
        elif self.prob_path == 'VP2':
            alpha_1mt = torch.exp(-0.5*(1-t)**2)
            mu_t = alpha_1mt*z1 
            sigma_t = torch.sqrt((1-alpha_1mt**2))       
        return mu_t, sigma_t
    
    def sample_from_Gaussian_t_xbar_x1(self, t, z1, **kwargs):
        mu, sigma = self.Gaussian_t_xbar_x1(t, z1, **kwargs)
        samples = sigma*torch.randn_like(z1) + mu
        return samples
    
    def u_t_xbarx1(self, t, z, z1, **kwargs):
        u = (z1 - (1-self.sigma_1)*z)/(1 - (1-self.sigma_1)*t)
        return u 
    
    def train_step(self, x, text, optimizer=None, *args, **kwargs):
        bs = len(x)
        
        # x_aug = torch.cat([x, x], dim=0)
        # text_aug = text + (self.core_text, )*bs
        # bs = 2*bs
        
        x_aug = x
        text_aug = text
        optimizer.zero_grad()
        
        t = torch.rand(bs, 1).to(x)
        z1 = self.mmp.encode(x_aug)
        z1 = (z1-self.mean)/self.std
        
        z = self.sample_from_Gaussian_t_xbar_x1(t, z1)
        u_t = self.u_t_xbarx1(t, z, z1)
        v_t = self(t, z, text_aug, device=x.device)
        
        loss = ((u_t.detach() - v_t)**2).mean()
        text_emb_loss = torch.zeros(1)
            
        loss.backward()
        optimizer.step()
        return {
            'loss': loss.detach().cpu().item(),
            'text_emb_loss_': text_emb_loss.detach().cpu().item()
        }
        
    def validation_step(self, x, text, optimizer=None, *args, **kwargs):
        bs = len(x)
        t = torch.rand(bs, 1).to(x)
        z1 = self.mmp.encode(x)
        z1 = (z1-self.mean)/self.std
        
        z = self.sample_from_Gaussian_t_xbar_x1(t, z1)
        u_t = self.u_t_xbarx1(t, z, z1)
        v_t = self(t, z, text, device=x.device)
        
        loss = ((u_t.detach() - v_t)**2).mean()
        return {
            'loss': loss.detach().cpu().item()
        }
        
    def eval_step(self, val_loader, device):
        dict_for_evals = val_loader.dataset.dict_for_evals
        mmd_list = []
        for key, item in tqdm(dict_for_evals.items()):
            samples = self.sample(
                100*[key], 
                dt=0.1, 
                sample_z=False, 
                guidance=1, 
                device=device)
            mmd = self.mmd_func(item.to(device), samples)
            mmd_list.append(copy.copy(torch.tensor([mmd], dtype=torch.float32)))
        mmd_torch = torch.cat(mmd_list)
        
        return {
            'mmd_avg_': mmd_torch.mean(),
            'mmd_std_': mmd_torch.std(),
            'mmd_ste_': mmd_torch.std()/np.sqrt(len(mmd_torch)),
        }