import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from utils.utils import label_to_color, figure_to_array, PD_metric_to_ellipse

from models.geometry import (
    relaxed_distortion_measure,
)

from sklearn.mixture import GaussianMixture

from models.lbf import Gaussian_basis, lbf, phi, LfD, vbf
from utils.utils import MMDCalculator
mmd_func = MMDCalculator(type_='L2_traj')
  
class MMPpp(nn.Module):
    def __init__(
        self, 
        encoder, 
        decoder, 
        dof=2,
        b=30,
        h_mul=1,
        basis='Gaussian',
        mode='promp',
        **kwargs
        ):
        super(MMPpp, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        self.b = b
        self.h_mul = h_mul
        self.basis = basis
        self.dof = dof 
        self.mode = mode
        self.kwargs = kwargs
        if self.mode == 'vmp':
            assert len(self.kwargs['via_points']) == 2
    
    def get_w_from_traj(self, x):
        '''
        x : (bs, L, dof)
        '''
        if self.mode == 'promp':
            w = LfD(x, basis=self.basis, b=self.b, h_mul=self.h_mul) # (bs, b, dof)
        elif self.mode == 'vmp':
            w = LfD(x, mode='vmp', basis=self.basis, b=self.b, h_mul=self.h_mul, **self.kwargs) # (bs, b, dof)
        return w.view(len(x), -1).detach() # (bs, b x dof)
    
    def encode(self, w):
        return self.encoder(w) 
        
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, w):
        z = self.encode(w)
        recon = self.decode(z)
        return recon
    
    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        w = self.get_w_from_traj(x)
        z = self.encode(w)
        recon = self.decode(z)
        loss = ((recon - w) ** 2).view(len(w), -1).mean(dim=1).mean()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}
    
    def validation_step(self, x, **kwargs):
        w = self.get_w_from_traj(x)
        recon = self(w)
        loss = ((recon - w) ** 2).view(len(w), -1).mean(dim=1).mean()
        return {"loss": loss.item()}
    
    def eval_step(self, dl, device='cpu', **kwargs):
        data_traj = dl.dataset.data
        self.fit_GMM(data_traj.to(device))
        
        sample_traj = self.sample(500, device=device, traj_len=data_traj.size(1))['q_traj_samples']
        mmd = mmd_func(data_traj.to(device), sample_traj)
        return {"mmd_": mmd}
    
    def fit_GMM(self, traj, n_components=2, **kwargs):
        w = self.get_w_from_traj(traj)
        z = self.encode(w).detach().cpu()
        self.gmm = GaussianMixture(
            n_components=n_components, 
            random_state=0,
            ).fit(z)
        self.gmm_thr = self.gmm.score_samples(z).min()
        
    def sample(self, n_samples, device=f'cuda:0', traj_len=201, clipping=True):
        Z_samples = []
        W_samples = []
        Q_traj_samples = []
        Cluster_samples = []
        num_data = 0
        while num_data < n_samples:
            samples = self.gmm.sample(n_samples=n_samples)
            z_samples = torch.tensor(
                samples[0],
                dtype=torch.float32            
                ).to(device)
            
            if clipping:
                idx = self.gmm.score_samples(z_samples.detach().cpu()) > self.gmm_thr
            else:
                idx = self.gmm.score_samples(z_samples.detach().cpu()) > -9999999
            cluster_samples = torch.tensor(samples[1], dtype=torch.long)
            w_samples = self.decode(z_samples).detach()
            w_samples = w_samples.view(n_samples, self.b, self.dof)
            z_values = torch.linspace(0, 1, traj_len).view(
                    1, -1, 1).repeat(n_samples, 1, 1).to(device)
            basis_values = Gaussian_basis(
                z_values,
                b=self.b)
            if self.mode == 'promp':
                q_traj_samples = lbf(
                    phi(basis_values), 
                    w_samples) # n_samples, traj_len, dof
            elif self.mode == 'vmp':
                q_traj_samples = vbf(
                    z_values, 
                    phi(basis_values), 
                    w_samples, 
                    **self.kwargs) # n_samples, traj_len, dof
            
            Z_samples.append(z_samples[idx]) 
            W_samples.append(w_samples[idx])
            Q_traj_samples.append(q_traj_samples[idx])
            Cluster_samples.append(cluster_samples[idx])
            num_data += len(q_traj_samples[idx])
        Z_samples = torch.cat(Z_samples, dim=0)[:n_samples]
        W_samples = torch.cat(W_samples, dim=0)[:n_samples]
        Q_traj_samples = torch.cat(Q_traj_samples, dim=0)[:n_samples]
        Cluster_samples = torch.cat(Cluster_samples, dim=0)[:n_samples]
        return {
            'z_samples': Z_samples,
            'w_samples': W_samples,
            'q_traj_samples': Q_traj_samples,
            'cluster_samples': Cluster_samples
        }

class IMMPpp(MMPpp):
    def __init__(
        self, 
        encoder, 
        decoder, 
        dof=2,
        b=30,
        h_mul=1,
        basis='Gaussian',
        mode='promp',
        iso_reg=1,
        metric='curve',
        **kwargs
        ):
        super(IMMPpp, self).__init__(
            encoder, 
            decoder, 
            dof=dof,
            b=b,
            h_mul=h_mul,
            basis=basis,
            mode=mode,
            **kwargs
        )   
        self.iso_reg = iso_reg
        self.metric = metric
        self.set_Riemannian_metric()
        if self.metric == 'curve-invden':
            ds = kwargs['ds']
            self.traj_dataset = ds.data
             
    def inverse_density(self, w):
        # w : (bs, b x dim)
        traj_dataset = self.traj_dataset.view(len(self.traj_dataset), -1) # total_data, total_dim
        traj = self.decode(w).view(len(w), -1)
        dist = torch.norm(traj.view(1, len(w), -1) - traj_dataset(len(traj_dataset), 1, -1), dim=-1)
        return dist.min(dim=1)**2 # (len(w), )        
     
    def set_Riemannian_metric(self, num=10000):
        # (bs, theta, theta) 
        z = torch.linspace(0, 1, num).view(1, -1, 1) # 1, num, 1
        if self.basis == 'Gaussian':
            basis = Gaussian_basis(z, b=self.b, h_mul=self.h_mul) # (1, num, b)
            dq_dw = phi(basis) # (1, num, b)
            if self.mode == 'vbf':
                dq_dw = (z)*(1-z)*phi # (1, num, b)
        self.H = (dq_dw.permute(0, 2, 1)@dq_dw)/num # (1, b, b)
    
    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        w = self.get_w_from_traj(x)
        z = self.encode(w)
        recon = self.decode(z)
        loss = ((recon - w) ** 2).view(len(w), -1).mean(dim=1).mean()
        iso_loss = relaxed_distortion_measure(
            self.decode, 
            z, 
            eta=0.2, 
            metric=self.metric, 
            dim=x.size(2),
            H=self.H,
            invden=self.inverse_density
            )
        
        loss = loss + self.iso_reg * iso_loss
        
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "iso_loss": iso_loss.item()}