import numpy as np
import torch
import torch.nn as nn

from transformers import RobertaTokenizer, RobertaModel
from vis_utils.plotly_SE3 import visualize_SE3

from utils.utils import SE3smoothing
from utils.LieGroup_torch import log_SO3

def get_kernel_function(kernel):
    if kernel['type'] == 'binary':
        def kernel_func(x_c, x_nn):
            '''
            x_c.size() = (bs, dim), 
            x_nn.size() = (bs, num_nn, dim)
            '''
            bs = x_nn.size(0)
            num_nn = x_nn.size(1)
            x_c = x_c.view(bs, -1)
            x_nn = x_nn.view(bs, num_nn, -1)
            eps = 1.0e-12
            index = torch.norm(x_c.unsqueeze(1)-x_nn, dim=2) > eps
            output = torch.ones(bs, num_nn).to(x_c)
            output[index] = kernel['lambda']
            return output # (bs, num_nn)
    return kernel_func

class MMP(nn.Module):
    def __init__(
            self, 
            encoder, 
            decoder, 
            smoothness_weight=10.0, 
            type_='SE3'
        ):
        super(MMP, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.smoothness_weight = smoothness_weight
        self.type = type_
        
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon
    
    def compute_mse(self, x, recon):
        if self.type == 'SE3':
            ## rotation part is weighted 10 times
            bs = len(x)
            xR = x.view(-1, 4, 4)[:, :3, :3]
            xp = x.view(-1, 4, 4)[:, :3, 3]
            reconR = recon.view(-1, 4, 4)[:, :3, :3] 
            reconp = recon.view(-1, 4, 4)[:, :3, 3] 
            mse_r = 0.5*(log_SO3(xR.permute(0,2,1)@reconR)**2).sum(dim=-1).sum(dim=-1)
            mse_p = ((xp-reconp)**2).sum(dim=1)
            return 10*mse_r.view(bs, -1).mean(dim=1), mse_p.view(bs, -1).mean(dim=1)
        else:
            return torch.zeros(1), ((x-recon)**2).mean(dim=-1)
        
    def smoothness_loss(self, z, eta=0.4):
        bs = len(z)
        z_perm = z[torch.randperm(bs)]
        if eta is not None:
            alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
            z_augmented = alpha*z + (1-alpha)*z_perm
        else:
            z_augmented = z
        x = self.decode(z_augmented)
        if self.type == 'SE3':
            xdot = x[:, 1:, :, :] - x[:, :-1, :, :]
            xdot = xdot.view(len(x), -1, 16)
            energy = ((xdot)**2).sum(dim=-1).mean(dim=-1)
        else:
            bs, traj_len, x_dim = x.size()
            xdot = x[:, 1:] - x[:, :-1]
            xdot = xdot.view(bs, -1, x_dim)
            energy = ((xdot)**2).sum(dim=-1).mean(dim=-1)
        return energy
     
    def train_step(self, x, optimizer=None, **kwargs):
        optimizer.zero_grad()
        recon = self(x)
        mse_r, mse_p = self.compute_mse(x, recon)
        loss = mse_r.mean() + mse_p.mean()
        
        z = self.encode(x)
        energy = self.smoothness_loss(z)
        loss = loss + self.smoothness_weight * energy.mean()
        
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "mse_r_": mse_r.mean().item(), "mse_p_": mse_p.mean().item(), "energy_": energy.mean().item()}
    
    def validation_step(self, x, **kwargs):
        recon = self(x)
        mse_r, mse_p = self.compute_mse(x, recon)
        loss = mse_r.mean() + mse_p.mean()
        return {"loss": loss.item(), "rmse_r_": torch.sqrt(mse_r.mean()).item(), "rmse_p_": torch.sqrt(mse_p.mean()).item()}

    def eval_step(self, dl, **kwargs):
        pass
    
    def visualization_step(self, dl, **kwargs):
        device = kwargs["device"]
        for data in dl:
            x = data[0]
            break
        skip_size = 5
        recon = self(x.to(device))
        fig = visualize_SE3(
            x.detach().cpu()[:, 0: -1: skip_size], 
            recon.detach().cpu()[:, 0: -1: skip_size]
        )
        return {"SE3traj_data_and_recon#": fig}
 
class NRMMP(MMP):
    def __init__(self, encoder, decoder, approx_order=1, kernel=None):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.approx_order = approx_order
        self.kernel_func = get_kernel_function(kernel)
    
    def jacobian(self, z, dz, create_graph=True):
        batch_size = dz.size(0)
        num_nn = dz.size(1)
        z_dim = dz.size(2)

        v = dz.view(-1, z_dim)  # (bs * num_nn , z_dim)
        inputs = (z.unsqueeze(1).repeat(1, num_nn, 1).view(-1, z_dim))  # (bs * num_nn , z_dim)
        jac = torch.autograd.functional.jvp(self.decoder, inputs, v=v, create_graph=create_graph)[1].view(batch_size, num_nn, -1)
        return jac        

    def jacobian_and_hessian(self, z, dz, create_graph=True):
        batch_size = dz.size(0)
        num_nn = dz.size(1)
        z_dim = dz.size(2)

        v = dz.view(-1, z_dim)  # (bs * num_nn , z_dim)
        inputs = (z.unsqueeze(1).repeat(1, num_nn, 1).view(-1, z_dim))  # (bs * num_nn , z_dim)

        def jac_temp(inputs):
            jac = torch.autograd.functional.jvp(self.decoder, inputs, v=v, create_graph=create_graph)[1].view(batch_size, num_nn, -1)
            return jac

        temp = torch.autograd.functional.jvp(jac_temp, inputs, v=v, create_graph=create_graph)

        jac = temp[0].view(batch_size, num_nn, -1)
        hessian = temp[1].view(batch_size, num_nn, -1)
        return jac, hessian
        
    def neighborhood_recon(self, z_c, z_nn):
        recon = self.decoder(z_c)
        recon_x = recon.view(z_c.size(0), -1).unsqueeze(1)  # (bs, 1, x_dim)
        dz = z_nn - z_c.unsqueeze(1)  # (bs, num_nn, z_dim)
        if self.approx_order == 1:
            Jdz = self.jacobian(z_c, dz)  # (bs, num_nn, x_dim)
            n_recon = recon_x + Jdz
        elif self.approx_order == 2:
            Jdz, dzHdz = self.jacobian_and_hessian(z_c, dz)
            n_recon = recon_x + Jdz + 0.5*dzHdz
        return n_recon

    def train_step(self, x_c, x_nn, optimizer, **kwargs):
        optimizer.zero_grad()
        bs = x_nn.size(0)
        num_nn = x_nn.size(1)

        z_c = self.encoder(x_c)
        z_dim = z_c.size(1)
        z_nn = self.encoder(x_nn.view([-1] + list(x_nn.size()[2:]))).view(bs, -1, z_dim)
        n_recon = self.neighborhood_recon(z_c, z_nn)
        n_loss = torch.norm(x_nn.view(bs, num_nn, -1) - n_recon, dim=2)**2
        weights = self.kernel_func(x_c, x_nn)
        loss = (weights*n_loss).mean()
        
        energy = self.smoothness_loss(z_c)
        loss = loss + self.smoothness_weight * energy.mean()
        
        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}