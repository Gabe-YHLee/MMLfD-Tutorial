import torch

from utils.LieGroup_torch import log_SO3, exp_so3, skew

def Gaussian_basis(z, b=30, h_mul=1):
    '''
    b: number of basis
    b^G:[0, 1] -> R^b
    b^G_i:[0, 1] -> R
    z: (bs, L, 1)
    '''
    c = torch.linspace(0, 1, b).to(z)
    h = h_mul*1/(b-1)
    values = torch.exp(-(z - c.view(1, 1, -1))**2/h**2) # (bs, L, b)
    return values # (bs, L, b)

def phi(basis_values):
    '''
    phi_i(z) = b_i(z)/sum(b_i(z))
    '''
    return basis_values/basis_values.sum(dim=-1, keepdim=True) # (bs, L, b)

def lbf(phi_values, weight):
    '''
    q(z;w) = phi(z)w  
    phi_values: (bs, L, b)
    weight: (bs, b, dim)
    '''
    return phi_values@weight # (bs, L, dim)

def vbf(z, phi_values, weight, **kwargs):
    '''
    q(z;w) = h(z) + z(1-z)phi(z)w  
    phi_values: (bs, L, b)
    weight: (bs, b, dim)
    z: (bs, L, 1)
    '''
    init = torch.tensor(kwargs['via_points'][0], dtype=torch.float32).to(phi_values).view(1, 1, -1) # (1, 1, dim)
    final = torch.tensor(kwargs['via_points'][-1], dtype=torch.float32).to(phi_values).view(1, 1, -1) # (1, 1, dim)
    return z*(final) + (1-z)*(init) + (z)*(1-z)*phi_values@weight

def LfD(trajs, mode='promp', basis='Gaussian', b=30, h_mul=1, **kwargs):
    '''
    trajs: (bs, L, dim)
    weight: (bs, b, dim)
    '''
    
    bs, L, dim = trajs.size()
    z = torch.linspace(0, 1, L).view(1, -1, 1).to(trajs) # (1, L, 1)
    if basis == 'Gaussian':
        basis_values = Gaussian_basis(z, b=b, h_mul=h_mul) # (1, L, b)
    else:
        raise NotImplementedError
    Phi = phi(basis_values).view(1, L, -1) # (1, L, b)
    if mode == 'promp':
        return torch.linalg.pinv(Phi)@trajs # (bs, b, dim)
    elif mode == 'vmp':
        init = torch.tensor(kwargs['via_points'][0], dtype=torch.float32).to(trajs).view(1, 1, dim) # (1, 1, dim)
        final = torch.tensor(kwargs['via_points'][-1], dtype=torch.float32).to(trajs).view(1, 1, dim) # (1, 1, dim)
        Phi = z*(1-z)*Phi
        return torch.linalg.pinv(Phi)@(trajs - z*(final) - (1-z)*(init))

def SE3LfD(trajs, basis='Gaussian', b=30, h_mul=1, **kwargs):
    '''
    trajs: (bs, L, 4, 4)
    weight: (bs, b, 6)
    '''
    bs, L, _, _ = trajs.size()
    z = torch.linspace(0, 1, L).view(1, -1, 1).to(trajs) # (1, L, 1)
    if basis == 'Gaussian':
        basis_values = Gaussian_basis(z, b=b, h_mul=h_mul) # (1, L, b)
    else:
        raise NotImplementedError
    Phi = phi(basis_values).view(1, L, -1) # (1, L, b)
    
    init = trajs[:, 0]
    init_R = init[:, :3, :3]
    init_p = init[:, :3, 3].view(-1, 1, 3)
    final = trajs[:, -1]
    final_R = final[:, :3, :3]
    final_p = final[:, :3, 3].view(-1, 1, 3)
    init_logR = skew(log_SO3(init_R)).view(-1, 1, 3)
    final_logR = skew(log_SO3(final_R)).view(-1, 1, 3)
    
    Phi = z*(1-z)*Phi # (1, L, b)
    w_p = torch.linalg.pinv(Phi[:, 1:-1, :])@(trajs[:, 1:-1, :3, 3] - z[:, 1:-1, :]*(final_p) - (1-z[:, 1:-1, :])*(init_p)) # bs, b, 3
    
    R_trajs = trajs[:, :, :3, :3]
    log_nominalSO3 = log_SO3(
        init_R.permute(0, 2, 1)@final_R
    ).view(bs, 1, 3, 3)*z.view(1, -1, 1, 1)
    R_nominal = init_R.view(
        -1, 1, 3, 3)@exp_so3(
            log_nominalSO3.view(-1, 3, 3)).view(bs, L, 3, 3) # (bs, L, 3, 3)
    
    delta_R = R_nominal.permute(0, 1, 3, 2)@R_trajs # (bs, L, 3, 3)
    log_delta_R = skew(log_SO3(delta_R.view(-1, 3, 3))).view(bs, L, 3)
   
    w_R = torch.linalg.pinv(Phi[:, 1:-1, :])@log_delta_R[:, 1:-1, :]
    w = torch.cat([w_R, w_p], dim=-1) # (bs, b, 6)
    params = torch.cat([
        torch.cat([init_logR, init_p], dim=-1),
        torch.cat([final_logR, final_p], dim=-1),
        w], dim=1)
    return params[:, 1:, :] # (bs, b, 6)
    
    