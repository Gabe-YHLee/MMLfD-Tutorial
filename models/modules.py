import torch
import torch.nn as nn
from utils import LieGroup_torch as lie

def get_activation(s_act):
    if s_act == "relu":
        return nn.ReLU(inplace=True)
    elif s_act == "sigmoid":
        return nn.Sigmoid()
    elif s_act == "softplus":
        return nn.Softplus()
    elif s_act == "linear":
        return None
    elif s_act == "tanh":
        return nn.Tanh()
    elif s_act == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == "softmax":
        return nn.Softmax(dim=1)
    elif s_act == "selu":
        return nn.SELU()
    elif s_act == "elu":
        return nn.ELU()
    elif s_act == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unexpected activation: {s_act}")

class FC_vec(nn.Module):
    def __init__(
        self,
        in_chan=480*12,
        out_chan=2,
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(FC_vec, self).__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        return self.net(x)

class FC_traj2latent(FC_vec):
    def __init__(
        self,
        in_chan=201*2,
        out_chan=2,
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(FC_traj2latent, self).__init__(
            in_chan=in_chan, 
            out_chan=out_chan,
            l_hidden=l_hidden, 
            activation=activation,
            out_activation=out_activation)

    def forward(self, x):
        x = x.view(len(x), -1)
        return self.net(x)

class FC_latent2traj(FC_vec):
    def __init__(
        self,
        in_chan=2,
        out_chan=201*2,
        l_hidden=None,
        activation=None,
        out_activation=None,
        x_dim=2
    ):
        super(FC_latent2traj, self).__init__(
            in_chan=in_chan, 
            out_chan=out_chan,
            l_hidden=l_hidden, 
            activation=activation,
            out_activation=out_activation)
        self.x_dim = x_dim
        
    def forward(self, x):
        return self.net(x).view(len(x), -1, self.x_dim)
    
class FC_SE32vec(FC_vec):
    def __init__(
        self,
        in_chan=480*12,
        out_chan=2,
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(FC_SE32vec, self).__init__(
            in_chan=in_chan, 
            out_chan=out_chan, 
            l_hidden=l_hidden, 
            activation=activation,
            out_activation=out_activation)

    def forward(self, x):
        """_summary_

        Args:
            x (torch.tensor): (bs, len, 4, 4)

        Returns:
            torch.tensor: (bs, z_dim)
        """
        x = x[:, :, :3, :]
        x = x.reshape(len(x), -1)
        return self.net(x)
    
class FC_vec2SE3(FC_vec):
    def __init__(
        self,
        in_chan=2,
        out_chan=480*6,
        l_hidden=None,
        activation=None,
        out_activation='linear',
    ):
        super(FC_vec2SE3, self).__init__(
            in_chan=in_chan, 
            out_chan=out_chan, 
            l_hidden=l_hidden, 
            activation=activation,
            out_activation=out_activation)

    def forward(self, x):
        """_summary_

        Args:
            x (torch.tensor): (bs, z_dim)

        Returns:
            torch.tensor: (bs, len, 4, 4)
        """
        bs = len(x)
        y_local = self.net(x).view(bs, -1, 6).view(-1, 6)
        R = lie.exp_so3(y_local[:, :3]).reshape(-1, 3, 3)
        p = (y_local[:, 3:]).reshape(-1, 3, 1)
        y_global = torch.cat([
            torch.cat([R, p], dim=2),
            torch.tensor([0.,0.,0.,1]).view(1, 1, 4).repeat(len(y_local), 1, 1).to(x)
            ], dim=1).view(bs, -1, 4, 4) 
        return y_global

class vf_FC_vec(nn.Module):
    def __init__(
        self,
        in_chan=768+1+1,
        out_chan=1,
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(vf_FC_vec, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, t, z, q):
        return self.net(torch.cat([t, z, q], dim=1))
    
class vf_FC_se3(nn.Module):
    def __init__(
        self,
        in_chan=480*4*4+3+3,
        out_chan=6,
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(vf_FC_se3, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, t, x):
        # x : bs, len, 4, 4
        bs, traj_len, _, _ = x.size()
        x = x.view(bs, -1)
        return self.net(torch.cat([t, x], dim=1)).view(bs, traj_len, 6)    
 