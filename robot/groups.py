from re import T
import numpy as np
import torch.nn as nn
import torch
from utils import LieGroup_torch as lie

class BaseGroup():
    def __init__(self) -> None:
        pass

    def get_h_bar(self, tau):
        # return: h_bar
        raise NotImplementedError
    
    def _check_valid(self, tol=1e-4):
        for i in range(50):
            h = self._random_h()
            h_inv = self.get_inv(h)
            tau = self._random_task()
            traj = self._random_traj()
            htau, htraj = self.action_traj(h, tau, traj)

            hbar_1 = self.get_inv(self.get_h_bar(tau))
            hbar_2 = self.get_inv(self.get_h_bar(htau))

            tau0_1, traj0_1 = self.action_traj(hbar_1, tau, traj)
            tau0_2, traj0_2 = self.action_traj(hbar_2, htau, htraj)
            tau0_error = torch.norm(tau0_1 - tau0_2)
            traj0_error = torch.norm(traj0_1 - traj0_2, dim=1).mean()
            h_inv_error = torch.norm(tau - self.action_task(h_inv, self.action_task(h, tau)))
            if tau0_error <= tol and traj0_error <= tol and h_inv_error <= tol:
                continue
            else:
                print(f'iteration = {i}')
                print('tau0_delta:')
                print(tau0_error)
                print('traj0_delta:')
                print(traj0_error)
                print('h_inv_error')
                print(h_inv_error)
                return False
        return True

    def _random_h(self, batchsize=10):
        raise NotImplementedError
    
    def _random_task(self, batchsize=10):
        raise NotImplementedError

    def _random_traj(self, batchsize=10):
        raise NotImplementedError

    def get_inv(self, h):
        raise NotImplementedError

    def action_task(self, h, tau):
        raise NotImplementedError

    def action_traj(self, h, tau, traj):
        raise NotImplementedError

    def _random_data_aug(self, tau, traj=None, n_aug=1):
        n_data = len(tau)
        tau_aug_list = []
        traj_aug_list = []
        if traj is not None:
            original_shape = traj.shape
        if n_aug > 0:
            for i in range(n_aug):
                h = self._random_h(batch_size=n_data).to(tau)
                if traj is not None:
                    htau, htraj = self.action_traj(h, tau, traj)
                    traj_aug_list.append(htraj)
                else:
                    htau = self.action_task(h, tau)
                tau_aug_list.append(htau)
            tau_aug = torch.cat(tau_aug_list, dim=0)
            if traj is not None:
                traj_aug = torch.cat(traj_aug_list, dim=0)
        else:
            tau_aug = tau
            traj_aug = traj
        if traj is not None:
            return tau_aug, traj_aug.reshape(-1, *original_shape[1:])
        else:
            return tau_aug
    
class PouringGroup(BaseGroup):
    def __init__(self) -> None:
        super().__init__()
        # tau: a torch tensor (x_cup, y_cup, x_bottle, y_bottle, water_weight, \theta_i) (p_i, theta oo)
        # h = ([x, y], theta1(rotation around the cup),
        #       theta2(rotation around the z-axis of the end-effector))
        # traj: SE(3)^N_t
        self.tau_dim_squeezed = 2
        self.tau_dim_local = 6
        self.tau_dim_extended = 7
        self.bottle_height_bottom = 0.2
    
    def _random_h(self, batch_size=10):
        random_x = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_y = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_theta1_cup = torch.rand(batch_size, 1) * 2 * np.pi
        random_theta2_bottle = torch.rand(batch_size, 1) * 2 * np.pi
        random_h = torch.cat([random_x, random_y, random_theta1_cup, random_theta2_bottle], dim=1)
        return random_h
    
    def _random_task(self, batch_size=10):
        random_x_cup = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_y_cup = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_x_i = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_y_i = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_h = torch.rand(batch_size, 1) * 0.21 + 0.2
        random_theta_i= torch.rand(batch_size, 1) * 2 * torch.pi
        random_tau = torch.cat([
            random_x_cup, random_y_cup, random_x_i, random_y_i, random_h, random_theta_i
        ], dim=1)
        return random_tau

    def _random_traj(self, batch_size = 10):
        S = (torch.rand(batch_size, 100, 6)*10) - 5
        Slong = S.reshape(-1, 6)
        Xlong = lie.exp_se3(Slong)
        X = Xlong.reshape(batch_size, 100, 4, 4)
        return X
    
    # squeeze_hat_w and unsqueeze_hat_w : for reduced task parameter learning of equivariant_TCVAE
    def squeeze_hat_h(self, hat_h):
        return hat_h[:, [2, 4]]
    
    def unsqueeze_hat_tau(self, squeezed_hat_tau):
        batch_size = len(squeezed_hat_tau)
        temp = torch.zeros((batch_size, 1))
        return torch.cat([temp, temp, squeezed_hat_tau[:, 0], temp, squeezed_hat_tau[:, 1], temp], dim=1)
    
    # extended_w and unextended_w : for extended task parameter learning of TCVAE
    def extend_tau(self, tau):
        theta = tau[:, 5]
        cos_theta = torch.cos(theta).unsqueeze(1); sin_theta = torch.sin(theta).unsqueeze(1)

        return torch.cat((tau[:, :5], cos_theta, sin_theta), dim=1)
    
    def unextend_tau(self, extended_tau):
        theta = torch.atan2(extended_tau[:, 6], extended_tau[:, 5]).unsqueeze(1)
        
        return torch.cat((extended_tau[:, :5], theta), dim=1)
    
    def get_h_bar_inv(self, tau):
        batch_size = len(tau)
        h = torch.zeros(batch_size, 4).to(tau)
        h[:, :2] = -tau[:, :2]
        d_xy = tau[:, 2:4] - tau[:, :2]
        theta_1 = torch.atan2(d_xy[:, 1], d_xy[:, 0])
        theta_2 = tau[:, -1]
        h[:, 2] = -theta_1
        h[:, 3] = -theta_2 + theta_1
        return h
        
    def get_h_bar(self, tau):
        batch_size = len(tau)
        h = torch.zeros(batch_size, 4).to(tau)
        h[:, :2] = tau[:, :2]
        d_xy = tau[:, 2:4] - tau[:, :2]
        theta_1 = torch.atan2(d_xy[:, 1], d_xy[:, 0])
        theta_2 = tau[:, -1]
        h[:, 2] = theta_1
        h[:, 3] = theta_2 - theta_1
        return h
        
    def get_inv(self, h):
        return -h
    
    def rot_z(self, theta, SE3=True):
        batch_size = len(theta)
        tau = torch.zeros(batch_size, 3).to(theta)
        tau[:, 2] = theta
        R = lie.exp_so3(tau)
        if SE3 == False:
            return R
        else:
            T = torch.cat(
                [torch.cat([R, torch.zeros(batch_size, 3, 1).to(theta)], dim=2), 
                torch.zeros(batch_size, 1, 4).to(theta)
                ], dim=1)
            T[:, -1, -1] = 1
            return T

    def action_task(self, h, tau):
        batch_size = len(h)
        htau = torch.zeros_like(tau)
        htau[:, :2] = tau[:, :2] + h[:, :2]
        theta_1 = h[:, 2]
        theta_2 = h[:, 3]
        d_x = tau[:, 2] - tau[:, 0]
        d_y = tau[:, 3] - tau[:, 1]
        c1 = torch.cos(theta_1)
        s1 = torch.sin(theta_1)
        htau[:, 2] = tau[:, 0] + h[:, 0] + (c1 * d_x - s1 * d_y)
        htau[:, 3] = tau[:, 1] + h[:, 1] + (s1 * d_x + c1 * d_y)
        htau[:, 4] = tau[:, 4]
        htau[:, 5] = tau[:, 5] + theta_1 + theta_2
        return htau

    def action_traj(self, h, tau, traj):
        original_shape = traj.shape
        dim12 = False
        if len(original_shape) == 2:
            dim12 = True
            time_step = int(traj.shape[-1] / 12)
        else:
            time_step = traj.shape[1]
        batch_size = len(h)
        traj = traj.reshape(batch_size, time_step, -1, 4)
        if traj.size()[2] == 3:
            aug = torch.zeros((batch_size, time_step, 1, 4)).to(traj)
            aug[:, :, :, 3] = 1
            traj = torch.cat((traj, aug), dim=2)
            dim12 = True
        htau = self.action_task(h, tau)
        theta_1 = h[:, 2]
        theta_2 = h[:, 3]
        S = torch.zeros(batch_size, 6).to(traj)
        S[:, 2] = 1
        S[:, 3] = tau[:, 1]
        S[:, 4] = -tau[:, 0]
        T_screw = lie.exp_se3(S * theta_1.unsqueeze(1))
        traj_after_screw_motion = torch.einsum('ijk, ihkl -> ihjl', T_screw, traj)
        htraj = torch.einsum('ijkl, ilh -> ijkh', traj_after_screw_motion, self.rot_z(theta_2))
        # gtraj = T_screw.unsqueeze(1) @ traj @ self.rot_z(theta_2)
        htraj[:, :, 0, -1] += h[:, 0].unsqueeze(1)
        htraj[:, :, 1, -1] += h[:, 1].unsqueeze(1)
        if dim12:
            htraj = htraj[:, :, :3, :].reshape(*original_shape)
        return htau, htraj

################################################################################################
######################################### Toy 2D ###############################################
################################################################################################
class PlanarMobileRobot(BaseGroup):
    def __init__(self) -> None:
        super().__init__()
        # tau: a torch tensor (x_robot, y_robot, wall_rotation_angle)
        # H = {0, 1, 2 ,3} x {0, 1} x SO(2),
        # h = (rot, mirror, \theta_wall_robot)
        # traj: (R^2)^n
        self.R = torch.zeros(4, 2, 2)
        self.R[0, 0, 0] = 1
        self.R[0, 1, 1] = 1
        self.R[1, 0, 1] = -1
        self.R[1, 1, 0] = 1
        self.R[2, 0, 0] = -1
        self.R[2, 1, 1] = -1
        self.R[3, 0, 1] = 1
        self.R[3, 1, 0] = -1
        self.h_dim_squeezed = 2
        self.tau_dim_local = 3
        self.tau_dim_extended = 4



    def get_wall(self, tau):
        low = 1.2
        high = 4   
        # tau = (n, 3)
        wall1 = torch.tensor([[low, low, high], [high, low, low]])
        wall2 = torch.tensor([[low, low, high], [-high, -low, -low]])
        wall3 = torch.tensor([[-low, -low, -high], [high, low, low]])
        wall4 = torch.tensor([[-low, -low, -high], [-high, -low, -low]])
        walls_original = torch.cat([wall1, wall2, wall3, wall4], dim=1) # (2, 12)
        Rot_mat = self.rot_z(tau[:, -1]).cpu()  # (n, 2, 2)
        walls_rot = Rot_mat @ walls_original
        wall1_rot = walls_rot[:, :, :3]
        wall2_rot = walls_rot[:, :, 3:6]
        wall3_rot = walls_rot[:, :, 6:9]
        wall4_rot = walls_rot[:, :, 9:]
        return wall1_rot, wall2_rot, wall3_rot, wall4_rot
    
    def get_wrong_entrance(self, tau):
        low = 2.0
        # tau = (n, 3)
        ent1 = torch.tensor([[low, low], [-low, low]])
        ent2 = torch.tensor([[-low, low], [low, low]])
        ent3 = torch.tensor([[-low, -low], [-low, low]])
        ent4 = torch.tensor([[-low, low], [-low, -low]])
        ents_original = torch.cat([ent1, ent2, ent3, ent4], dim=1)
        Rot_mat = self.rot_z(tau[:, -1]).cpu()  # (n, 2, 2)
        ents_rot = Rot_mat @ ents_original
        ent1_rot = ents_rot[:, :, :2] # batch, 2
        ent2_rot = ents_rot[:, :, 2:4] 
        ent3_rot = ents_rot[:, :, 4:6]
        ent4_rot = ents_rot[:, :, 6:]
        
        init_points = tau[:, :2].cpu()
        init_points_no_rot = (Rot_mat.transpose(1, 2) @ init_points.unsqueeze(-1)).squeeze(-1)
        theta_init_points = torch.atan2(init_points_no_rot[:, 1], init_points_no_rot[:, 0])
        theta_1 = (theta_init_points >= 0) * (theta_init_points < np.pi/2)
        theta_2 = (theta_init_points >= np.pi/2) * (theta_init_points < np.pi)
        theta_3 = (theta_init_points >= -np.pi) * (theta_init_points < -np.pi/2)
        theta_4 = (theta_init_points >= -np.pi/2) * (theta_init_points < 0)
        
        ent_selected_1 = torch.zeros_like(ent1_rot)
        ent_selected_2 = torch.zeros_like(ent1_rot)
        ent_selected_1[theta_1] = ent3_rot[theta_1]
        ent_selected_2[theta_1] = ent4_rot[theta_1]
        ent_selected_1[theta_2] = ent1_rot[theta_2]
        ent_selected_2[theta_2] = ent4_rot[theta_2]
        ent_selected_1[theta_3] = ent1_rot[theta_3]
        ent_selected_2[theta_3] = ent2_rot[theta_3]
        ent_selected_1[theta_4] = ent2_rot[theta_4]
        ent_selected_2[theta_4] = ent3_rot[theta_4]
        return ent_selected_1, ent_selected_2
        
    def _random_h(self, batch_size=10):
        random_rot = torch.randint(0, 4, (batch_size, 1))
        random_mirror = torch.randint(0, 2, (batch_size, 1))
        random_rot_wall = torch.rand((batch_size, 1)) * 0.5 * torch.pi - 0.25 * torch.pi
        random_h = torch.cat([random_rot, random_mirror, random_rot_wall], dim=1)
        return random_h
    
    
    def _random_task(self, batch_size=10):
        random_phi = torch.rand(batch_size, 1) * 2 * torch.pi
        random_r = torch.sqrt(torch.rand(batch_size, 1)) * 5 + 5
        random_x = torch.cos(random_phi) * random_r
        random_y = torch.sin(random_phi) * random_r
        random_theta = torch.rand((batch_size, 1)) * 0.5 * torch.pi - 0.25 * torch.pi
        random_tau = torch.cat([
            random_x, random_y, random_theta
        ], dim=1)
        return random_tau

    def _random_traj(self, batch_size=10):
        traj = (torch.rand(batch_size, 100, 2)*10)
        return traj
    
    
    # squeeze_hat_w and unsqueeze_hat_w : for reduced task parameter learning of equivariant_TCVAE
    def squeeze_hat_tau(self, hat_tau):
        return hat_tau[:, :2]
    
    def unsqueeze_hat_tau(self, squeezed_hat_tau):
        batch_size = len(squeezed_hat_tau)
        temp = torch.zeros((batch_size, 1)).to(squeezed_hat_tau)
        return torch.cat((squeezed_hat_tau, temp), dim=1)
    
    # extended_w and unextended_w : for extended task parameter learning of TCVAE
    def extend_tau(self, tau):
        if tau.shape[-1] == 2:
            tau = self.unsqueeze_hat_tau(tau)
        theta = tau[:, 2]
        cos_theta = torch.cos(theta).unsqueeze(1)
        sin_theta = torch.sin(theta).unsqueeze(1)
        return torch.cat((tau[:, :2], cos_theta, sin_theta), dim=1)
    
    def unextend_tau(self, extended_tau):
        theta = torch.atan2(extended_tau[:, 3], extended_tau[:, 2]).unsqueeze(1)
        return torch.cat((extended_tau[:, :2], theta), dim=1)
    
    def get_h_bar_inv(self, tau):
        tau_theta = tau[:, -1]
        R_temp = self.rot_z(-tau_theta)
        h_rot_wall = -tau_theta
        xy_temp = (R_temp @ tau[:, :2].unsqueeze(-1)).squeeze(-1)
        
        batch_size = len(tau)
        h_rot = torch.zeros(batch_size).to(tau)
        h_mirror = torch.zeros(batch_size).to(tau)
        theta = torch.atan2(xy_temp[:, 1], xy_temp[:, 0])
        pi4 = torch.pi/4
        h_rot[(theta >= -pi4) * (theta < pi4)] = 0
        h_rot[(theta >= pi4) * (theta < 2 * pi4) + (theta >= -2 * pi4) * (theta < -pi4)] = 1
        h_rot[(theta >= 3* pi4) * (theta <= 4 * pi4) + (theta >= -4 * pi4) * (theta < -3 * pi4)] = 2
        h_rot[(theta >= 2* pi4) * (theta < 3 * pi4) + (theta >= -3 * pi4) * (theta < -2 * pi4)] = 3
        
        h_mirror[(theta >= 0) * (theta < pi4) + (theta >= 2 * pi4) * (theta < 3 * pi4) + 
           (theta >= -4 * pi4) * (theta < -3 * pi4) + (theta >= -2 * pi4) * (theta < -pi4)] = 0
        h_mirror[(theta >= -pi4) * (theta < 0) + (theta >= pi4) * (theta < 2 * pi4) + 
           (theta >= 3 * pi4) * (theta <= 4 * pi4) + (theta >= -3 * pi4) * (theta < -2 * pi4)] = 1
        h = torch.cat([h_rot.unsqueeze(1), h_mirror.unsqueeze(1), h_rot_wall.unsqueeze(1)], dim=1)
        return h

    def get_h_bar(self, tau):
        tau_theta = tau[:, -1]
        R_temp = self.rot_z(-tau_theta)
        h_rot_wall = -tau_theta
        xy_temp = (R_temp @ tau[:, :2].unsqueeze(-1)).squeeze(-1)
        
        batch_size = len(tau)
        h_rot = torch.zeros(batch_size).to(tau)
        h_mirror = torch.zeros(batch_size).to(tau)
        theta = torch.atan2(xy_temp[:, 1], xy_temp[:, 0])
        pi4 = torch.pi/4
        h_rot[(theta >= -pi4) * (theta < pi4)] = 0
        h_rot[(theta >= pi4) * (theta < 3 * pi4)] = 1
        h_rot[(theta >= 3* pi4) * (theta <= 4 * pi4) + 
              (theta >= -4 * pi4) * (theta < -3 * pi4)] = 2
        h_rot[(theta >= -3 * pi4) * (theta < -pi4)] = 3
        
        h_mirror[(theta >= 0) * (theta < pi4) + (theta >= 2 * pi4) * (theta < 3 * pi4) + 
           (theta >= -4 * pi4) * (theta < -3 * pi4) + (theta >= -2 * pi4) * (theta < -pi4)] = 0
        h_mirror[(theta >= -pi4) * (theta < 0) + (theta >= pi4) * (theta < 2 * pi4) + 
           (theta >= 3 * pi4) * (theta <= 4 * pi4) + (theta >= -3 * pi4) * (theta < -2 * pi4)] = 1
        h = torch.cat([h_rot.unsqueeze(1), h_mirror.unsqueeze(1), h_rot_wall.unsqueeze(1)], dim=1)
        return h

    
    def get_inv(self, h):
        hinv = h.clone()
        hinv[h[:, 1] == 0, 0] =(-h[h[:, 1] == 0, 0]) % 4
        hinv[:, 2] = -hinv[:, 2]
        return hinv
    
    def rot_z(self, theta):
        ct = torch.cos(theta).unsqueeze(-1).unsqueeze(-1)
        st = torch.sin(theta).unsqueeze(-1).unsqueeze(-1)
        R = torch.cat([
            torch.cat([ct, -st], dim=2),
            torch.cat([st, ct], dim=2)],
            dim=1)
        return R
    
    def action_task(self, h, tau):
        # mirroring first, rotation later
        self.R = self.R.to(tau)
        batch_size = len(h)
        h_rot = h[:, 0]
        h_mirror = h[:, 1]
        tau_theta = tau[:, -1]
        R_temp = self.rot_z(-tau_theta)
        xy_temp = (R_temp @ tau[:, :2].unsqueeze(-1)).squeeze(-1)
        
        
        xy_temp[h_mirror == 1, 1] = -xy_temp[h_mirror == 1, 1]
        xy_temp[h_rot == 1] = torch.einsum('ij, kj -> ki', self.R[1], xy_temp[h_rot == 1])
        xy_temp[h_rot == 2] = torch.einsum('ij, kj -> ki', self.R[2], xy_temp[h_rot == 2])
        xy_temp[h_rot == 3] = torch.einsum('ij, kj -> ki', self.R[3], xy_temp[h_rot == 3])
        htheta = (tau_theta + h[:, 2]) % (2*torch.pi)
        htheta[htheta > torch.pi] -= (2*torch.pi)
        R_temp2 = self.rot_z(htheta)
        hxy = (R_temp2 @ xy_temp.unsqueeze(-1)).squeeze(-1)
        htau = torch.cat([hxy, htheta.unsqueeze(-1)], dim=1)
        return htau

    def action_traj(self, h, tau, traj):
        h = h.to(traj)
        tau = tau.to(traj)
        batch_size = len(h)
        traj = traj.reshape(batch_size, -1, 2)
        htau = self.action_task(h, tau)
        
        htraj = traj.clone()
        tau_theta = tau[:, -1]
        R_temp = self.rot_z(-tau_theta)
        htraj = torch.einsum('nij, nkj -> nki', R_temp, htraj)
        h_rot = h[:, 0]
        h_mirror = h[:, 1]
        htraj[h_mirror == 1, :, -1] = -htraj[h_mirror == 1, :, -1]
        htraj[h_rot == 1] = torch.einsum('ij, klj -> kli', self.R[1], htraj[h_rot == 1])
        htraj[h_rot == 2] = torch.einsum('ij, klj -> kli', self.R[2], htraj[h_rot == 2])
        htraj[h_rot == 3] = torch.einsum('ij, klj -> kli', self.R[3], htraj[h_rot == 3])
        htheta = (tau_theta + h[:, 2]) % (2*torch.pi)
        htheta[htheta > torch.pi] -= (2*torch.pi)
        R_temp2 = self.rot_z(htheta)
        htraj = torch.einsum('nij, nkj -> nki', R_temp2, htraj)
        return htau, htraj