import numpy as np
from numpy.linalg import inv
import torch
import matplotlib.pyplot as plt
# from scipy import interpolate
# from scipy import integrate
# from scipy.optimize import fsolve
# import scipy
# import scipy.stats
import math

from torch.autograd import grad

dtype = torch.float
device_c = torch.device("cpu")
# device = torch.device("cuda:0")


def skew(w):
    n = w.shape[0]
    if w.shape == (n, 3, 3):
        W = torch.cat([-w[:, 1, 2].unsqueeze(-1),
                       w[:, 0, 2].unsqueeze(-1),
                       -w[:, 0, 1].unsqueeze(-1)], dim=1)
    else:
        zero1 = torch.zeros(n, 1, 1).to(w)
        w = w.unsqueeze(-1).unsqueeze(-1)
        W = torch.cat([torch.cat([zero1, -w[:, 2], w[:, 1]], dim=2),
                       torch.cat([w[:, 2], zero1, -w[:, 0]], dim=2),
                       torch.cat([-w[:, 1], w[:, 0], zero1], dim=2)], dim=1)
    return W


def skew_v2(w):
    n = w.shape[0]
    m = w.shape[1]
    if w.shape == (n, m, 3, 3):
        W = torch.cat([-w[:, :, 1, 2].unsqueeze(-1),
                       w[:, :, 0, 2].unsqueeze(-1),
                       -w[:, :, 0, 1].unsqueeze(-1)], dim=2)
    else:
        zero1 = torch.zeros(n, m, 1, 1).to(w)
        w = w.unsqueeze(-1).unsqueeze(-1)
        W = torch.cat([torch.cat([zero1, -w[:, :, 2], w[:, :, 1]], dim=3),
                       torch.cat([w[:, :, 2], zero1, -w[:, :, 0]], dim=3),
                       torch.cat([-w[:, :, 1], w[:, :, 0], zero1], dim=3)], dim=2)
    return W


def screw_bracket(V):
    if isinstance(V, str):
        # print(V)
        return 'trace error'
    n = V.shape[0]
    out = 0
    if V.shape == (n, 4, 4):
        out = torch.cat([-V[:, 1, 2].unsqueeze(-1), V[:, 0, 2].unsqueeze(-1), -V[:, 0, 1].unsqueeze(-1), V[:, :3, 3]],
                        dim=1)
    else:
        W = skew(V[:, 0:3])
        out = torch.cat([torch.cat([W, V[:, 3:].unsqueeze(-1)], dim=2), torch.zeros(n, 1, 4).to(V)], dim=1)
        print(torch.cat([W, V[:, 0:3].unsqueeze(-1)], dim=2))
    return out


def exp_so3(Input):
    n = Input.shape[0]
    if Input.shape == (n, 3, 3):
        W = Input
        w = skew(Input)
    else:
        w = Input
        W = skew(w)

    wnorm_sq = torch.sum(w * w, dim=1)
    wnorm_sq_unsqueezed = wnorm_sq.unsqueeze(-1).unsqueeze(-1)

    wnorm = torch.sqrt(wnorm_sq)
    wnorm_unsqueezed = torch.sqrt(wnorm_sq_unsqueezed)

    cw = torch.cos(wnorm).view(-1, 1).unsqueeze(-1)
    sw = torch.sin(wnorm).view(-1, 1).unsqueeze(-1)
    w0 = w[:, 0].unsqueeze(-1).unsqueeze(-1)
    w1 = w[:, 1].unsqueeze(-1).unsqueeze(-1)
    w2 = w[:, 2].unsqueeze(-1).unsqueeze(-1)
    eps = 1e-7

    R = torch.zeros(n, 3, 3).to(Input)

    R[wnorm > eps] = torch.cat((torch.cat((cw - ((w0 ** 2) * (cw - 1)) / wnorm_sq_unsqueezed,
                                           - (w2 * sw) / wnorm_unsqueezed - (w0 * w1 * (cw - 1)) / wnorm_sq_unsqueezed,
                                           (w1 * sw) / wnorm_unsqueezed - (w0 * w2 * (cw - 1)) / wnorm_sq_unsqueezed),
                                          dim=2),
                                torch.cat(((w2 * sw) / wnorm_unsqueezed - (w0 * w1 * (cw - 1)) / wnorm_sq_unsqueezed,
                                           cw - ((w1 ** 2) * (cw - 1)) / wnorm_sq_unsqueezed,
                                           - (w0 * sw) / wnorm_unsqueezed - (w1 * w2 * (cw - 1)) / wnorm_sq_unsqueezed),
                                          dim=2),
                                torch.cat((-(w1 * sw) / wnorm_unsqueezed - (w0 * w2 * (cw - 1)) / wnorm_sq_unsqueezed,
                                           (w0 * sw) / wnorm_unsqueezed - (w1 * w2 * (cw - 1)) / wnorm_sq_unsqueezed,
                                           cw - ((w2 ** 2) * (cw - 1)) / wnorm_sq_unsqueezed),
                                          dim=2)),
                               dim=1)[wnorm > eps]

    R[wnorm <= eps] = torch.eye(3).to(Input) + W[wnorm < eps] + 1 / 2 * W[wnorm < eps] @ W[wnorm < eps]
    return R


def Dexp_so3(w):
    R = exp_so3(w)
    N = w.shape[0]
    Id = torch.eye(3).to(w)
    dRdw = torch.zeros(N, 3, 3, 3).to(w)
    wnorm = torch.sqrt(torch.einsum('ni,ni->n', w, w))
    eps = 1e-5
    e_skew = skew(Id)
    if w.shape == (N, 3):
        W = skew(w)
    else:
        W = w
        w = skew(W)
        assert (False)
    temp1 = torch.einsum('ni,njk->nijk', w, W)
    temp2 = torch.einsum('njk,nki->nij', W, (-R + torch.eye(3).to(w)).to(w))
    temp2_2 = skew(temp2.reshape(N * 3, 3)).reshape(N, 3, 3, 3)
    wnorm_square = torch.einsum('ni,ni->n', w, w).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1, 1)
    dRdw[wnorm > eps] = (((temp1 + temp2_2) / wnorm_square) @ R.unsqueeze(1))[wnorm > eps]
    dRdw[wnorm < eps] = e_skew

    # dRdw is (N, 3, 3, 3) tensor. dRdw(n, i, :, :) is dR/dwi of n^th sample from the batch

    return dRdw


def D2exp_so3(w):
    R = exp_so3(w)
    N = w.shape[0]
    Id = torch.eye(3).to(w)
    e_skew = skew(Id)
    if w.shape == (N, 3):
        W = skew(w)
    else:
        W = w
        w = skew(W)
        assert (False)
    DR = Dexp_so3(w)

    wnorm_square = torch.einsum('ni,ni->n', w, w).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    temp1 = -2 * torch.einsum('ni,njkl->nijkl', w, DR) / wnorm_square
    temp2 = -torch.einsum('njkh,nihl->nijkl', DR, DR.transpose(-2, -1)) @ R.unsqueeze(1).unsqueeze(1)
    temp3 = ((torch.einsum('ij,nkl->nijkl', Id, W) + torch.einsum('nj,ikl->nijkl', w, e_skew))
             @ R.unsqueeze(1).unsqueeze(1)) / wnorm_square
    temp4_pre = (torch.einsum('ikl,nlj->nijk', e_skew, -R + Id)
                 + torch.einsum('nkl,nilj->nijk', W, -DR)).reshape(-1, 3)
    temp4 = ((skew(temp4_pre).reshape(N, 3, 3, 3, 3) @ R.unsqueeze(1).unsqueeze(1)) / wnorm_square)
    d2Rdw = temp1 + temp2 + temp3 + temp4
    # d2Rdw is (N, 3, 3, 3, 3) tensor. d2Rdw(n, i, j, :, :) is d(dR/dwj)/dwi of n^th sample from the batch

    return d2Rdw


def proj_minus_one_plus_one(x):
    eps = 1e-6
    x = torch.min(x, (1 - eps) * (torch.ones(x.shape).to(x)))
    x = torch.max(x, (-1 + eps) * (torch.ones(x.shape).to(x)))
    return x


def log_SO3(R):
    batch_size = R.shape[0]
    eps = 1e-4
    trace = torch.sum(R[:, range(3), range(3)], dim=1)

    omega = R * torch.zeros(R.shape).to(R)

    theta = torch.acos(proj_minus_one_plus_one((trace - 1) / 2))

    temp = theta.unsqueeze(-1).unsqueeze(-1)

    omega[(torch.abs(trace + 1) > eps) * (theta > eps)] = ((temp / (2 * torch.sin(temp))) * (R - R.transpose(1, 2)))[
        (torch.abs(trace + 1) > eps) * (theta > eps)]

    omega_temp = (R[torch.abs(trace + 1) <= eps] - torch.eye(3).to(R)) / 2

    omega_vector_temp = torch.sqrt(omega_temp[:, range(3), range(3)] + torch.ones(3).to(R))
    A = omega_vector_temp[:, 1] * torch.sign(omega_temp[:, 0, 1])
    B = omega_vector_temp[:, 2] * torch.sign(omega_temp[:, 0, 2])
    C = omega_vector_temp[:, 0]
    omega_vector = torch.cat([C.unsqueeze(1), A.unsqueeze(1), B.unsqueeze(1)], dim=1)
    omega[torch.abs(trace + 1) <= eps] = skew(omega_vector) * math.pi

    return omega


def log_SO3_v2(R):
    batch_size1 = R.shape[0]
    batch_size2 = R.shape[1]
    eps = 1e-7
    trace = torch.sum(R[:, :, range(3), range(3)], dim=2)

    omega = torch.zeros(batch_size1, batch_size2, 3, 3).to(R)
    theta = torch.acos((trace - 1) / 2)
    temp = theta.unsqueeze(-1).unsqueeze(-1)
    omega[(torch.abs(trace + 1) > eps) * (theta > eps)] = ((temp / (2 * torch.sin(temp))) * (R - R.transpose(2, 3)))[
        (torch.abs(trace + 1) > eps) * (theta > eps)]

    omega_temp = (R[torch.abs(trace + 1) <= eps] - torch.eye(3).to(R).unsqueeze(0).unsqueeze(0)) / 2
    omega_vector = torch.sqrt(
        omega_temp[:, :, range(3), range(3)] + torch.ones(3).to(R).unsqueeze(0).unsqueeze(0))
    omega_vector[:, :, 1] *= torch.sign(omega_temp[:, :, 0, 1])
    omega_vector[:, :, 2] *= torch.sign(omega_temp[:, :, 0, 2])

    omega[torch.abs(trace + 1) <= eps] = skew_v2(omega_vector) * math.pi

    return omega


def Riemannian_metric_so3(w):
    dRdw = Dexp_so3(w)
    G = torch.einsum('nikl,njkl->nij', dRdw, dRdw)
    return G


def DRiemannian_metric_so3(w):
    dRdw = Dexp_so3(w)
    d2Rdw = D2exp_so3(w)
    DG = torch.einsum('nkiab, njba -> nkij', d2Rdw, dRdw) + torch.einsum('niab, nkjba -> nkij', dRdw, d2Rdw)
    return DG


def Christoffel_symbol(w):
    G_inv = torch.inverse(Riemannian_metric_so3(w))
    DG = DRiemannian_metric_so3(w)
    Gamma = 0.5 * (torch.einsum('nir, nkrj -> nkij', G_inv, DG)
                   + torch.einsum('nir, njrk -> nkij', G_inv, DG)
                   - torch.einsum('nir, nrjk -> nkij', G_inv, DG))
    return Gamma


def exp_so3_T(S):
    n = S.shape[0]
    if S.shape == (n, 4, 4):
        S1 = skew(S[:, :3, :3]).clone()
        S2 = S[:, 0:3, 3].clone()
        S = torch.cat([S1, S2], dim=1)
    # shape(S) = (n,6,1)
    w = S[:, :3]  # dim= n,3
    v = S[:, 3:].unsqueeze(-1)  # dim= n,3

    eps = 1e-014
    W = skew(w)
    T = torch.cat([torch.cat([exp_so3(w), v], dim=2), (torch.zeros(n, 1, 4).to(S))], dim=1)
    T[:, -1, -1] = 1
    return T


def exp_se3(S, dim12=False):
    n = S.shape[0]
    if S.shape == (n, 4, 4):
        S1 = skew(S[:, :3, :3]).clone()
        S2 = S[:, 0:3, 3].clone()
        S = torch.cat([S1, S2], dim=1)
    # shape(S) = (n,6,1)
    w = S[:, :3]  # dim= n,3
    v = S[:, 3:].unsqueeze(-1)  # dim= n,3
    wsqr = torch.tensordot(w, w, dims=([1], [1]))[[range(n), range(n)]]  # dim = (n)
    wsqr_unsqueezed = wsqr.unsqueeze(-1).unsqueeze(-1)  # dim = (n,1,1)
    wnorm = torch.sqrt(wsqr)  # dim = (n)
    wnorm_unsqueezed = torch.sqrt(wsqr_unsqueezed)  # dim = (n,1,1)
    wnorm_inv = 1 / wnorm_unsqueezed  # dim = (n)
    cw = torch.cos(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1,1)
    sw = torch.sin(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1,1)

    eps = 1e-014
    W = skew(w)
    P = torch.eye(3).to(S) + (1 - cw) * (wnorm_inv ** 2) * W + (wnorm_unsqueezed - sw) * (
                wnorm_inv ** 3) * torch.matmul(W, W)  # n,3,3
    P[wnorm < eps] = torch.eye(3).to(S)
    T = torch.cat([torch.cat([exp_so3(w), P @ v], dim=2), (torch.zeros(n, 1, 4).to(S))], dim=1)
    T[:, -1, -1] = 1
    if dim12:
        T = (T[:, :3]).reshape(n, 12)
    return T


def inverse_SE3(T):
    n = T.shape[0]
    R = T[:, 0:3, 0:3]  # n,3,3
    p = T[:, 0:3, 3].unsqueeze(-1)  # n,3,1
    T_inv = torch.cat(
        [torch.cat([R.transpose(1, 2), (-R.transpose(1, 2)) @ p], dim=2), torch.zeros(n, 1, 4).to(T)], dim=1)
    T_inv[:, -1, -1] = 1
    # np.vstack((np.hstack(( np.transpose(R), np.reshape(-np.matmul(np.transpose(R),p),[3,1]) )),np.array([0,0,0,1])))
    return T_inv


def large_Ad(T):
    n = T.shape[0]
    R = T[:, 0:3, 0:3]  # n,3,3
    p = T[:, 0:3, 3]  # .unsqueeze(-1) #n,3
    AdT = torch.cat([torch.cat([R, torch.zeros(n, 3, 3).to(T)], dim=2), torch.cat([skew(p) @ R, R], dim=2)],
                    dim=1)
    # np.hstack((np.vstack((R,np.matmul(skew(p),R))), np.vstack((np.zeros([3,3]),R))))
    return AdT


def small_ad(V):
    # shape(V) = (n,6)
    n = V.shape[0]
    w = V[:, :3]
    v = V[:, 3:]
    wskew = skew(w)
    vskew = skew(v)
    adV = torch.cat([torch.cat([wskew, torch.zeros(n, 3, 3).to(V)], dim=2), torch.cat([vskew, wskew], dim=2)],
                    dim=1)
    # adV = np.hstack((np.vstack((skew(w),skew(v))), np.vstack((np.zeros([3,3]),skew(w)))))
    return adV


def log_SO3_T(T):
    # dim T = n,4,4
    R = T[:, 0:3, 0:3]  # dim n,3,3
    p = T[:, 0:3, 3].unsqueeze(-1)  # dim n,3,1
    n = T.shape[0]
    W = log_SO3(R)  # n,3,3

    return torch.cat([torch.cat([W, p], dim=2), torch.zeros(n, 1, 4).to(T)], dim=1)  # n,4,4


def log_SE3(T):
    # dim T = n,4,4
    R = T[:, 0:3, 0:3]  # dim n,3,3
    p = T[:, 0:3, 3].unsqueeze(-1)  # dim n,3,1
    n = T.shape[0]
    W = log_SO3(R)  # n,3,3
    # print(W)
    w = skew(W)  # n,3

    wsqr = torch.tensordot(w, w, dims=([1], [1]))[[range(n), range(n)]]  # dim = (n)
    wsqr_unsqueezed = wsqr.unsqueeze(-1).unsqueeze(-1)  # dim = (n,1)
    wnorm = torch.sqrt(wsqr)  # dim = (n)
    wnorm_unsqueezed = torch.sqrt(wsqr_unsqueezed)  # dim - (n,1)
    wnorm_inv = 1 / wnorm_unsqueezed  # dim = (n)
    cw = torch.cos(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1)
    sw = torch.sin(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1)

    P = torch.eye(3).to(W) + (1 - cw) * (wnorm_inv ** 2) * W + (wnorm_unsqueezed - sw) * (
                wnorm_inv ** 3) * torch.matmul(W, W)  # n,3,3
    v = torch.inverse(P) @ p  # n,3,1
    return torch.cat([torch.cat([W, v], dim=2), torch.zeros(n, 1, 4).to(W)], dim=1)


def Vb_to_qdot(T, Vb):
    R = T[:, :3, :3]
    Rt = R.transpose(1, 2)

    w = Vb[:, :3, :]
    v = R @ Vb[:, 3:, :]

    Dexp = Dexp_so3(skew(log_SO3(R)))
    Temp = torch.einsum('nij, nkjl -> nkil', Rt, Dexp)

    Dexp_qdot_w = skew(Temp.reshape(-1, 3, 3)).reshape(-1, 3, 3)

    qdot = (torch.cat([torch.inverse(Dexp_qdot_w.transpose(1, 2)) @ w, v], dim=1).squeeze(2))

    return qdot


def convert_SO3_to_quaternion(R):
    # dim(R) = n,3,3
    W = log_SO3(R)  # n,3,3
    w = skew(W)  # n,3
    theta_1dim = torch.sqrt(torch.sum(w ** 2, dim=1))
    theta = theta_1dim.unsqueeze(-1)  # n,1
    w_hat = w / theta  # n,3
    w_hat[theta_1dim < 1.0e-016] = 0
    return torch.cat([w_hat[:, 0].unsqueeze(-1) * torch.sin(theta / 2),
                      w_hat[:, 1].unsqueeze(-1) * torch.sin(theta / 2),
                      w_hat[:, 2].unsqueeze(-1) * torch.sin(theta / 2),
                      torch.cos(theta / 2)], dim=1)


# def Uniform_sampling(qtraj, batch_size):
#     qtraj_torch = qtraj
#     qtraj = qtraj.cpu().detach().numpy()
#     SO3_sampler = scipy.stats.special_ortho_group(dim=3)
#     random_R = torch.tensor(SO3_sampler.rvs(batch_size)).to(qtraj)

#     if isinstance(log_SO3(random_R), str):
#         random_q = Uniform_sampling(qtraj_torch, batch_size)
#         return random_q

#     random_w = skew(log_SO3(random_R))

#     qmin = np.min(qtraj[:, 3:], axis=0)
#     qmax = np.max(qtraj[:, 3:], axis=0)
#     qlength = np.linalg.norm(qmax - qmin)
#     min_offset = qlength / 1
#     max_offset = qlength / 1

#     random_p1 = torch.tensor(np.random.uniform(low=np.min(qtraj[:, 3]) - min_offset,
#                                                high=np.max(qtraj[:, 3]) + max_offset, size=[batch_size, 1])).to(qtraj)
#     random_p2 = torch.tensor(np.random.uniform(low=np.min(qtraj[:, 4]) - min_offset,
#                                                high=np.max(qtraj[:, 4]) + max_offset, size=[batch_size, 1])).to(qtraj)
#     random_p3 = torch.tensor(np.random.uniform(low=np.min(qtraj[:, 5]) - min_offset,
#                                                high=np.max(qtraj[:, 5]) + max_offset, size=[batch_size, 1])).to(qtraj)

#     random_q = torch.cat([random_w, random_p1, random_p2, random_p3], dim=1)
#     random_q.requires_grad = True

#     return random_q


def Gaussian_sampling(qtraj, w_std, p_std, batch_size=1):
    eps = 1e-10
    if w_std == 0:
        w_std = eps
    if p_std == 0:
        p_std = eps
    num_timesteps = qtraj.shape[0]
    T = exp_so3_T(qtraj)
    traj_samples = T[torch.randint(0, num_timesteps, [batch_size])]
    w_distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3) * w_std)
    p_distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3) * p_std)
    Gaussian_w = w_distribution.sample((batch_size,)).to(qtraj)
    Gaussian_p = p_distribution.sample((batch_size,)).unsqueeze(-1).to(qtraj)
    R_samples = traj_samples[:, :3, :3] @ exp_so3(Gaussian_w)
    p_samples = traj_samples[:, :3, 3:4] + Gaussian_p  # 끝 부분에 3:4는 3 한거랑 결과 값은 같게나오지만 dim=(n,3,1)이 나옴 (3 넣으면 (n,3))
    random_T = torch.cat([torch.cat([R_samples, p_samples], dim=2), torch.zeros(batch_size, 1, 4).to(R_samples)],
                         dim=1).detach()
    # random_T.requires_grad=True
    random_q = screw_bracket(log_SO3_T(random_T))
    if isinstance(random_q, str):
        random_q = Gaussian_sampling(qtraj, w_std, p_std, batch_size)
    random_q.requires_grad = True  # n x 6 dim

    return random_q  # random_T


# def SE3_spline(xpoints, Vpoints, T, delta, num_timesteps):
#     af = 10
#     deltaT = delta * T
#     aa = np.linspace(0, af, xpoints.shape[1])
#     tt = np.linspace(0, deltaT, Vpoints.shape[0])
#     dt = T / num_timesteps
#     # a_total = np.linspace(0,af,100)

#     tckw1 = interpolate.splrep(aa, xpoints[0])
#     tckw2 = interpolate.splrep(aa, xpoints[1])
#     tckw3 = interpolate.splrep(aa, xpoints[2])
#     tckp1 = interpolate.splrep(aa, xpoints[3])
#     tckp2 = interpolate.splrep(aa, xpoints[4])
#     tckp3 = interpolate.splrep(aa, xpoints[5])
#     tckV = interpolate.splrep(tt, Vpoints)

#     t_total = np.linspace(0, T, num_timesteps + 1)
#     timesteps = (t_total[:-1] + t_total[1:]) / 2

#     def splfun(a):
#         w1 = interpolate.splev(a, tckw1)
#         w2 = interpolate.splev(a, tckw2)
#         w3 = interpolate.splev(a, tckw3)
#         p1 = interpolate.splev(a, tckp1)
#         p2 = interpolate.splev(a, tckp2)
#         p3 = interpolate.splev(a, tckp3)
#         return w1, w2, w3, p1, p2, p3

#     def splder(a):
#         dw1 = interpolate.splev(a, tckw1, der=1)
#         dw2 = interpolate.splev(a, tckw2, der=1)
#         dw3 = interpolate.splev(a, tckw3, der=1)
#         dp1 = interpolate.splev(a, tckp1, der=1)
#         dp2 = interpolate.splev(a, tckp2, der=1)
#         dp3 = interpolate.splev(a, tckp3, der=1)
#         return dw1, dw2, dw3, dp1, dp2, dp3

#     def lenfun(a):
#         dw1, dw2, dw3, dp1, dp2, dp3 = splder(a)
#         dl = np.sqrt(dp1 ** 2 + dp2 ** 2 + dp3 ** 2)
#         return dl

#     def len_int(a):
#         length = integrate.quad(lenfun, 0, a)[0]
#         return length

#     def V(t):
#         if t <= deltaT:
#             speed = interpolate.splev(t, tckV)
#         else:
#             c = Vpoints[-1] / ((T - deltaT) ** 2)
#             speed = -c * ((t - deltaT) ** 2) + Vpoints[-1]
#         return speed

#     def l(t):
#         distance = integrate.quad(V, 0, t)[0]
#         return distance

#     def V2(t):
#         # V_mul = total_length/lt_final
#         speed = V_mul * V(t)
#         return speed

#     def V2mat(t):
#         speed = np.zeros(t.shape)

#         speed[t <= deltaT] = interpolate.splev(t[t <= deltaT], tckV)
#         c = Vpoints[-1] / ((T - deltaT) ** 2)
#         speed[t > deltaT] = -c * ((t[t > deltaT] - deltaT) ** 2) + Vpoints[-1]
#         return V_mul * speed

#     def l2(t):
#         distance = integrate.quad(V2, 0, t)[0]
#         return distance

#     total_length = len_int(af)
#     lt_final = l(T)
#     V_mul = total_length / lt_final

#     def xy_at_t(t):
#         # 3-1. t vs l(t)
#         lt = l2(t)

#         # 3-2. l(t) vs a
#         def len_res(a):
#             return len_int(a) - lt

#         a = fsolve(len_res, (af / 2))  # t*af/T))
#         w1, w2, w3, p1, p2, p3 = splfun(a)
#         return a, w1, w2, w3, p1, p2, p3

#     ww1 = np.zeros([num_timesteps])
#     ww2 = np.zeros([num_timesteps])
#     ww3 = np.zeros([num_timesteps])
#     pp1 = np.zeros([num_timesteps])
#     pp2 = np.zeros([num_timesteps])
#     pp3 = np.zeros([num_timesteps])
#     aa = np.zeros([num_timesteps])
#     dw1_f = np.zeros([num_timesteps])
#     dw2_f = np.zeros([num_timesteps])
#     dw3_f = np.zeros([num_timesteps])

#     for i in range(num_timesteps):
#         print('\r current timestep = ' + str(i + 1), end=' ')
#         # print(timesteps[i])
#         aa[i], ww1[i], ww2[i], ww3[i], pp1[i], pp2[i], pp3[i] = xy_at_t(timesteps[i])
#     ww = torch.from_numpy(np.vstack([ww1, ww2, ww3])).T
#     ww = ww.type(dtype)
#     R_pre = exp_so3(ww[:-1])
#     R_cur = exp_so3(ww[1:])
#     dw_temp = skew(log_SO3(R_pre.transpose(1, 2) @ R_cur) / dt)
#     dw_f = torch.cat([dw_temp, torch.zeros(1, 3).to(dw_temp)], dim=0)

#     #### 4. asigning speed to the spline
#     dw1, dw2, dw3, dp1, dp2, dp3 = splder(aa)
#     pnorm = np.sqrt(dp1 ** 2 + dp2 ** 2 + dp3 ** 2)
#     dp1_normalized = dp1 / pnorm
#     dp2_normalized = dp2 / pnorm
#     dp3_normalized = dp3 / pnorm
#     dp1_f = dp1_normalized * V2mat(timesteps)
#     dp2_f = dp2_normalized * V2mat(timesteps)
#     dp3_f = dp3_normalized * V2mat(timesteps)
#     # exp_so3

#     #### 5. xtraj and x_dot
#     screw_traj = torch.zeros(num_timesteps, 6)
#     twist_traj = torch.zeros(num_timesteps, 6)

#     screw_traj[:, 0] = torch.from_numpy(ww1)
#     screw_traj[:, 1] = torch.from_numpy(ww2)
#     screw_traj[:, 2] = torch.from_numpy(ww3)
#     screw_traj[:, 3] = torch.from_numpy(pp1)
#     screw_traj[:, 4] = torch.from_numpy(pp2)
#     screw_traj[:, 5] = torch.from_numpy(pp3)

#     twist_traj[:, :3] = dw_f
#     twist_traj[:, 3] = torch.from_numpy(dp1_f)
#     twist_traj[:, 4] = torch.from_numpy(dp2_f)
#     twist_traj[:, 5] = torch.from_numpy(dp3_f)

#     return screw_traj, twist_traj


# def dim3_spline(xpoints, ypoints, zpoints, Vpoints, T, delta, num_timesteps):
#     af = 10
#     deltaT = delta * T
#     aa = np.linspace(0, af, xpoints.shape[0])
#     tt = np.linspace(0, deltaT, Vpoints.shape[0])
#     dt = T / num_timesteps

#     tckp1 = interpolate.splrep(aa, xpoints)
#     tckp2 = interpolate.splrep(aa, ypoints)
#     tckp3 = interpolate.splrep(aa, zpoints)
#     tckV = interpolate.splrep(tt, Vpoints)

#     t_total = np.linspace(0, T, num_timesteps + 1)
#     timesteps = (t_total[:-1] + t_total[1:]) / 2

#     def splfun(a):
#         p1 = interpolate.splev(a, tckp1)
#         p2 = interpolate.splev(a, tckp2)
#         p3 = interpolate.splev(a, tckp3)
#         return p1, p2, p3

#     def splder(a):
#         dp1 = interpolate.splev(a, tckp1, der=1)
#         dp2 = interpolate.splev(a, tckp2, der=1)
#         dp3 = interpolate.splev(a, tckp3, der=1)
#         return dp1, dp2, dp3

#     def lenfun(a):
#         dp1, dp2, dp3 = splder(a)
#         dl = np.sqrt(dp1 ** 2 + dp2 ** 2 + dp3 ** 2)
#         return dl

#     def len_int(a):
#         length = integrate.quad(lenfun, 0, a)[0]
#         return length

#     def V(t):
#         if t <= deltaT:
#             speed = interpolate.splev(t, tckV)
#         else:
#             c = Vpoints[-1] / ((T - deltaT) ** 2)
#             speed = -c * ((t - deltaT) ** 2) + Vpoints[-1]
#         return speed

#     def l(t):
#         distance = integrate.quad(V, 0, t)[0]
#         return distance

#     def V2(t):
#         # V_mul = total_length/lt_final
#         speed = V_mul * V(t)
#         return speed

#     def V2mat(t):
#         speed = np.zeros(t.shape)

#         speed[t <= deltaT] = interpolate.splev(t[t <= deltaT], tckV)
#         c = Vpoints[-1] / ((T - deltaT) ** 2)
#         speed[t > deltaT] = -c * ((t[t > deltaT] - deltaT) ** 2) + Vpoints[-1]
#         return V_mul * speed

#     def l2(t):
#         distance = integrate.quad(V2, 0, t)[0]
#         return distance

#     total_length = len_int(af)
#     lt_final = l(T)
#     V_mul = total_length / lt_final

#     def xy_at_t(t):
#         # 3-1. t vs l(t)
#         lt = l2(t)

#         # 3-2. l(t) vs a
#         def len_res(a):
#             return len_int(a) - lt

#         a = fsolve(len_res, (af / 2))  # t*af/T))
#         p1, p2, p3 = splfun(a)
#         return a, p1, p2, p3

#     pp1 = np.zeros([num_timesteps])
#     pp2 = np.zeros([num_timesteps])
#     pp3 = np.zeros([num_timesteps])
#     aa = np.zeros([num_timesteps])

#     for i in range(num_timesteps):
#         print('\r current timestep = ' + str(i + 1), end=' ')
#         # print(timesteps[i])
#         aa[i], pp1[i], pp2[i], pp3[i] = xy_at_t(timesteps[i])

#     #### 4. asigning speed to the spline
#     dp1, dp2, dp3 = splder(aa)
#     pnorm = np.sqrt(dp1 ** 2 + dp2 ** 2 + dp3 ** 2)
#     dp1_normalized = dp1 / pnorm
#     dp2_normalized = dp2 / pnorm
#     dp3_normalized = dp3 / pnorm
#     dp1_f = dp1_normalized * V2mat(timesteps)
#     dp2_f = dp2_normalized * V2mat(timesteps)
#     dp3_f = dp3_normalized * V2mat(timesteps)
#     # exp_so3

#     #### 5. xtraj and x_dot
#     ptraj = torch.zeros(num_timesteps, 3)
#     pdot = torch.zeros(num_timesteps, 3)

#     ptraj[:, 0] = torch.from_numpy(pp1)
#     ptraj[:, 1] = torch.from_numpy(pp2)
#     ptraj[:, 2] = torch.from_numpy(pp3)

#     pdot[:, 0] = torch.from_numpy(dp1_f)
#     pdot[:, 1] = torch.from_numpy(dp2_f)
#     pdot[:, 2] = torch.from_numpy(dp3_f)

#     return ptraj, pdot


def line_traj(q1, q2, q3, T, delta, num_timesteps):
    timesteps = torch.linspace(0, T, num_timesteps + 1)

    qinit = torch.tensor([q1[0], q2[0], q3[0]])
    qfinal = torch.tensor([q1[1], q2[1], q3[1]])
    q_dot = torch.zeros(num_timesteps, 3)
    qtraj = torch.zeros(num_timesteps, 3)
    deltaT = delta * T
    direction = (qfinal - qinit) / torch.norm((qfinal - qinit), 3)
    max_speed = (torch.norm((qfinal - qinit), 2) /
                 (T + -1 / ((1 - delta) * (1 - delta) * T * T * 3) * (T - deltaT) * (T - deltaT) * (T - deltaT)))

    def V(t):
        if t <= deltaT:
            speed = max_speed
        else:
            speed = -(max_speed) / ((1 - delta) * (1 - delta) * T * T) * (t - deltaT) * (t - deltaT) + max_speed
        return speed

    def s(t):
        if t <= deltaT:
            distance = max_speed * t
        else:
            distance = (max_speed * t + -(max_speed) / ((1 - delta) * (1 - delta) * T * T * 3) *
                        (t - deltaT) * (t - deltaT) * (t - deltaT))
        return distance

    for i in range(num_timesteps):
        t = (timesteps[i] + timesteps[i + 1]) / 2
        vel_current = V(t)
        dist_current = s(t)
        q_dot[i, 0] = vel_current * direction[0]
        q_dot[i, 1] = vel_current * direction[1]
        q_dot[i, 2] = vel_current * direction[2]

        qtraj[i, 0] = qinit[0] + dist_current * direction[0]
        qtraj[i, 1] = qinit[1] + dist_current * direction[1]
        qtraj[i, 2] = qinit[1] + dist_current * direction[2]

    return qtraj, q_dot


def line_traj_rot_storage(traj_number):
    if traj_number == 1:
        q1 = [-0.5, 0]
        q2 = [0.6, 0]
        q3 = [1., 0]

    else:
        print('Wrong traj_number!')
        assert (False)

    return q1, q2, q3


def line_traj_trans_storage(traj_number):
    if traj_number == 1:
        q1 = [-0.3, 0.1]
        q2 = [0, 0.3]
        q3 = [0, -0.3]

    elif traj_number == 2:
        q1 = [-6, 0.1]
        q2 = [0, 3]
        q3 = [0, -3]

    else:
        print('Wrong traj_number!')
        assert (False)

    return q1, q2, q3


def spline_traj_rot_storage(traj_number):
    if traj_number == 1:
        xpoints = np.array([0.5, 1.5, 2.0, 1.5, 1.0]) + 0.3
        ypoints = np.array([0.5, 0.8, 1.1, 1.4, 1.7]) + 0.5
        zpoints = xpoints
        Vpoints = np.array([1.5, 1.7, 1.9, 2.3, 2.6])

    elif traj_number == 2:
        xpoints = np.array([0.5, 1.5, 2.0, 1.5, 1.0])
        ypoints = np.array([0.5, 0.8, 1.1, 1.4, 1.7])
        Vpoints = np.array([1.5, 1.7, 1.9, 2.3, 2.6])

    else:
        print('Wrong traj_number!')
        assert (False)

    return xpoints, ypoints, zpoints, Vpoints


def spline_traj_trans_storage(traj_number):
    if traj_number == 1:
        xpoints = np.array([-1.8, -1, -0.5, 0])
        ypoints = np.array([1.8234766, 1.8188585, 0.5, 0])
        zpoints = np.array([1.8234766, 1.2188585, 0.9, 0])
        Vpoints = np.array([1.5, 1.7, 1.9, 2.3, 1.5])
    elif traj_number == 2:
        xpoints = np.array([1, 0, 0, 0, 0])
        ypoints = np.array([0, 0.5, 1, 0.5, 0])
        zpoints = np.array([0, 0.5, 0, 0.5, 1])
        Vpoints = np.array([1.5, 1.7, 1.9, 2.3, 1.5])
    else:
        print('Wrong traj_number!')
        assert (False)

    return xpoints, ypoints, zpoints, Vpoints


def traj_guide_message(traj_type_rot, traj_type_trans):
    main_message = "Note that input must be in the form of (traj_type_rot, traj_type_trans, "
    print("Possible each traj_type is one of: ('line', 'spline', 'custom_line', 'custom_spline')")
    print("Current traj_type_rot is " + traj_type_rot)
    print("Current traj_type_trans is " + traj_type_trans)

    if traj_type_rot == 'line' or traj_type_rot == 'spline':
        main_message += "traj_number_rot, "
    elif traj_type_rot == 'custom_line':
        main_message += "w1, w2, w3, "
    elif traj_type_rot == 'custom_spline':
        main_message += "w1points, w2points, w3points, wVpoints, "
    else:
        print('traj_type_rot error!')
        assert (False)

    if traj_type_trans == 'line' or traj_type_trans == 'spline':
        main_message += "traj_number_trans, "
    elif traj_type_trans == 'custom_line':
        main_message += "p1, p2, p3, "
    elif traj_type_trans == 'custom_spline':
        main_message += "p1points, p2points, p3points, pVpoints, "
    else:
        print('traj_type_trans error!')
        assert (False)

    main_message += "t_final, delta, num_timesteps, **kwargs)"
    print(main_message)


def SE3_traj(traj_type_rot, traj_type_trans, *args, plot=True, **kargs):
    traj_guide_message(traj_type_rot, traj_type_trans)

    t_final = args[-4]
    delta = args[-3]
    num_timesteps = args[-2]
    plot_gap_sample = args[-1]

    # Rotation
    if traj_type_rot == 'line':
        traj_number_rot = args[0]
        print("traj_number_rot = " + str(traj_number_rot))
        trans_args_num = 1

        w1, w2, w3 = line_traj_rot_storage(traj_number_rot)
        wtraj, wdot = line_traj(w1, w2, w3, t_final, delta, num_timesteps)

    elif traj_type_rot == 'spline':
        traj_number_rot = args[0]
        print("traj_number_rot = " + str(traj_number_rot))
        trans_args_num = 1

        w1, w2, w3, wVpoints = spline_traj_rot_storage(traj_number_rot)
        wtraj, wdot = dim3_spline(w1, w2, w3, wVpoints, t_final, delta, num_timesteps)

    elif traj_type_rot == 'custom_line':
        w1 = args[0]
        w2 = args[1]
        w3 = args[2]
        trans_args_num = 3

        wtraj, wdot = line_traj(w1, w2, w3, t_final, delta, num_timesteps)

    elif traj_type_rot == 'custom_spline':
        w1 = args[0]
        w2 = args[1]
        w3 = args[2]
        wVpoints = args[3]
        trans_args_num = 4

        wtraj, wdot = dim3_spline(w1, w2, w3, wVpoints, t_final, delta, num_timesteps)

    # Translation
    if traj_type_trans == 'line':
        traj_number_trans = args[trans_args_num]
        print("traj_number_trans = " + str(traj_number_trans))

        p1, p2, p3 = line_traj_trans_storage(traj_number_trans)
        ptraj, pdot = line_traj(p1, p2, p3, t_final, delta, num_timesteps)

    elif traj_type_trans == 'spline':
        traj_number_trans = args[trans_args_num]
        print("traj_number_trans = " + str(traj_number_trans))

        p1, p2, p3, pVpoints = spline_traj_trans_storage(traj_number_trans)
        ptraj, pdot = dim3_spline(p1, p2, p3, pVpoints, t_final, delta, num_timesteps)

    elif traj_type_trans == 'custom_line':
        p1 = args[trans_args_num]
        p2 = args[trans_args_num + 1]
        p3 = args[trans_args_num + 2]

        ptraj, pdot = line_traj(p1, p2, p3, t_final, delta, num_timesteps)

    elif traj_type_trans == 'custom_spline':
        p1 = args[trans_args_num + 0]
        p2 = args[trans_args_num + 1]
        p3 = args[trans_args_num + 2]
        pVpoints = args[trans_args_num + 3]

        ptraj, pdot = dim3_spline(p1, p2, p3, pVpoints, t_final, delta, num_timesteps)

    ptraj = ptraj
    wtraj = wtraj
    pdot = pdot
    wdot = wdot

    qtraj = torch.cat([wtraj, ptraj], dim=1)
    qdot = torch.cat([wdot, pdot], dim=1)

    qfinal = qtraj[-1].unsqueeze(0)
    Xstable = exp_so3_T(qfinal).squeeze(0)

    if plot == True:
        traj_plot(qtraj, Xstable, plot_gap_ref=(plot_gap_sample), **kargs)

    return qtraj, qdot, Xstable


def traj_plot(qtraj, Xstable, plot_gap_ref=1, **kwargs):
    num_timesteps = qtraj.shape[0]
    T_traj = exp_so3_T(qtraj)
    qmin = torch.min(qtraj[:, 3:], dim=0).values
    qmax = torch.max(qtraj[:, 3:], dim=0).values
    qlength = torch.norm(qmax - qmin)

    fig = plt.figure(figsize=(14, 12))
    ax = plt.axes(projection='3d')
    if 'axis_range' in kwargs.keys():
        axis_range = kwargs['axis_range']
        ax.set_xlim3d(axis_range[0, 0], axis_range[0, 1])
        ax.set_ylim3d(axis_range[1, 0], axis_range[1, 1])
        ax.set_zlim3d(axis_range[2, 0], axis_range[2, 1])

    if 'view_angle' in kwargs.keys():
        view_angle = kwargs['view_angle']
        ax.view_init(view_angle[0], view_angle[1])

    length = qlength / 20
    for i in range(num_timesteps):
        RR = T_traj[i, :3, :3].detach().to(device_c).numpy()
        pp = T_traj[i, :3, 3].detach().to(device_c).numpy()
        if i % plot_gap_ref == 0:
            ax.plot3D([pp[0], pp[0] + length * RR[0, 0]], [pp[1], pp[1] + length * RR[1, 0]],
                      [pp[2], pp[2] + length * RR[2, 0]], 'red')
            ax.plot3D([pp[0], pp[0] + length * RR[0, 1]], [pp[1], pp[1] + length * RR[1, 1]],
                      [pp[2], pp[2] + length * RR[2, 1]], 'green')
            ax.plot3D([pp[0], pp[0] + length * RR[0, 2]], [pp[1], pp[1] + length * RR[1, 2]],
                      [pp[2], pp[2] + length * RR[2, 2]], 'blue')

    ax.scatter(T_traj[0, 0, 3].to(device_c).detach().numpy(), T_traj[0, 1, 3].to(device_c).detach().numpy(),
               T_traj[0, 2, 3].to(device_c).detach().numpy(), marker="s", s=170, c='cyan')
    ax.scatter(Xstable[0, 3].to(device_c).detach().numpy(), Xstable[1, 3].to(device_c).detach().numpy(),
               Xstable[2, 3].to(device_c).detach().numpy(), marker="o", s=170, c='blue')
    plt.show()


def random_sampling_check(qtraj, Xstable, w_std, p_std, num_samples=10, plot_gap_ref=1):
    T_traj = exp_so3_T(qtraj)
    qmin = torch.min(qtraj[:, 3:], dim=0).values
    qmax = torch.max(qtraj[:, 3:], dim=0).values
    qlength = torch.norm(qmax - qmin)
    num_timesteps = qtraj.shape[0]

    fig = plt.figure(figsize=(14, 12))
    ax = plt.axes(projection='3d')
    length = qlength / 20
    for i in range(num_timesteps):
        RR = T_traj[i, :3, :3].detach().to(device_c).numpy()
        pp = T_traj[i, :3, 3].detach().to(device_c).numpy()
        if i % plot_gap_ref == 0:
            ax.plot3D([pp[0], pp[0] + length * RR[0, 0]], [pp[1], pp[1] + length * RR[1, 0]],
                      [pp[2], pp[2] + length * RR[2, 0]], 'red')
            ax.plot3D([pp[0], pp[0] + length * RR[0, 1]], [pp[1], pp[1] + length * RR[1, 1]],
                      [pp[2], pp[2] + length * RR[2, 1]], 'green')
            ax.plot3D([pp[0], pp[0] + length * RR[0, 2]], [pp[1], pp[1] + length * RR[1, 2]],
                      [pp[2], pp[2] + length * RR[2, 2]], 'blue')
    ax.scatter(T_traj[0, 0, 3].to(device_c).detach().numpy(), T_traj[0, 1, 3].to(device_c).detach().numpy(),
               T_traj[0, 2, 3].to(device_c).detach().numpy(), marker="s", s=170, c='cyan')
    ax.scatter(Xstable[0, 3].to(device_c).detach().numpy(), Xstable[1, 3].to(device_c).detach().numpy(),
               Xstable[2, 3].to(device_c).detach().numpy(), marker="o", s=170, c='blue')

    if w_std == 0 and p_std == 0:
        qsample = Uniform_sampling(qtraj, num_samples)
    else:
        qsample = Gaussian_sampling(qtraj, w_std, p_std, num_samples)

    Tsample = exp_so3_T(qsample)

    for j in range(num_samples):
        RR = Tsample[j, :3, :3].detach().to(device_c).numpy()
        pp = Tsample[j, :3, 3].detach().to(device_c).numpy()

        ax.plot3D([pp[0], pp[0] + length * RR[0, 0]], [pp[1], pp[1] + length * RR[1, 0]],
                  [pp[2], pp[2] + length * RR[2, 0]], 'red')
        ax.plot3D([pp[0], pp[0] + length * RR[0, 1]], [pp[1], pp[1] + length * RR[1, 1]],
                  [pp[2], pp[2] + length * RR[2, 1]], 'green')
        ax.plot3D([pp[0], pp[0] + length * RR[0, 2]], [pp[1], pp[1] + length * RR[1, 2]],
                  [pp[2], pp[2] + length * RR[2, 2]], 'blue')
        ax.scatter(pp[0], pp[1], pp[2], marker='o', c='purple')

def SE3_12dim_to_mat(vec):
    original_shape = vec.shape
    num_traj = original_shape[0]
    if len(original_shape) == 2:
        num_points = int(original_shape[-1]/12)
    else:
        num_points = original_shape[1]
    vec = vec.reshape(-1, 3, 4)
    T = torch.zeros(len(vec), 4, 4).to(vec)
    T[:, :3] = vec
    T[:, -1, -1] = 1
    T = T.reshape(num_traj, num_points, 4, 4)
    return T

def quaternions_to_rotation_matrices_torch(quaternions):
    assert quaternions.shape[1] == 4

    # initialize
    K = quaternions.shape[0]
    R = quaternions.new_zeros((K, 3, 3))

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[:, 1] ** 2
    yy = quaternions[:, 2] ** 2
    zz = quaternions[:, 3] ** 2
    ww = quaternions[:, 0] ** 2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = quaternions.new_zeros((K, 1))
    s[n != 0] = 2 / n[n != 0]

    xy = s[:, 0] * quaternions[:, 1] * quaternions[:, 2]
    xz = s[:, 0] * quaternions[:, 1] * quaternions[:, 3]
    yz = s[:, 0] * quaternions[:, 2] * quaternions[:, 3]
    xw = s[:, 0] * quaternions[:, 1] * quaternions[:, 0]
    yw = s[:, 0] * quaternions[:, 2] * quaternions[:, 0]
    zw = s[:, 0] * quaternions[:, 3] * quaternions[:, 0]

    xx = s[:, 0] * xx
    yy = s[:, 0] * yy
    zz = s[:, 0] * zz

    idxs = torch.arange(K).to(quaternions.device)
    R[idxs, 0, 0] = 1 - yy - zz
    R[idxs, 0, 1] = xy - zw
    R[idxs, 0, 2] = xz + yw

    R[idxs, 1, 0] = xy + zw
    R[idxs, 1, 1] = 1 - xx - zz
    R[idxs, 1, 2] = yz - xw

    R[idxs, 2, 0] = xz - yw
    R[idxs, 2, 1] = yz + xw
    R[idxs, 2, 2] = 1 - xx - yy

    return R