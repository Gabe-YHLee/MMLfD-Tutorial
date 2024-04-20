import numpy as np

# skew
def skew(w):
    if len(w) == 3:
        W = np.array([[0, -w[2], w[1]],
                    [w[2], 0, -w[0]],
                    [-w[1], w[0], 0]])
    if len(w) == 6:
        W = np.array([[0, -w[2], w[1]],
                    [w[2], 0, -w[0]],
                    [-w[1], w[0], 0]])
        W = np.vstack([
            np.concatenate([W, w[3:].reshape(3,1)], axis=1), np.array([0,0,0,0])
        ])
    return W

def invskew(W):
    if len(W) == 3:
        w = np.array([-W[1,2], W[0,2], -W[0,1]])
    if len(W) == 4:
        w = np.array([-W[1,2], W[0,2], -W[0,1]])
        w = np.hstack([w, W[:3, 3]])
    return w
    
# SO3 exponential
def exp_so3(w):
    if len(w) != 3:
        raise ValueError('Dimension is not 3')
    if w.shape == (3,3):
        w = invskew(w)
    eps = 1e-14
    wnorm = np.sqrt(sum(w*w))
    if wnorm < eps:
        R = np.eye(3)
    else:
        wnorm_inv = 1 / wnorm
        cw = np.cos(wnorm)
        sw = np.sin(wnorm)
        W = skew(w)
        R = np.eye(3) + sw * wnorm_inv * W + (1 - cw) * np.power(wnorm_inv,2) * W.dot(W)
    return R

# SE3 exponential
def exp_se3(S):
    if len(S) != 6:
        raise ValueError('Dimension is not 6')
    if S.shape == (4,4):
        S = invskew(S)
    w = S[0:3]
    v = S[3:6]
    eps = 1e-14
    wnorm = np.sqrt(sum(w*w))
    if wnorm < eps:
        T = np.eye(4)
        T[0:3,3] = v.reshape(3)
    else:
        wnorm_inv = 1 / wnorm
        cw = np.cos(wnorm)
        sw = np.sin(wnorm)
        W = skew(w)
        P = np.eye(3) + (1 - cw) * np.power(wnorm_inv,2) * W + (wnorm - sw) * np.power(wnorm_inv,3) * W.dot(W)
        T = np.eye(4)
        T[0:3,0:3] = exp_so3(w)
        T[0:3,3] = P.dot(v).reshape(3)
    return T

def clipping(x, low=-1.0, high=1.0):
    eps = 1e-6
    if x <= low:
        x = low + eps
    elif x >= high:
        x = high - eps
    return x

# SO3 log
def log_SO3(R):
    angle_threshold = 1e-6
    trace = np.trace(R)
    theta = np.arccos(clipping((trace - 1) / 2))
    if np.abs(trace + 1) >= angle_threshold:
        skew_w = (R - R.transpose()) / (2 * np.sin(theta)) * theta
    elif np.abs(trace - 3) < 1e-4:
        skew_w = np.zeros((3,3))
    elif np.abs(trace + 1) < angle_threshold:
        if R[2,2] != -1:
            r = R[2, 2]
            w = R[:, 2]
            # print(w)
            w[2] += 1
        elif R[1,1] != -1:
            r = R[1,1]
            w = R[:,1]
            w[1] += 1
        elif R[0,0] != -1:
            r = R[0,0]
            w = R[:,0]
            w[0] += 1
        else:
            print(f'ERROR: should be fixed.')
            NotImplementedError
        skew_w = skew(np.pi / np.sqrt((2 * (1 + r))) * w)
    else:
        skew_w = np.zeros((3,3))
    return skew_w

# SE3 log
def log_SE3(T):
    angle_threshold = 1e-6
    R = T[:3,:3]
    trace = np.trace(R)
    skew_S = np.zeros((4,4))
    if np.abs(trace-3) < angle_threshold:
        skew_S[:3, 3] = T[:3, 3]
    if np.abs(trace-3) >= angle_threshold:
        skew_w = log_SO3(R)
        theta = np.arccos(clipping(0.5*(trace-1)))
        wmat = skew_w / theta
        identity = np.eye(3)
        invG = (1/theta) * identity - 0.5 * wmat + (1/theta - 0.5/np.tan(0.5*theta)) * wmat@wmat
        skew_S[:3, :3] = skew_w
        skew_S[:3, 3] = (theta * (invG@T[:3, 3:4])).reshape(3)
    return skew_S

# Adjoint of SE3
def Adjoint_SE3(T):
    R = T[0:3,0:3]
    p = T[0:3,3]
    skewp = skew(p)
    
    AdT = np.eye(6)
    AdT[0:3,0:3] = R
    AdT[0:3,3:6] = np.zeros((3,3))
    AdT[3:6,0:3] = skewp.dot(R)
    AdT[3:6,3:6] = R
    return AdT

# small adjoint of se3
def adjoint_se3(S):
    w = S[0:3]
    v = S[3:6]
    skeww = skew(w)
    skewv = skew(v)
    
    adS = np.zeros((6,6))
    adS[0:3,0:3] = skeww
    adS[3:6,0:3] = skewv
    adS[3:6,3:6] = skeww
    return adS

# Joint and theta to SO3
def Joint_to_SO3(w, theta):
    if len(w) != 3:
        raise ValueError('Dimension is not 3')
    cw = np.cos(theta)
    sw = np.sin(theta)
    W = skew(w)
    R = np.eye(3) + sw * W + (1 - cw) * W.dot(W)
    return R

# Screw and theta to SE3
def Screw_to_SE3(S, theta):
    if len(S) != 6:
        raise ValueError('Dimension is not 6')
    w = S[0:3]
    v = S[3:6]
    eps = 1e-14
    wnorm = np.sqrt(sum(w*w))    
    if wnorm < eps:
        T = np.eye(4)
        T[0:3,3] = v * theta
    else:
        cw = np.cos(theta)
        sw = np.sin(theta)
        W = skew(w)
        P = theta * np.eye(3) + (1 - cw) * W + (theta - sw) * W.dot(W)
        T = np.eye(4)
        T[0:3,0:3] = Joint_to_SO3(w, theta)
        T[0:3,3] = P.dot(v).reshape(3)
    return T

# Inverse of SE3
def inv_SE3(T):
    R = T[0:3,0:3]
    p = T[0:3,3]
    Tinv = np.eye(4)
    Tinv[0:3,0:3] = np.transpose(R)
    Tinv[0:3,3] = -np.transpose(R).dot(p).reshape(3)
    return Tinv

def pinv(J):
    return J.transpose()@(np.linalg.inv(J@J.transpose()))

def Dlog_SO3(R, Rdot):
    trace = np.trace(R)
    theta = np.arccos(clipping(0.5*(trace-1)))
    
    eps = 1.0e-10
    if theta < eps:
        return -(1/8) * np.trace(Rdot) *(R-R.transpose()) + \
            1/2 * (Rdot-Rdot.transpose())
    else:
        return np.trace(Rdot) * (
            theta * np.cos(theta)/np.sin(theta) - 1
        )*(R-R.transpose())/(4*np.sin(theta)**2) + \
            theta/(2*np.sin(theta)) * (Rdot-Rdot.transpose())