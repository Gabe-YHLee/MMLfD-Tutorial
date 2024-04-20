import torch
from robot.groups import PlanarMobileRobot

def get_wall_original_batch(batchsize=1):
    low = 1.2
    high = 4   
    # tau = (n, 3)
    wall1 = torch.tensor([[low, low, high], [high, low, low]])
    wall2 = torch.tensor([[low, low, high], [-high, -low, -low]])
    wall3 = torch.tensor([[-low, -low, -high], [high, low, low]])
    wall4 = torch.tensor([[-low, -low, -high], [-high, -low, -low]])
    walls_original = torch.cat([wall1, wall2, wall3, wall4], dim=1).unsqueeze(0).repeat(batchsize, 1, 1) # (bs, 2, 12)
    # walls_rot = Rot_mat @ walls_original
    wall1_rot = walls_original[:, :, :3]
    wall2_rot = walls_original[:, :, 3:6]
    wall3_rot = walls_original[:, :, 6:9]
    wall4_rot = walls_original[:, :, 9:]
    return wall1_rot, wall2_rot, wall3_rot, wall4_rot
    
def collision_check_wall_traj(trajs_batch):
    batchsize = len(trajs_batch)
    walls_batch = get_wall_original_batch(batchsize)
    if trajs_batch.device is not torch.device('cpu'):
        trajs_batch = trajs_batch.cpu()
    batch_size = len(trajs_batch)
    trajs_batch = trajs_batch.reshape(batch_size, -1, 2)
    wall1_batch = walls_batch[0].transpose(1, 2) # bs, 3, 2
    wall2_batch = walls_batch[1].transpose(1, 2)
    wall3_batch = walls_batch[2].transpose(1, 2)
    wall4_batch = walls_batch[3].transpose(1, 2)

    p_wall1 = wall1_batch[:, :2]
    p_wall2 = wall2_batch[:, :2]
    p_wall3 = wall3_batch[:, :2]
    p_wall4 = wall4_batch[:, :2]
    q_wall1 = wall1_batch[:, 1:]
    q_wall2 = wall2_batch[:, 1:]
    q_wall3 = wall3_batch[:, 1:]
    q_wall4 = wall4_batch[:, 1:]

    p_wall = torch.cat([p_wall1, p_wall2, p_wall3, p_wall4], dim=1)
    p_wall_repeat_flat = p_wall.unsqueeze(2).repeat(1, 1, trajs_batch.shape[1]- 1, 1).reshape(-1, 2)
    q_wall = torch.cat([q_wall1, q_wall2, q_wall3, q_wall4], dim=1)
    q_wall_repeat_flat = q_wall.unsqueeze(2).repeat(1, 1, trajs_batch.shape[1]- 1, 1).reshape(-1, 2)

    p_traj = trajs_batch[:, :-1]
    p_traj_repeat_flat = p_traj.repeat(1, p_wall.shape[1], 1).reshape(-1, 2)
    q_traj = trajs_batch[:, 1:]
    q_traj_repeat_flat = q_traj.repeat(1, p_wall.shape[1], 1).reshape(-1, 2)

    intersect = intersection_check_batch(
        p_wall_repeat_flat, q_wall_repeat_flat,
        p_traj_repeat_flat, q_traj_repeat_flat)
    intersect_total = intersect.reshape(batch_size, -1).sum(dim=-1)
    intersect_total[intersect_total > 0] = 1
    return intersect_total

def on_segment_batch(p, q, r):
    on_seg = torch.zeros(len(r)).to(p)
    mins = torch.minimum(p, q)
    maxes = torch.maximum(p, q)
    on_seg[((r[:, 0] >= mins[:, 0]) * 
            (r[:, 0] <= maxes[:, 0]) * 
            (r[:, 1] >= mins[:, 1]) * 
            (r[:, 1] <= maxes[:, 1])
            )] = 1
    return on_seg

def orientation_batch(p, q, r):
    val = ((q[:, 1] - p[:, 1]) * (r[:, 0] - q[:, 0])) - ((q[:, 0] - p[:, 0]) * (r[:, 1] - q[:, 1]))
    val[val > 0] = 1
    val[val < 0] = -1
    return val

def intersection_check_batch(p1, q1, p2, q2):
    intersect = torch.zeros(len(p1)).to(p1)
    o1 = orientation_batch(p1, q1, p2)
    o2 = orientation_batch(p1, q1, q2)
    o3 = orientation_batch(p2, q2, p1)
    o4 = orientation_batch(p2, q2, q1)
    intersect[(o1 != o2) * (o3 != o4)] = 1
    intersect[(o1 == 0) * (on_segment_batch(p1, q1, p2) == 1)] = 1
    intersect[(o2 == 0) * (on_segment_batch(p1, q1, q2) == 1)] = 1
    intersect[(o3 == 0) * (on_segment_batch(p2, q2, p1) == 1)] = 1
    intersect[(o4 == 0) * (on_segment_batch(p2, q2, q1) == 1)] = 1
    return intersect