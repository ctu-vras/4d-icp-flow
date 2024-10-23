import random
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import open3d as o3d
from tqdm import tqdm
from sklearn.cluster import DBSCAN


from utils import MinimumBoundingBox, estimateMinimumAreaBox

def load_and_sample_PONE_data(PONE_PATH : str, MAX_RANGE=60, GROUND_HEIGHT=0.6, Y_RANGE=30, frame=0, max_frame=10):
    
    pcd_file = np.load(PONE_PATH, allow_pickle=True)

    all_scans = pcd_file["scan_list"]
    scan = all_scans[frame]
    poses = pcd_file["odom_list"]  
    ego_trans = poses['transformation']              

    pts_list = []
    

    for i in range(frame, len(all_scans)):
        if i >= max_frame:
            break

        scan = all_scans[i]
        pts = np.concatenate([scan["x"], scan["z"].reshape(-1, 1)], axis=1)
        pts = pts[np.linalg.norm(pts, axis=1) < MAX_RANGE]
        pts = pts[pts[:, 2] > GROUND_HEIGHT]
        pts = pts[np.abs(pts[:, 1]) < Y_RANGE]

        # synchronize
        pose = ego_trans[i]
        pts = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        pts = pts @ pose.T 
        pts[:, 3] = i
        
        pts_list.append(pts)
        

        
    pts = np.concatenate(pts_list, axis=0)

    return pts






def ICP4dSolver(time_pts_list, available_times, threshold=2.5, per_frame_move_thresh=0.05, RECONSTRUCT=False):
    
    trans_init = np.eye(4)
    boxes = []
    # corner_pts_list = []
    trans_list = []

    for t_idx, t in enumerate(sorted(available_times)):
        
        
        # if t != 0 and RECONSTRUCT:
            # print('not implemented correctly')
            # target_pts = np.concatenate(time_pts_list[:t_idx], axis=0)
        # else:
        target_pts = instance_pts[mask1, :3]
        box_t, corner_pts_t = estimateMinimumAreaBox(target_pts)
        target_pts = target_pts - box_t[:3]
        
        mask2 = instance_pts[:, -1] == t
        
        source_pts = instance_pts[mask2, :3]
        box_s, corner_pts_s = estimateMinimumAreaBox(source_pts)
        source_pts = source_pts - box_s[:3]    


        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_pts)

        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_pts)
            

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        

        trans = reg_p2p.transformation
        trans_init = trans 

        traj_trans = trans.copy()    # inverse

        trans_list.append(traj_trans)
        boxes.append(box_s)

    
    # Trajectory
    trajectory = []
    for i in range(len(time_pts_list)):
        trajectory_p = np.linalg.inv(trans_list[i])[:3, -1] + boxes[i][:3]
        trajectory.append(trajectory_p)

    trajectory = np.stack(trajectory)

    position_difference = np.linalg.norm(trajectory[0] - trajectory[-1], axis=0)

    dynamic = position_difference > per_frame_move_thresh * len(trajectory)

    return trajectory, dynamic, trans_list, boxes, 


if __name__ == '__main__':
    RECONSTRUCT = False
    EPS = 0.4
    MAX_ADJACENT_TIMES = 7
    Z_SCALE = 0.3
    MIN_NBR_PTS = 30
    MIN_INCLUDED_TIMES = 1
    
    ##### Following lines are used to generate data
    PONE_PATH = '/mnt/personal/vacekpa2/data/PONE/Veh01_20240807_102809_000_PCD.npz'
    pts = load_and_sample_PONE_data(PONE_PATH, frame=0, max_frame=6)
    pts[:, 3] = pts[:, 3] * EPS / (MAX_ADJACENT_TIMES + 0.01)   # specified time constraints for clustering
    np.save('data/pts.npy', pts)

    pts = np.load('data/pts.npy')
    
    pts[:, 2] *= Z_SCALE
    cluster_ids = DBSCAN(eps=EPS, min_samples=10).fit_predict(pts)
    cluster_ids += 1

    dynamic_mask = np.zeros(cluster_ids.shape[0], dtype=bool)
    logic_mask = np.zeros(cluster_ids.shape[0], dtype=bool)
    trajectories = {}
    boxes_dict = {}
    dynamic_id = {}

    for i in tqdm(range(cluster_ids.max()), desc='Per-instance 4D ICP'):
        if i == 0: continue # noise

        i_mask = cluster_ids == i
        instance_pts = pts[i_mask]

        if len(instance_pts) < MIN_NBR_PTS:
            continue
        
        if len(np.unique(instance_pts[:,-1])) < MIN_INCLUDED_TIMES:
            continue

        available_times = sorted(np.unique(instance_pts[:,-1]))
        mask1 = instance_pts[:, -1] == available_times[0]

        time_pts_list = [instance_pts[:, :4][instance_pts[:, -1] == t] for t in available_times]
        
        try:
            trajectory, dynamic, trans_list, boxes = ICP4dSolver(time_pts_list, available_times, threshold=3.5, per_frame_move_thresh=0.1, RECONSTRUCT=False)
            dynamic_mask[i_mask] = dynamic
            dynamic_id[i] = dynamic
            trajectories[i] = trajectory
            boxes_dict[i] = boxes
            logic_mask[i_mask] = True
            
        except:
            # Sometimes, there is too few points or skipped frames by occlussion.
            # These problems are not handled in this code.
            continue

    # Plotting
    plt.close()
    plt.figure(dpi=100, figsize=(10,10))
    plt.plot(pts[~dynamic_mask, 0], pts[~dynamic_mask, 1], 'b.', markersize=.3)

    for i in trajectories.keys():
        if dynamic_id[i]:
            pts_i = pts[cluster_ids == i]
            plt.plot(pts_i[:, 0], pts_i[:, 1], '.', markersize=.6)


    for i in trajectories.keys():
        # Plot only dynamic trajectories
        if dynamic_id[i]:
            plt.plot(trajectories[i][:,0], trajectories[i][:,1], marker='+', color='k', markersize=4, linestyle='-', linewidth=1.0)


    plt.tight_layout()
    plt.axis('equal')
    plt.title("PONE - Sequence")
    plt.savefig('assets/PONE_4DICP.png')
        