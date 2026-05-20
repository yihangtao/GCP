import argparse
import time
from itertools import count
import os
from copy import deepcopy
import cv2  
import matplotlib.pyplot as plt 
import seaborn as sns
import torch.optim as optim
from torch.utils.data import DataLoader

from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import *
from coperception.utils.loss import *
from coperception.utils.mean_ap import eval_map
from coperception.models.det import *
from coperception.models.det.LSTMAutoencoder import LSTMAE
from coperception.utils.detection_util import late_fusion, gated_late_fusion
from coperception.utils.data_util import apply_pose_noise
from coperception.utils.bac_attack import generate_bac_attack
import random
from tqdm import tqdm
from torch.autograd import Variable
from box_matching import associate_2_detections
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import numpy as np

# BH Decision Module for alternative decision making
try:
    from bh_decision import BHDecisionModule
    BH_AVAILABLE = True
except ImportError:
    BH_AVAILABLE = False
    print("Warning: BH Decision Module not available. Using threshold-based decision only.")


def get_fixed_decision_threshold(args):
    """Resolve the threshold used when BH calibration is unavailable."""
    if getattr(args, "fixed_decision_threshold", None) is not None:
        return args.fixed_decision_threshold, "manual"
    derived_threshold = 0.1 * args.reconstruction_threshold + args.box_matching_thresh
    return derived_threshold, "derived"


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True



def get_jaccard_index(config, num_agent_list, padded_voxel_point, reg_target, anchors_map, gt_max_iou, result_1, result_2):
    num_sensor = num_agent_list[0][0].numpy()
    det_results_local_1 = [[] for i in range(num_sensor)]
    annotations_local_1 = [[] for i in range(num_sensor)]
    det_results_local_2 = [[] for i in range(num_sensor)]
    annotations_local_2 = [[] for i in range(num_sensor)]
    ego_idx = args.ego_agent
    # for k in range(num_sensor):
    data_agents = {'bev_seq': torch.unsqueeze(padded_voxel_point[ego_idx, :, :, :, :], 1),
                'reg_targets': torch.unsqueeze(reg_target[ego_idx, :, :, :, :, :], 0),
                'anchors': torch.unsqueeze(anchors_map[ego_idx, :, :, :, :], 0)}
    temp = gt_max_iou[ego_idx]
    data_agents['gt_max_iou'] = temp[0]['gt_box'][0, :, :]
    result_temp_1 = result_1[ego_idx]
    result_temp_2 = result_2[ego_idx]
    temp_1 = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 'result': result_temp_1[0][0],
            'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
            'anchors_map': data_agents['anchors'].cpu().numpy()[0],
            'gt_max_iou': data_agents['gt_max_iou']}
    temp_2 = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 'result': result_temp_2[0][0],
            'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
            'anchors_map': data_agents['anchors'].cpu().numpy()[0],
            'gt_max_iou': data_agents['gt_max_iou']}
    
    det_results_local_1[ego_idx], annotations_local_1[ego_idx] = cal_local_mAP(config, temp_1, det_results_local_1[ego_idx], annotations_local_1[ego_idx])
    det_results_local_2[ego_idx], annotations_local_2[ego_idx] = cal_local_mAP(config, temp_2, det_results_local_2[ego_idx], annotations_local_2[ego_idx])
    
    print("Calculating in the view of Agent {}:".format(ego_idx))
    # shape of det_results_local_1 [k][0][0] is (N, 9)
    # The final value of the array is confidence. Ignored
    if len(det_results_local_1[ego_idx]) == 0:
        # if ego have no detection, return 0
        return 0 
    # det_1 = det_results_local_1[ego_idx][0][0][:,0:8]
    # det_2 = det_results_local_2[ego_idx][0][0][:,0:8]
    det_1 = det_results_local_1[ego_idx][0][0]
    det_2 = det_results_local_2[ego_idx][0][0]
    # jac_index = calculate_jaccard(det_results_local_1[k][0][0], det_results_local_2[k][0][0])
    jac_index, _ = associate_2_detections(det_1, det_2)
    return jac_index

def confidence_scaled_jaccard_index(config, num_agent_list, padded_voxel_point, reg_target, anchors_map, gt_max_iou, result_1, result_2):
    num_sensor = num_agent_list[0][0].numpy()
    det_results_local_1 = [[] for i in range(num_sensor)]
    annotations_local_1 = [[] for i in range(num_sensor)]
    det_results_local_2 = [[] for i in range(num_sensor)]
    annotations_local_2 = [[] for i in range(num_sensor)]
    ego_idx = args.ego_agent
    # for k in range(num_sensor):
    data_agents = {'bev_seq': torch.unsqueeze(padded_voxel_point[ego_idx, :, :, :, :], 1),
                'reg_targets': torch.unsqueeze(reg_target[ego_idx, :, :, :, :, :], 0),
                'anchors': torch.unsqueeze(anchors_map[ego_idx, :, :, :, :], 0)}
    temp = gt_max_iou[ego_idx]
    data_agents['gt_max_iou'] = temp[0]['gt_box'][0, :, :]
    result_temp_1 = result_1[ego_idx]
    result_temp_2 = result_2[ego_idx]
    temp_1 = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 'result': result_temp_1[0][0],
            'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
            'anchors_map': data_agents['anchors'].cpu().numpy()[0],
            'gt_max_iou': data_agents['gt_max_iou']}
    temp_2 = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 'result': result_temp_2[0][0],
            'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
            'anchors_map': data_agents['anchors'].cpu().numpy()[0],
            'gt_max_iou': data_agents['gt_max_iou']}
    
    det_results_local_1[ego_idx], annotations_local_1[ego_idx] = cal_local_mAP(config, temp_1, det_results_local_1[ego_idx], annotations_local_1[ego_idx])
    det_results_local_2[ego_idx], annotations_local_2[ego_idx] = cal_local_mAP(config, temp_2, det_results_local_2[ego_idx], annotations_local_2[ego_idx])
    
    print("Calculating in the view of Agent {}:".format(ego_idx))
    # shape of det_results_local_1 [k][0][0] is (N, 9)
    # The final value of the array is confidence.
    if len(det_results_local_1[ego_idx]) == 0:
        # if ego have no detection, return 0
        return 0 
    det_1 = det_results_local_1[ego_idx][0][0]
    det_2 = det_results_local_2[ego_idx][0][0]
    # jac_index = calculate_jaccard(det_results_local_1[k][0][0], det_results_local_2[k][0][0])
    _, conf_jac_index = associate_2_detections(det_1, det_2)
    return conf_jac_index

def visualize_bev_flow_matching(current_frame, box_sequences, box_losses, matched_indices, 
                              unmatched_indices, save_path, frame_seq, is_attack_frame, history_length):
    """
    Visualize BEV Flow matching trajectories, showing only center point movements
    """

    
    plt.figure(figsize=(8, 8))  # Smaller canvas size
    
    # Set axis range and grid
    plt.xlim(-10, 10)  # Smaller display range
    plt.ylim(-10, 10)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Draw all trajectories
    for i, box in enumerate(current_frame):
        if i in matched_indices:
            # Successfully matched trajectory
            seq_idx = matched_indices.index(i)
            sequence = box_sequences[seq_idx]
            
            # Collect all points in the trajectory
            trajectory_points = []
            for seq in sequence:
                center = seq[0][:2]  # Only take x,y coordinates
                trajectory_points.append(center)
            
            # Draw continuous trajectory line
            trajectory_points = np.array(trajectory_points)
            plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], 
                    '-', color='green', alpha=0.3, linewidth=1)
            
            # Draw arrows between consecutive points
            for j in range(len(trajectory_points)-1):
                current_center = trajectory_points[j]
                next_center = trajectory_points[j+1]
                
                dx = next_center[0] - current_center[0]
                dy = next_center[1] - current_center[1]
                
                # Smaller arrow size
                plt.arrow(current_center[0], current_center[1], dx, dy,
                         head_width=0.2, head_length=0.3,  # Smaller arrow size
                         fc='green', ec='green', alpha=0.5,
                         length_includes_head=True)
                
        elif i in unmatched_indices and box_sequences:
            # Unmatched trajectory, show partial match
            seq_idx = unmatched_indices.index(i)
            if seq_idx < len(box_sequences):
                partial_sequence = box_sequences[seq_idx]
                
                # Collect partial trajectory points
                trajectory_points = []
                for seq in partial_sequence:
                    center = seq[0][:2]
                    trajectory_points.append(center)
                
                # Draw partial trajectory line
                trajectory_points = np.array(trajectory_points)
                plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], 
                        '--', color='red', alpha=0.2, linewidth=1)
                
                # Draw arrows
                for j in range(len(trajectory_points)-1):
                    current_center = trajectory_points[j]
                    next_center = trajectory_points[j+1]
                    
                    dx = next_center[0] - current_center[0]
                    dy = next_center[1] - current_center[1]
                    
                    plt.arrow(current_center[0], current_center[1], dx, dy,
                             head_width=0.2, head_length=0.3,
                             fc='red', ec='red', alpha=0.3,
                             length_includes_head=True)
    
    # Add title
    plt.title(f'Frame {frame_seq} - {"Attack" if is_attack_frame else "Normal"} Frame\n'
              f'Matched: {len(matched_indices)}, Unmatched: {len(unmatched_indices)}')
    
    # Add legend
    matched_line = plt.Line2D([], [], color='green', alpha=0.5, label='Matched Trajectory')
    unmatched_line = plt.Line2D([], [], color='red', linestyle='--', alpha=0.3, label='Partial Trajectory')
    plt.legend(handles=[matched_line, unmatched_line])
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def bev_flow_generation(config, num_agent_list, padded_voxel_point, reg_target, anchors_map, gt_max_iou, result_1, result_2):
    num_sensor = num_agent_list[0][0].numpy()
    det_results_local_1 = [[] for i in range(num_sensor)]
    annotations_local_1 = [[] for i in range(num_sensor)]
    det_results_local_2 = [[] for i in range(num_sensor)]
    annotations_local_2 = [[] for i in range(num_sensor)]
    ego_idx = args.ego_agent
    # for k in range(num_sensor):
    data_agents = {'bev_seq': torch.unsqueeze(padded_voxel_point[ego_idx, :, :, :, :], 1),
                'reg_targets': torch.unsqueeze(reg_target[ego_idx, :, :, :, :, :], 0),
                'anchors': torch.unsqueeze(anchors_map[ego_idx, :, :, :, :], 0)}
    temp = gt_max_iou[ego_idx]
    data_agents['gt_max_iou'] = temp[0]['gt_box'][0, :, :]
    result_temp_1 = result_1[ego_idx]
    result_temp_2 = result_2[ego_idx]
    temp_1 = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 'result': result_temp_1[0][0],
            'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
            'anchors_map': data_agents['anchors'].cpu().numpy()[0],
            'gt_max_iou': data_agents['gt_max_iou']}
    temp_2 = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 'result': result_temp_2[0][0],
            'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
            'anchors_map': data_agents['anchors'].cpu().numpy()[0],
            'gt_max_iou': data_agents['gt_max_iou']}
    
    det_results_local_1[ego_idx], annotations_local_1[ego_idx] = cal_local_mAP(config, temp_1, det_results_local_1[ego_idx], annotations_local_1[ego_idx])
    det_results_local_2[ego_idx], annotations_local_2[ego_idx] = cal_local_mAP(config, temp_2, det_results_local_2[ego_idx], annotations_local_2[ego_idx])
    
    print("Calculating in the view of Agent {}:".format(ego_idx))
    # shape of det_results_local_1 [k][0][0] is (N, 9)
    # The final value of the array is confidence.
    if len(det_results_local_1[ego_idx]) == 0:
        # if ego have no detection, return 0
        return 0 
    det_1 = det_results_local_1[ego_idx][0][0]
    det_2 = det_results_local_2[ego_idx][0][0]
    
    # Remove confidence scores (last column), keep only coordinates
    det_1 = det_1[:, :8]  # Keep only first 8 columns (coordinates)
    det_2 = det_2[:, :8]  # Keep only first 8 columns (coordinates)

    # if args.visualization:
    #     save_path = os.path.join(args.logpath, 'bev_flow_vis')
    #     os.makedirs(save_path, exist_ok=True)
        
    #     # Assume we have these detection results
    #     clean_dets = (det_1_prev, det_1)  # BEV Flow of K frame clean
    #     attacked_dets = (det_1, det_2_attacked)  # BEV Flow of K+1 frame attacked
    #     interpolated_dets = (det_1, det_2_interpolated)  # BEV Flow after KF interpolation
    #     next_attacked_dets = (det_2_interpolated, det_3_attacked)  # BEV Flow of K+2 frame attacked
        
    #     visualize_bev_flow_sequence(
    #         clean_dets, 
    #         attacked_dets,
    #         interpolated_dets,
    #         next_attacked_dets,
    #         save_path,
    #         frame_idx=idx
    #     )
    
    return det_1, det_2



    
def visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_point, gt_max_iou, vis_tag, save_center_matrix=False):
    """
    Args:
        ... (original parameters)
        save_center_matrix: bool, whether to save detection box center point matrix
    """
    print("Visualizing: {}".format(vis_tag))
    det_results_local = [[] for i in range(6)]
    annotations_local = [[] for i in range(6)]

    padded_voxel_point = data['bev_seq']
    padded_voxel_points_teacher = data['bev_seq_teacher']
    reg_target = data['reg_targets']
    anchors_map = data['anchors']

    loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent_list[0][0])
            
    # local qualitative evaluation
    num_sensor = num_agent_list[0][0].numpy()
    print(f'num_sensor: {num_sensor}')
    for k in range(num_sensor):
        data_agents = {'bev_seq': torch.unsqueeze(padded_voxel_point[k, :, :, :, :], 1),
                    'bev_seq_teacher': torch.unsqueeze(padded_voxel_points_teacher[k, :, :, :, :], 1),
                    'reg_targets': torch.unsqueeze(reg_target[k, :, :, :, :, :], 0),
                    'anchors': torch.unsqueeze(anchors_map[k, :, :, :, :], 0)}
        temp = gt_max_iou[k]
        data_agents['gt_max_iou'] = temp[0]['gt_box'][0, :, :]
        result_temp = result[k]
        
        temp = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 
                'bev_seq_teacher': data_agents['bev_seq_teacher'][0, -1].cpu().numpy(),
                'result': result_temp[0][0],
                'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
                'anchors_map': data_agents['anchors'].cpu().numpy()[0],
                'gt_max_iou': data_agents['gt_max_iou'],
                'vis_tag': vis_tag}
        
        det_results_local[k], annotations_local[k] = cal_local_mAP(config, temp, det_results_local[k], annotations_local[k])
        print("Agent {}:".format(k))
        filename = str(filename0[0][0])
        cut = filename[filename.rfind('agent') + 7:]
        seq_name = cut[:cut.rfind('_')]
        idx = cut[cut.rfind('_') + 1:cut.rfind('/')]
        seq_save = os.path.join(save_fig_path[k], seq_name)
        check_folder(seq_save)
        idx_save = '{}_{}.png'.format(str(idx), vis_tag)

        if args.visualization:
            # Original visualization
            visualization(config, temp, None, None, 0, os.path.join(seq_save, idx_save))
            
            # New center point matrix visualization
            if save_center_matrix:
                # Create center point matrix (256x256)
                center_matrix = np.zeros((256, 256), dtype=np.float32)
                
                # Get detection results
                if len(det_results_local[k]) > 0 and len(det_results_local[k][0]) > 0:
                    detections = det_results_local[k][0][0]  # shape: (N, 9)
                    
                    for det in detections:
                        # Calculate detection box center point
                        x1, y1, x2, y2, x3, y3, x4, y4 = det[:8]
                        center_x = int((x1 + x2 + x3 + x4) / 4)
                        center_y = int((y1 + y2 + y3 + y4) / 4)
                        
                        # Ensure coordinates are within valid range
                        if 0 <= center_x < 256 and 0 <= center_y < 256:
                            center_matrix[center_y, center_x] = 1
                
                # Create save directory
                center_save_dir = os.path.join(seq_save, 'center_matrices')
                check_folder(center_save_dir)
                
                # Save binary image
                center_image_path = os.path.join(center_save_dir, f'{idx}_{vis_tag}_centers.png')
                cv2.imwrite(center_image_path, (center_matrix * 255).astype(np.uint8))
                
                # Save grayscale image
                plt.figure(figsize=(10, 10))
                plt.imshow(center_matrix, cmap='gray')
                plt.colorbar()
                plt.title(f'Center Matrix - Agent {k}')
                plt.savefig(os.path.join(center_save_dir, f'{idx}_{vis_tag}_centers_gray.png'))
                plt.close()
                
                # Save numpy matrix
                np.save(os.path.join(center_save_dir, f'{idx}_{vis_tag}_centers.npy'), center_matrix)
                
                # Save text matrix
                np.savetxt(os.path.join(center_save_dir, f'{idx}_{vis_tag}_centers.txt'), 
                          center_matrix, fmt='%d', delimiter=',')
                
                print(f"Center matrices saved in: {center_save_dir}")
    
def time_str():
        t = time.time()- 60*60*24*30
        time_string = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime(t))
        return time_string

def local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local):
    # If has RSU, do not count RSU's output into evaluation
    # eval_start_idx = 0 if args.no_cross_road else 1
    eval_start_idx = 0
    # update global result
    for k in range(eval_start_idx, num_agent):
        data_agents = {
            "bev_seq": torch.unsqueeze(padded_voxel_points[k, :, :, :, :], 1),
            "reg_targets": torch.unsqueeze(reg_target[k, :, :, :, :, :], 0),
            "anchors": torch.unsqueeze(anchors_map[k, :, :, :, :], 0),
        }
        temp = gt_max_iou[k]

        if len(temp[0]["gt_box"]) == 0:
            data_agents["gt_max_iou"] = []
        else:
            data_agents["gt_max_iou"] = temp[0]["gt_box"][0, :, :]


        result_temp = result[k]

        temp = {
            "bev_seq": data_agents["bev_seq"][0, -1].cpu().numpy(),
            "result": [] if len(result_temp) == 0 else result_temp[0][0],
            "reg_targets": data_agents["reg_targets"].cpu().numpy()[0],
            "anchors_map": data_agents["anchors"].cpu().numpy()[0],
            "gt_max_iou": data_agents["gt_max_iou"],
        }
        det_results_local[k], annotations_local[k] = cal_local_mAP(
            config, temp, det_results_local[k], annotations_local[k]
        )
    return det_results_local, annotations_local


def cal_robosac_steps(num_agent, num_consensus, num_attackers):
    # exclude ego agent
    num_agent = num_agent - 1
    eta = num_attackers / num_agent
    # print(f'eta: {eta}')
    # print(f's(num_agent): {num_agent}')
    N = np.ceil(np.log(1 - 0.99) / np.log(1 - np.power(1 - eta, num_consensus))).astype(int)
    return N

def cal_robosac_consensus(num_agent, step_budget, num_attackers):
    num_agent = num_agent - 1
    eta = num_attackers / num_agent
    s = np.floor(np.log(1-np.power(1-0.99, 1/step_budget)) / np.log(1-eta)).astype(int)
    return s


def cw_l2_attack(model, inputs, labels, device, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01) :
    # Define f-function
    def f(x) :

        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())
        
        # If targeted, optimize for making the other class most likely 
        if targeted :
            return torch.clamp(i-j, min=-kappa)
        
        # If untargeted, optimize for making the other class most likely 
        else :
            return torch.clamp(j-i, min=-kappa)
    
    w = torch.zeros_like(inputs, requires_grad=True).to(device)

    optimizer = optim.Adam([w], lr=learning_rate)

    prev = 1e10
    
    for step in range(max_iter) :

        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, inputs)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost
        
        print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')

    attack_inputs = 1/2*(nn.Tanh()(w) + 1)

    return attack_inputs


# @torch.no_grad()
# We cannot use torch.no_grad() since we need to calculate the gradient for perturbation
def main(args):
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)

    need_log = args.log
    num_workers = args.nworker
    apply_late_fusion = args.apply_late_fusion
    pose_noise = args.pose_noise
    compress_level = args.compress_level
    only_v2i = args.only_v2i
    batch_size = args.batch

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    config.inference = args.inference
    if args.bound == "upperbound":
        flag = "upperbound"
    else:
        if args.com == "when2com":
            flag = "when2com"
            if args.inference == "argmax_test":
                flag = "who2com"
            if args.warp_flag:
                flag = flag + "_warp"
        elif args.com in {"v2v", "disco", "sum", "mean", "max", "cat", "agent"}:
            flag = args.com
        else:
            flag = "lowerbound"
            if args.box_com:
                flag += "_box_com"

    print("flag", flag)
    config.flag = flag
    config.split = "test"

    num_agent = args.num_agent
    # agent0 is the cross road
    agent_idx_range = range(1, num_agent) if args.no_cross_road else range(num_agent)
    validation_dataset = V2XSimDet(
        dataset_roots=[f"{args.data}/agent{i}" for i in agent_idx_range],
        config=config,
        config_global=config_global,
        split="val",
        val=True,
        bound=args.bound,
        kd_flag=args.kd_flag,
        no_cross_road=args.no_cross_road,
    )
    validation_data_loader = DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )
    print("Validation dataset size:", len(validation_dataset))

    if args.no_cross_road:
        num_agent -= 1

    if flag == "upperbound" or flag.startswith("lowerbound"):
        model = FaFNet(
            config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent
        )
    elif flag.startswith("when2com") or flag.startswith("who2com"):
        # model = PixelwiseWeightedFusionSoftmax(config, layer=args.layer)
        model = When2com(
            config,
            layer=args.layer,
            warp_flag=args.warp_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "disco":
        model = DiscoNet(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "sum":
        model = SumFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "mean":
        model = MeanFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "max":
        model = MaxFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "cat":
        model = CatFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "agent":
        model = AgentWiseWeightedFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    else:
        model = V2VNet(
            config,
            gnn_iter_times=args.gnn_iter_times,
            layer=args.layer,
            layer_channel=256,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )

    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }

    fafmodule = FaFModule(model, model, config, optimizer, criterion, args.kd_flag)

    model_save_path = args.resume[: args.resume.rfind("/")]

    if args.inference == "argmax_test":
        model_save_path = model_save_path.replace("when2com", "who2com")

    os.makedirs(model_save_path, exist_ok=True)

    checkpoint = torch.load(
        args.resume, map_location="cpu"
    )  # We have low GPU utilization for testing
    start_epoch = checkpoint["epoch"] + 1
    fafmodule.model.load_state_dict(checkpoint["model_state_dict"])
    fafmodule.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    fafmodule.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))
    
    if args.log:
        if args.logpath:
             log_dir = args.logpath
             if not os.path.exists(log_dir):
                 os.makedirs(log_dir)
        else:
             log_dir = model_save_path
             
        log_file_name = os.path.join(log_dir, "log_epoch{}_scene{}_ego{}_{}attackers_{}_{}.txt".format(checkpoint["epoch"], args.scene_id, args.ego_agent, args.number_of_attackers, args.gcp, time_str()))
        saver = open(log_file_name, "a")
        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()

    def print_and_write_log(log_str):
        print(log_str)
        if args.log:
            saver.write(log_str + "\n")
            saver.flush()

    #  ===== eval =====
    fafmodule.model.eval()
    save_fig_path = [
        check_folder(os.path.join(model_save_path, f"vis{i}")) for i in agent_idx_range
    ]
    tracking_path = [
        check_folder(os.path.join(model_save_path, f"tracking{i}"))
        for i in agent_idx_range
    ]

    # for local and global mAP evaluation
    det_results_local = [[] for i in agent_idx_range]
    annotations_local = [[] for i in agent_idx_range]
    

    for k, v in fafmodule.model.named_parameters():
        v.requires_grad = False  # fix parameters



    assert args.gcp in ["upperbound", "lowerbound", "no_defense", "robosac_validation", "robosac_mAP", "adaptive", "fix_attackers", "performance_eval", "probing", "made", "gcp", "test", "gated_late_fusion"]



    # NOTE: ONLY SUPPORT SINGLE SCENE BY NOW
    frame_count = 100
    # array for robosac total steps
    steps = np.zeros(frame_count)
    # array for ego prediction count
    ego_steps = np.zeros(frame_count)
    fpss = np.zeros(frame_count)

    # array for consensus set sizes(for adaptive sampling)
    consensus_set_sizes = np.zeros(frame_count)

    # start from select 1 collab agent(for adaptive sampling)
    # keep it out of the loop for not initializing every time
    consensus_set_size = 1
    
    # cnt for adaptive sampling steps from frame 0 
    total_adaptive_steps = 0

    # once failed, need a flag to record(for adaptive sampling)
    failed_once = False
    
    # for probing
    N_th_frame_of_each_estimation = [-1] * 5
    # TODO: set ratios as input
    estimate_attacker_ratio = [0.0, 0.2, 0.4, 0.6, 0.8]
    estimated_attacker_ratio = 1.0
    
    consensus_tries = [5,4,3,2,1]
    consensus_tries_is_needed = [1,1,1,1,1]
    # probing_step_limit_by_attacker_ratio
    NMax = []
    for ratio in estimate_attacker_ratio:
        # TODO: set 5 to a variable
        temp_num_attackers = round(5 * (ratio))
        temp_num_consensus = 5 - temp_num_attackers
        NMax.append(cal_robosac_steps(num_agent, temp_num_consensus, temp_num_attackers))
    
    # Special case when assuming all agents are benign.(i.e. attacker ratio = 1.0)
    # means once if we can't test consensus in 1 try, there's definitely at least 1 attacker.
    NMax[0] = 1
    # print("NMax:", NMax)
    # {5: 1, 4: 9, 3: 19, 2: 27, 1: 21}
    NTry = [0] * len(estimate_attacker_ratio)
    total_sampling_step =0

    # succ count for robosac eval
    succ = 0 
    partial_succ = 0
    fail = 0
    # counters for relative frame in a single scene
    frame_seq = 0

    fix_attackers_generated = False
    fix_attackers_collab_agent_list = []
    fix_attackers_total_step = 0

    interpolation_counts = {}

    def generate_adv_array(length=100, alpha=0.5, mode='Random'):
        num_infected = int(length * alpha)  # Determine number of infected frames
        infection_array = np.zeros(length, dtype=int)  # Create zero array of given length
        
        if mode == 'Random':
            # Random mode: randomly select positions to set as 1
            infected_indices = np.random.choice(length, num_infected, replace=False)
            infection_array[infected_indices] = 1
        
        elif mode == 'Poission':
            # Poisson distribution mode: use Poisson process to distribute 1s
            # Note: Poisson distribution models event occurrence; here we use simplified version
            rates = np.random.poisson(lam=1, size=length)
            sorted_indices = np.argsort(rates)[::-1]  # Sort by Poisson rates, select top num_infected
            top_indices = sorted_indices[:num_infected]
            infection_array[top_indices] = 1
            
        elif mode == 'SI':
            # SI model distribution: simulate disease spreading, starting with one infected node
            infected_index = np.random.randint(length)
            infection_array[infected_index] = 1
            for _ in range(num_infected - 1):
                while True:
                    neighbor = infected_index + np.random.choice([-1, 1])  # Randomly choose left or right neighbor
                    if 0 <= neighbor < length:
                        infected_index = neighbor
                        if infection_array[infected_index] == 0:
                            infection_array[infected_index] = 1
                            break
        elif mode == 'Demo':
            # Demo mode: first 5 frames are 0, frames 6-7 are 1, rest are 0
            infection_array[6:8] = 1  # Set indices 5 and 6 (frames 6-7) to 1
            
        return infection_array
    
    adv_array = generate_adv_array(length=frame_count, alpha=args.attack_ratio, mode=args.attack_mode)
    
    for cnt, sample in enumerate(tqdm(validation_data_loader)):

        t = time.time()
        (
            padded_voxel_point_list,
            padded_voxel_points_teacher_list,
            label_one_hot_list,
            reg_target_list,
            reg_loss_mask_list,
            anchors_map_list,
            vis_maps_list,
            gt_max_iou,
            filenames,
            target_agent_id_list,
            num_agent_list,
            trans_matrices_list,
        ) = zip(*sample)

        filename0 = filenames[0]
        filename = str(filename0[0][0])
        cut = filename[filename.rfind('agent') + 7:]
        seq_name = cut[:cut.rfind('_')]
        idx = cut[cut.rfind('_') + 1:cut.rfind('/')]
        

        if (int(seq_name) not in args.scene_id):
            continue
        
        # Frame range control
        frame_idx = int(idx)
        if (args.sample_start is not None):
            if frame_idx < args.sample_start:
                continue
        
        if (args.sample_end is not None):
            if frame_idx > args.sample_end:
                continue
        
        # Legacy support for --sample_id (same as --sample_start)
        if (args.sample_id is not None):
            if frame_idx < args.sample_id:
                continue

        frame_seq += 1
        print_and_write_log("\nScene {}, Frame {}:".format(seq_name, idx))
        trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
        target_agent_ids = torch.stack(tuple(target_agent_id_list), 1)
        num_all_agents = torch.stack(tuple(num_agent_list), 1)


        if args.no_cross_road:
            num_all_agents -= 1
        padded_voxel_points = torch.cat(tuple(padded_voxel_point_list), 0)
        padded_voxel_points_teacher = torch.cat(tuple(padded_voxel_points_teacher_list), 0)

        label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
        reg_target = torch.cat(tuple(reg_target_list), 0)
        reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
        anchors_map = torch.cat(tuple(anchors_map_list), 0)
        vis_maps = torch.cat(tuple(vis_maps_list), 0)

        data = {
            "bev_seq": padded_voxel_points.to(device),
            "bev_seq_teacher": padded_voxel_points_teacher.to(device),
            "labels": label_one_hot.to(device),
            "reg_targets": reg_target.to(device),
            "anchors": anchors_map.to(device),
            "vis_maps": vis_maps.to(device),
            "reg_loss_mask": reg_loss_mask.to(device).type(dtype=torch.bool),
            "target_agent_ids": target_agent_ids.to(device),
            "num_agent": num_all_agents.to(device),
            'ego_agent': args.ego_agent,
            'pert': None,
            'no_fuse': False,
            'collab_agent_list': None,
            'trial_agent_id': None,
            'confidence': None,
            'unadv_pert': None,
            'attacker_list' : None,
            'eps': None,
            "trans_matrices": trans_matrices.to(device),
        }

        if args.gcp == "performance_eval":
            fafmodule.cal_forward_time(data, 1)
            continue


        # STEP 1:
        # get original ego agent class prediction of all anchors, without adv pert and fuse, return cls pred of all agents
        cls_result  = fafmodule.cls_predict(data, batch_size, no_fuse=True)
        # change logits to one-hot
        mean = torch.mean(cls_result, dim=2)
        cls_result[:,:,0] = cls_result[:,:,0] > mean
        cls_result[:,:,1] = cls_result[:,:,1] > mean
        pseudo_gt = cls_result.clone().detach()
        # torch.Size([6, 393216, 2])

        # if args.visualization:
        #     # visulize ego only det result, without fusion
        #     data['no_fuse'] = True
        #     visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='ego_only')
        #     # visulize original fusion result
        #     data['no_fuse'] = False
        #     visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='original_fusion')
            

        if args.gcp == 'upperbound':
            # no attacker is attacking and all agents are in collaboration, everything is just fine
            data['pert'] = None
            if args.partial_upperbound:
                # Sometimes we need to eval partially colloborated agents
                num_sensor = num_agent_list[0][0]
                ego_idx = args.ego_agent
                all_agent_list = [i for i in range(num_sensor)]
                all_agent_list.remove(ego_idx)
                collab_agent_list = random.sample(all_agent_list, k=args.robosac_k)
                data['collab_agent_list'] = collab_agent_list
                print_and_write_log("\nPartial upperbound, collab agent list: {}".format(collab_agent_list))
            else:
                data['collab_agent_list'] = None
            data['trial_agent_id'] = None
            data['no_fuse'] = False
            loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
            if args.visualization:
                visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='upperbound')
            det_results_local, annotations_local = local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)
            continue

        elif args.gcp == 'lowerbound':
            # Suppose all neighboring agents are malicious, and only the ego agent is trusted
            # Each agent only use its own features to perform object detection
            data['pert'] = None
            data['collab_agent_list'] = None
            data['trial_agent_id'] = None
            data['no_fuse'] = True
            loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
            if args.visualization:
                # visualize attacked result
                visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='lowerbound')
            det_results_local, annotations_local = local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)
            continue
        
        else:
            # There are attackers among us: 
            # Define ego agent and sensor info for attack generation
            num_sensor = num_agent_list[0][0]
            ego_idx = args.ego_agent
            all_agent_list = [i for i in range(num_sensor)]
            all_agent_list.remove(ego_idx)
            
            # Define attacker list
            if args.gcp == 'fix_attackers' or args.fix_attackers:
                if not fix_attackers_generated:
                    if args.number_of_attackers == 2:
                        attacker_list = [0, 2]
                    else:
                        attacker_list = random.sample(all_agent_list, k=args.number_of_attackers)
                    fix_attackers_generated = True
            else:
                if args.number_of_attackers == 0:
                    attacker_list = []
                elif args.number_of_attackers == 2:
                    attacker_list = [0, 2]
                else:
                    attacker_list = random.sample(all_agent_list, k=args.number_of_attackers)

            # STEP 2:
            # generate adv perturb
            if len(attacker_list) > 0:
                if args.adv_method == 'bac':
                    # BAC Attack: Generate perturbation with blind region segmentation
                    print_and_write_log("[BAC Attack] Generating BAC perturbation with blind region targeting...")
                    
                    # Use BAC attack module
                    try:
                        pert, blind_mask_np = generate_bac_attack(
                            fafmodule=fafmodule,
                            data=data,
                            num_agent=num_agent,
                            ego_agent=ego_idx,
                            attacker_list=attacker_list,
                            num_iterations=args.adv_iter,
                            alpha=args.pert_alpha,
                            eps=args.eps,
                            device=device,
                            config=config,
                            reg_target=reg_target,
                            anchors_map=anchors_map,
                            gt_max_iou=gt_max_iou,
                            padded_voxel_points=padded_voxel_points,
                            scene_id=int(seq_name),
                            frame_id=int(idx),  # Use scene-internal frame index, not global frame_seq
                            mask_update_rate=10  # 0.5 FPS slow update
                        )
                        
                        blind_mask = torch.from_numpy(blind_mask_np).float().to(device)
                        print_and_write_log(f"[BAC] Blind area ratio: {1 - blind_mask_np.mean():.2%}")
                    except Exception as e:
                        print_and_write_log(f"[BAC] Error generating BAC attack: {e}")
                        print_and_write_log("[BAC] Falling back to standard PGD...")
                        pert = torch.randn(6, 256, 32, 32) * 0.1
                        blind_mask = torch.ones(6, 32, 32)
                    
                elif args.adv_method == 'simple_bac':
                    # Simple BAC: Output-space weighting (original BSI implementation)
                    print_and_write_log("[Simple BAC] Using output-space weighting attack...")
                    pert = torch.randn(6, 256, 32, 32) * 0.1
                    blind_mask = torch.ones(6, 32, 32)  # No blind mask needed for simple version
                    
                elif args.adv_method == 'pgd':
                    # PGD random init   
                    pert = torch.randn(6, 256, 32, 32) * 0.1
                    blind_mask = torch.ones(6, 32, 32)  # No blind mask for PGD
                elif args.adv_method == 'bim' or args.adv_method == 'cw-l2':
                    # BIM/CW-L2 zero init
                    pert = torch.zeros(6, 256, 32, 32)
                    blind_mask = torch.ones(6, 32, 32)  # No blind mask for BIM/CW
                else:
                    raise NotImplementedError
            else:
                pert = torch.zeros(6, 256, 32, 32)
                blind_mask = torch.ones(6, 32, 32)

            # attacker_list already defined above before BAC attack generation
            data['attacker_list'] = attacker_list
            data['eps'] = args.eps
            data['no_fuse'] = False
            data['bac_eps'] = args.bac_eps
            data['simple_bac_eps'] = args.simple_bac_eps

            # adv_start_time = time.time()

            # For standard attacks (not full BAC), run PGD iterations
            if len(attacker_list) > 0 and args.adv_method != 'bac':
                for i in range(args.adv_iter):
                    pert.requires_grad = True
                    # Introduce adv perturbation
                    data['pert'] = pert.to(device)
                            
                    # STEP 3: Use inverted classification ground truth, minimze loss wrt inverted gt, to generate adv attacks based on cls(only)
                    # NOTE: Actual ground truth is not always available especially in real-world attacks
                    # We define the adversarial loss of the perturbed output with respect to an unperturbed output pseudo_gt instead of the ground truth
                    cls_loss = fafmodule.cls_step(data, batch_size, ego_loss_only=args.ego_loss_only, ego_agent=args.ego_agent, invert_gt=True, self_result=pseudo_gt, adv_method=args.adv_method)

                    pert = pert + args.pert_alpha * pert.grad.sign() * -1
                    pert.detach_()
            # For BAC attack, perturbation is already optimized in generate_bac_attack()
            
            # # Calculate time
            # adv_time = time.time() - adv_start_time
            # print_and_write_log(f"Time taken to generate adversarial perturbation: {adv_time:.4f} seconds")
            
            # Detach and clone perturbations from Pytorch computation graph, in case of gradient misuse.
            if adv_array[cnt % 100] == 0:
                pert = torch.zeros(6, 256, 32, 32)
                print_and_write_log("No perturbation is applied")
            else:
                pert = pert.detach().clone()
                if len(attacker_list) > 0:
                    print_and_write_log("Perturbation is applied on agent {}".format(attacker_list))
                else:
                    print_and_write_log("No perturbation is applied (no attackers specified)")
            # Apply the final perturbation to attackers' feature maps.
            data['pert'] = pert.to(device)
    
            if args.visualization:
                # visualize attacked result
                visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='attacked_fusion')

            if args.gcp == 'no_defense':
                # attacker is always attacking and no defense is applied
                data['pert'] = pert.to(device)
                data['no_fuse'] = False
                loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
                det_results_local, annotations_local = local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)
                continue
            

            if args.use_history_frame == True:
                # use history frame to save one forward pass
                if int(idx) == 0:
                    # first frame, use current ego only result as reference result
                    print_and_write_log("first frame, use current ego only result as reference result")
                    data['pert'] = None
                    data['collab_agent_list'] = None
                    data['no_fuse'] = True
                    _, _, _, result_reference = fafmodule.predict_all(data, 1, num_agent=num_agent)
                    ego_steps[frame_seq-1] = 1
                # if not first frame, keep no-op since we use history frame and it will be updated at the end of the iteration
            else:
                # if not use history frame, use current frame as reference frame
                # Get the original(ego_only) prediction
                print_and_write_log("performing calculating ego only result...")
                data['pert'] = None
                data['collab_agent_list'] = None
                data['no_fuse'] = True
                _, _, _, result_reference = fafmodule.predict_all(data, 1, num_agent=num_agent)


            if args.gcp == 'fix_attackers':
                # Assume attacker_list is fixed and always attacking in the scene, then after reached consensus, omit sampling process
                num_sensor = num_agent_list[0][0]
                ego_idx = args.ego_agent
                all_agent_list = [i for i in range(num_sensor)]
                # We always trust ourself
                all_agent_list.remove(ego_idx)
                # Not including ego agent, since ego agent is always used.
                
                if fix_attackers_collab_agent_list == []:
                    # if consensus is not reached, keep sampling attackers
                    collab_agent_list = []
                    if args.robosac_k == None:
                        consensus_set_size = cal_robosac_consensus(
                            num_agent, args.step_budget, args.number_of_attackers)

                        # print_and_write_log("\nStep Budget {}, Calculated Consensus Set Size {}:".format(
                        #     args.step_budget, consensus_set_size))

                        if(consensus_set_size < 1):
                            print_and_write_log(
                                'Expected Consensus Agent below 1. Exit.'.format(consensus_set_size))
                            sys.exit()
                    found = False
                    # NOTE: 1~step_budget-1
                    for step in range(1, args.step_budget + 1):
                        # NOTE: random.choices will sample an agent more than once. eg.: [2, 3, 2]
                        # So we should use random.sample(population, k) to avoid this.
                        # collab_agent_list = random.sample(all_agent_list, k=args.robosac_k)
                        fix_attackers_total_step += 1
                        # print_and_write_log("\nScene {}, Frame {}, Step {}, Step Budget {}:".format(
                        # seq_name, idx, step, args.step_budget))

                        if args.robosac_k == None:
                            collab_agent_list = random.sample(
                                all_agent_list, k=consensus_set_size)
                        else:
                            collab_agent_list = random.sample(
                                all_agent_list, k=args.robosac_k)
                        data['collab_agent_list'] = collab_agent_list
                        data['no_fuse'] = False
                        data['pert'] = pert.to(device)

                        loss, cls_loss, loc_loss, result = fafmodule.predict_all(
                            data, 1, num_agent=num_agent)

                        # We use jaccard index to define the difference between two bbox sets
                        jac_index = get_jaccard_index(
                            config, num_agent_list, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result_reference, result)
                        print_and_write_log(
                            "Jaccard Coefficient: {}".format(jac_index))
                        if jac_index < args.box_matching_thresh:
                            print_and_write_log(
                                'Attacker(s) is(are) among {}'.format(collab_agent_list))
                        else:
                            sus_agent_list = [
                                i for i in all_agent_list if i not in collab_agent_list]
                            print_and_write_log('Achieved consensus at step {}, with agents {}. Attacker(s) is(are) among {}, excluded'.format(
                                step, collab_agent_list, sus_agent_list))
                            print_and_write_log('Now begin to keep collaborating with agents {}'.format(collab_agent_list))
                            
                            found = True
                            # reached consensus, break
                            fix_attackers_collab_agent_list = collab_agent_list
                            steps[frame_seq - 1] = step
                            succ += 1
                            
                            break

                    if not found:
                        print_and_write_log('No consensus!')
                        # Can't achieve consensus, so fall back to original ego only result
                        data['pert'] = None
                        data['collab_agent_list'] = None
                        data['no_fuse'] = True
                        _, _, _, result_self_only = fafmodule.predict_all(
                            data, 1, num_agent=num_agent)
                        result = result_self_only
                        steps[frame_seq - 1] = args.step_budget
                        fail += 1

                    if args.use_history_frame == True:
                        # update reference frame for next iteration
                        print_and_write_log("update frame {} result as reference frame result for the next frame".format(idx))
                        result_reference = result
                    else:
                        ego_steps[frame_seq - 1] = 1
                else: 
                    print_and_write_log("\nfound consensus, use fixed collaborator:{}".format(fix_attackers_collab_agent_list))
                    # found consensus, use fixed collaborator
                    data['collab_agent_list'] = fix_attackers_collab_agent_list
                    data['no_fuse'] = False
                    data['pert'] = pert.to(device)
                    steps[frame_seq - 1] = 1
                    loss, cls_loss, loc_loss, result = fafmodule.predict_all(
                        data, 1, num_agent=num_agent)
                    if args.visualization:
                        # visualize consensus result
                        visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='consensus')
                    

                # save the step num for current frame, Then calculate mean steps over the scene.

                det_results_local, annotations_local = local_eval(
                    num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)


            if args.gcp == "probing" :
                
                step = 0
                succ_result = None
                succ_probing_consensus_size = 0

                #TODO: set 5 to a variable
                assert args.step_budget >= 5 #ensuring probing tries will traverse all possible attacker ratios



                while step < args.step_budget and NTry < NMax:
                    # for consensus_set_size in consensus_tries:
                    #     # probe attackers
                    #     temp_num_attackers = (5-consensus_set_size)
                    #     temp_attacker_ratio = temp_num_attackers / 5
                    for i in range(len(estimate_attacker_ratio)):
                        temp_attacker_ratio = estimate_attacker_ratio[i]
                        consensus_set_size = round(5*(1-temp_attacker_ratio))
                        if NTry[i] < NMax[i]:
                            print_and_write_log("Probing {} agents for consensus".format(consensus_set_size))
                            step += 1
                            total_sampling_step += 1
                            # probing_step_tried_by_consensus_set_size[consensus_set_size] += 1
                            # step budget available for probing
                            # try to probe attacker ratio
                            collab_agent_list = random.sample(
                            all_agent_list, k=consensus_set_size)
                            data['collab_agent_list'] = collab_agent_list
                            data['no_fuse'] = False
                            data['pert'] = pert.to(device)

                            loss, cls_loss, loc_loss, result = fafmodule.predict_all(
                                data, 1, num_agent=num_agent)
                            
                            # We use jaccard index to define the difference between two bbox sets
                            jac_index = get_jaccard_index(
                                config, num_agent_list, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result_reference, result)
                            print_and_write_log(
                                "Jaccard Coefficient: {}".format(jac_index))

                            if jac_index < args.box_matching_thresh:
                                # fail to reach consensus
                                print_and_write_log('No consensus reached when probing {} consensus agents. Current step is {} in Frame {}.'.format(consensus_set_size,step,idx))
                                print_and_write_log('Attacker(s) is(are) among {}'.format(collab_agent_list))

                                NTry[i] += 1 
                                
                                # if temp_num_attackers == 0:
                                #     # Assumption of no attackers fails
                                #     consensus_tries_is_needed[i] = 0

                                if NTry[i] == NMax[i]:
                                    print_and_write_log("Probing of {} agents for consensus has reached its sampling limit {} with assumed attacker ratio {} and consensus set size {}.".format(consensus_set_size, NMax[i], temp_attacker_ratio, consensus_set_size))
                                    print_and_write_log("From now on we won't try to probe {} agents consensus since it seems unlikely to reach that.".format(consensus_set_size))
                            else:
                                # succeed to reach consensus
                                sus_agent_list = [
                                    i for i in all_agent_list if i not in collab_agent_list]
                                print_and_write_log('Achieved consensus at step {} in Frame{}, with {} agents: {}. Using the result as temporal final output of this frame, and skipping smaller consensus set tries. \n Attacker(s) is(are) among {}, excluded.'.format(
                                    step, idx, consensus_set_size, collab_agent_list, sus_agent_list))
                                
                                succ_result = result
                                succ_probing_consensus_size = consensus_set_size
                                
                                if temp_attacker_ratio < estimated_attacker_ratio:
                                    print_and_write_log('Larger consensus set ({} agents) probed. We will skip all the smaller consensus set tries. Update attacker ratio estimation to {}'.format(consensus_set_size, temp_attacker_ratio))
                                    estimated_attacker_ratio = temp_attacker_ratio
                                    # Record probing frame
                                    N_th_frame_of_each_estimation[i] = idx
                                    
                                    for j in range(i, len(estimate_attacker_ratio)):
                                        # set all the larger attacker ratio to 0
                                        NTry[j] = NMax[j]

                                    break                                    



            elif args.gcp == 'robosac_mAP': #Needs Evaluation                            
                # Given Step Budget N and Sampling Set Size s, perform predictions

                num_sensor = num_agent_list[0][0]
                ego_idx = args.ego_agent
                all_agent_list = [i for i in range(num_sensor)]
                # We always trust ourself
                all_agent_list.remove(ego_idx)
                # Not including ego agent, since ego agent is always used.
                collab_agent_list = []

                if args.robosac_k == None:
                    consensus_set_size = cal_robosac_consensus(num_agent, args.step_budget, args.number_of_attackers)

                    print_and_write_log("\nStep Budget {}, Calculated Consensus Set Size {}:".format(args.step_budget, consensus_set_size))

                    if(consensus_set_size < 1):
                        print_and_write_log('Expected Consensus Agent below 1. Exit.'.format(consensus_set_size))
                        sys.exit()

                found = False
                # NOTE: 0~step_budget-1
                for step in range(1, args.step_budget + 1):
                    # NOTE: random.choices will sample an agent more than once. eg.: [2, 3, 2]
                    # So we should use random.sample(population, k) to avoid this.
                    # collab_agent_list = random.sample(all_agent_list, k=args.robosac_k)
                    if args.robosac_k == None:
                        collab_agent_list = random.sample(all_agent_list, k=consensus_set_size)
                    else:
                        collab_agent_list = random.sample(all_agent_list, k=args.robosac_k)
                    data['collab_agent_list'] = collab_agent_list
                    data['no_fuse'] = False
                    data['pert'] = pert.to(device)

                    loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)

                    # We use jaccard index to define the difference between two bbox sets
                    jac_index = get_jaccard_index(config, num_agent_list, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result_reference, result)
                    print_and_write_log("Jaccard Coefficient: {}".format(jac_index))
                    if jac_index < args.box_matching_thresh:
                        print_and_write_log('Attacker(s) is(are) among {}'.format(collab_agent_list))
                    else:
                        sus_agent_list = [i for i in all_agent_list if i not in collab_agent_list]
                        print_and_write_log('Achieved consensus at step {}, with agents {}. Attacker(s) is(are) among {}, excluded'.format(step, collab_agent_list, sus_agent_list))
                        found = True
                        steps[frame_seq-1] = step
                        succ += 1
                        if args.visualization:
                            # visualize consensus result
                            visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='consensus')
                        break

                if not found:
                    print_and_write_log('No consensus!')
                    # Can't achieve consensus, so fall back to original ego only result
                    data['pert'] = None
                    data['collab_agent_list'] = None
                    data['no_fuse'] = True
                    _, _, _, result_self_only = fafmodule.predict_all(data, 1, num_agent=num_agent)
                    result = result_self_only
                    steps[frame_seq-1] = args.step_budget
                    ego_steps[frame_seq-1] = 1
                    fail += 1
                
                if args.use_history_frame == True:
                    # update reference frame for next iteration
                    print_and_write_log("update frame {} result as reference frame result for the next frame".format(idx))
                    result_reference = result
                else:
                    ego_steps[frame_seq - 1] = 1
                    

                det_results_local, annotations_local = local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)
    
            elif args.gcp == 'made':
                num_sensor = num_agent_list[0][0]
                ego_idx = args.ego_agent
                all_agent_list = [i for i in range(num_sensor)]
                # We always trust ourself
                all_agent_list.remove(ego_idx)
                # Not including ego agent, since ego agent is always used.
                collab_agent_list = []
                for agent_idx in all_agent_list:
                    data['collab_agent_list'] = [agent_idx]
                    data['no_fuse'] = False
                    data['pert'] = pert.to(device)
                    loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
                    # We use jaccard index to approximate the matching loss used in MADE
                    agent_start_time = time.time()
                    jac_index = get_jaccard_index(config, num_agent_list, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result_reference, result)
                    print_and_write_log("Jaccard Coefficient: {}".format(jac_index))
                    if jac_index < args.box_matching_thresh:
                        print_and_write_log('agent {} is an attacker'.format(agent_idx))
                    else:
                        print_and_write_log('agent {} is trusted'.format(agent_idx))
                        collab_agent_list.append(agent_idx)
                    agent_time = time.time() - agent_start_time
                    print_and_write_log(f'Agent {agent_idx} processing time: {agent_time:.4f} seconds')
                   
                   
                    
                if len(collab_agent_list) > 0:
                    # Use all trusted agents for final prediction
                    data['collab_agent_list'] = collab_agent_list 
                    data['no_fuse'] = False
                    data['pert'] = pert.to(device)
                    loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
                else:
                    # If no trusted agents, use ego only result
                    data['pert'] = None
                    data['collab_agent_list'] = None
                    data['no_fuse'] = True
                    loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
                
                # Update detection results
                det_results_local, annotations_local = local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)
                
                if args.visualization:
                    # visualize consensus result
                    visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='consensus')
            
            elif args.gcp == 'gcp':
                num_sensor = num_agent_list[0][0]
                ego_idx = args.ego_agent
                all_agent_list = [i for i in range(num_sensor)]
                # We always trust ourself
                all_agent_list.remove(ego_idx)
                # Not including ego agent, since ego agent is always used.
                collab_agent_list = []
                
                # Initialize LSTM-AE model and related components
                if not hasattr(args, 'lstm_ae'):
                    args.lstm_ae = LSTMAE(
                        input_dim=8,  # 8 coordinate values
                        hidden_dim=32,
                        num_layers=2,
                        seq_length=args.history_length
                    ).to(device)
                    
                    # Load pre-trained model
                    model_path = "/data2/user2/yihang/GCP/coperception/logs/model/best_model.pth"
                    if os.path.exists(model_path):
                        checkpoint = torch.load(model_path, map_location=device)
                        args.lstm_ae.load_state_dict(checkpoint['model_state_dict'])
                        # args.reconstruction_threshold = checkpoint.get('threshold', 0.1)
                        print_and_write_log(f"Loaded pre-trained LSTM-AE model: {model_path}")
                    else:
                        print_and_write_log("Pre-trained model not found, please train model first")
                        return
                        
                    args.lstm_ae.eval()
                
                # Initialize history frame cache once so temporal verification can
                # accumulate a full window across consecutive frames.
                if not hasattr(args, 'history_frames'):
                    args.history_frames = {}
                    args.history_scene_id = None

                current_scene_id = int(seq_name)
                if args.history_scene_id != current_scene_id:
                    args.history_frames = {}
                    args.history_scene_id = current_scene_id
                    print_and_write_log(
                        f"Reset temporal history cache for scene {current_scene_id}"
                    )
                
                # Initialize BH Decision Module if requested
                if args.use_bh and BH_AVAILABLE:
                    if not hasattr(args, 'bh_module'):
                        args.bh_module = BHDecisionModule(
                            alpha_bh=args.alpha_bh,
                            calibration_path=args.calibration_path
                        )
                        fixed_threshold, threshold_source = get_fixed_decision_threshold(args)
                        if args.bh_module.has_calibration():
                            print_and_write_log(f"Initialized BH Decision Module: alpha_bh={args.alpha_bh}")
                            print_and_write_log(f"  Calibration stats: {args.bh_module.get_stats()}")
                        else:
                            print_and_write_log(
                                "BH calibration is unavailable. "
                                f"Falling back to {threshold_source} threshold={fixed_threshold:.4f}"
                            )
                
                for agent_idx in all_agent_list:
                    # Spatial consistency check
                    data['collab_agent_list'] = [agent_idx]
                    data['no_fuse'] = False
                    data['pert'] = pert.to(device)
                    loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
                    
                    # Calculate spatial consistency score
                    t_spatial_start = time.time()
                    jac_index = get_jaccard_index(
                        config, num_agent_list, padded_voxel_points, reg_target, 
                        anchors_map, gt_max_iou, result_reference, result
                    )
                    t_spatial_end = time.time()
                    print_and_write_log(f"Module Runtime - Spatial Check Calculation (CSCLoss) (Agent {agent_idx}): {t_spatial_end - t_spatial_start:.6f}s")
                    print_and_write_log(f"Confidence-weighted Jaccard index: {jac_index}")
                    
                    # if jac_index < args.box_matching_thresh:
                    #     print_and_write_log(f'Agent {agent_idx} failed spatial check, using previous frame result')
                        
                        # # Ensure history frame cache is not empty
                        # if agent_idx in args.history_frames and len(args.history_frames[agent_idx]) > 0:
                        #     # Copy previous frame result
                        #     last_frame = args.history_frames[agent_idx][-1]
                        #     args.history_frames[agent_idx].append(last_frame.copy())
                        #     print_and_write_log(f'Copied previous frame result to current frame cache')
                        # else:
                        #     print_and_write_log(f'Agent {agent_idx} has no history frame record, skip current frame')
                        # continue
                        
                    # print_and_write_log(f'Agent {agent_idx} passed spatial check')
                    
                    # Generate BEV flow and perform temporal check
                    t_bev_start = time.time()
                    reference_bev_flow, current_bev_flow = bev_flow_generation(
                        config, num_agent_list, padded_voxel_points, reg_target, 
                        anchors_map, gt_max_iou, result_reference, result
                    )
                    t_bev_end = time.time()
                    print_and_write_log(f"Module Runtime - BEV Flow Generation (Agent {agent_idx}): {t_bev_end - t_bev_start:.6f}s")
                    
                    # Add current frame to history cache
                    if agent_idx not in args.history_frames:
                        args.history_frames[agent_idx] = []
                    # args.history_frames[agent_idx].append(current_bev_flow)

                    # If cache not full, continue to next agent
                    if len(args.history_frames[agent_idx]) < args.history_length:
                        if jac_index < args.box_matching_thresh:
                            print_and_write_log(f'Agent {agent_idx} failed spatial check, using previous frame result')
                            
                            # Ensure history frame cache is not empty
                            if agent_idx in args.history_frames and len(args.history_frames[agent_idx]) > 0:
                                # Copy previous frame result
                                last_frame = args.history_frames[agent_idx][-1]
                                args.history_frames[agent_idx].append(last_frame.copy())
                                print_and_write_log(f'Copied previous frame result to current frame cache')
                            else:
                                print_and_write_log(f'Agent {agent_idx} has no history frame record, skip current frame')
                        else:
                            args.history_frames[agent_idx].append(current_bev_flow)
                            cache_size = len(args.history_frames[agent_idx])
                            print_and_write_log(
                                f'Agent {agent_idx} passed spatial check, current frame added to history frame cache '
                                f'({cache_size}/{args.history_length})'
                            )
                            print_and_write_log(
                                f'Agent {agent_idx} temporal check skipped until cache is full'
                            )

                        continue

                    # Cache is full, perform temporal check on each box in current frame
                    print_and_write_log(
                        f'Agent {agent_idx} temporal history cache is full '
                        f'({len(args.history_frames[agent_idx])}/{args.history_length}), start temporal check'
                    )
                    history_frames = args.history_frames[agent_idx][-args.history_length:]
                    current_frame = history_frames[-1]
                    
                    total_loss = 0
                    matched_boxes = 0

                    # Prepare to save reconstruction loss file
                    if args.attack_mode == 'Demo':
                        filename = str(filename0[0][0])
                        cut = filename[filename.rfind('agent') + 7:]
                        seq_name = cut[:cut.rfind('_')]
                        idx = cut[cut.rfind('_') + 1:cut.rfind('/')]
                        
                        # Create save path
                        seq_save = os.path.join(save_fig_path[agent_idx], seq_name)
                        check_folder(seq_save)
                        
                        # Generate loss file name
                        loss_save_path = os.path.join(
                            seq_save,
                            f'reconstruction_loss_agent{agent_idx}_frame_{idx}.txt'
                        )
                        
                        with open(loss_save_path, 'w') as f:
                            f.write(f"Frame {frame_seq} - {'Attack' if adv_array[frame_seq-1] else 'Normal'} Frame\n")
                    
                    # Process each box in current frame
                    total_chain_loss = 0  # Total loss of all matched chains
                    matched_boxes = 0  # Number of successfully matched boxes
                    unmatched_boxes = 0  # Number of unmatched boxes
                    t_lstm_total = 0

                    
                    for box_idx in range(len(current_frame)):
                        current_box = current_frame[box_idx:box_idx+1]  # Keep 2D shape
                        box_sequence = []
                        box_sequence.append(current_box)
                        frame_losses = []  # Store reconstruction loss for each step of current box
                        
                        # Chain matching: start from current frame, match with previous frames sequentially
                        prev_box = current_box  # Intermediate box for chain matching
                        matched_success = True
                        chain_loss = 0  # Loss of current matching chain
                        
                        # Traverse history frames (from most recent to oldest)
                        for frame_idx in range(len(history_frames)-2, -1, -1):
                            prev_frame = history_frames[frame_idx]
                            best_match, best_iou = find_best_match(prev_box, prev_frame)
                            
                            if best_iou > args.box_matching_thresh:
                                box_sequence.insert(0, best_match)
                                prev_box = best_match  # Update matched box for next matching
                                
                                # If sequence length is greater than 1, calculate reconstruction loss of current chain
                                if len(box_sequence) > 1:
                                    # Reshape sequence to correct dimension
                                    current_chain = np.array(box_sequence)
                                    # Ensure shape is (sequence_length, features)
                                    current_chain = current_chain.reshape(len(box_sequence), -1)  
                                    # Convert to tensor and add batch dimension
                                    current_chain = torch.FloatTensor(current_chain).unsqueeze(0).to(device)
                                    
                                    # Use sliding window for reconstruction
                                    window_size = args.history_length
                                    if len(box_sequence) >= window_size:
                                        # Only take the most recent window_size frames
                                        current_chain = current_chain[:, -window_size:, :]
                                        # agent_start_time = time.time()
                                        t_lstm_start = time.time()
                                        reconstructed = args.lstm_ae(current_chain)
                                        t_lstm_total += time.time() - t_lstm_start
                                        # agent_time = time.time() - agent_start_time
                                        # print_and_write_log(f'Agent {agent_idx} time taken to process LSTM-AE: {agent_time:.4f} seconds')
                                        current_loss = F.mse_loss(reconstructed, current_chain)
                                        chain_loss += current_loss.item()
                                        frame_losses.append(current_loss.item())
                            else:
                                matched_success = False
                                break
                                
                        # Only complete chains with successful matching are counted into total loss
                        if matched_success and len(box_sequence) == args.history_length:
                            matched_boxes += 1
                            # Calculate average chain loss (divided by chain length-1, because there is a reconstruction loss between each two frames)
                            avg_chain_loss = chain_loss / (len(box_sequence) - 1)
                            total_chain_loss += avg_chain_loss

                             # Record matching result and loss
                            if args.attack_mode == 'Demo':
                                with open(loss_save_path, 'a') as f:
                                    f.write(f"Box {box_idx}:\n")
                                    if matched_success and len(box_sequence) == args.history_length:
                                        avg_loss = sum(frame_losses) / len(frame_losses)
                                        f.write(f"  Status: Matched (Full Sequence)\n")
                                        f.write(f"  Average Loss: {avg_loss:.6f}\n")
                                        f.write("  Frame-wise Losses:\n")
                                        for i, loss in enumerate(frame_losses):
                                            f.write(f"    Frame {frame_seq-args.history_length+i+1}: {loss:.6f}\n")
                                    else:
                                        f.write(f"  Status: Unmatched (Partial Sequence Length: {len(box_sequence)})\n")
                                        if frame_losses:
                                            f.write("  Partial Frame Losses:\n")
                                            for i, loss in enumerate(frame_losses):
                                                f.write(f"    Frame {frame_seq-len(frame_losses)+i+1}: {loss:.6f}\n")
                                    f.write("\n")
                            
                            print_and_write_log(f'Box {box_idx} completed chain matching, sequence length: {len(box_sequence)}, average chain loss: {avg_chain_loss:.4f}')
                        else:
                            unmatched_boxes += 1
                            print_and_write_log(f'Box {box_idx} chain matching failed, matched length: {len(box_sequence)}')
                    
                    
                    
                    # Calculate average loss of all matched chains
                    if matched_boxes > 0:
                        print_and_write_log(f"Module Runtime - LSTM-AE Inference (Agent {agent_idx}, {matched_boxes} matches): {t_lstm_total:.6f}s")
                        avg_total_loss = total_chain_loss / matched_boxes
                        print_and_write_log(f'Total average chain loss: {avg_total_loss:.4f}, matched boxes: {matched_boxes}, unmatched boxes: {unmatched_boxes}')
                    else:
                        avg_total_loss = float('inf')
                        print_and_write_log(f'No successful matching boxes, unmatched boxes: {unmatched_boxes}')
                    
                    # Add penalty for unmatched boxes
                    unmatched_penalty = unmatched_boxes * 0.05
                    if args.time_only == False:
                        total_loss = 0.1*(avg_total_loss + unmatched_penalty) + 1 - jac_index
                        print_and_write_log(f'Total spatio-temporal detection loss: {total_loss:.4f}')
                    else:
                        total_loss = avg_total_loss + unmatched_penalty
                        print_and_write_log(f'Total temporal detection loss: {total_loss:.4f}')
                    

                    # Check if loss exceeds threshold
                    # Determine rejection based on method (threshold or BH)
                    threshold, threshold_source = get_fixed_decision_threshold(args)
                    
                    if (
                        args.use_bh and BH_AVAILABLE and hasattr(args, 'bh_module')
                        and args.bh_module.has_calibration()
                    ):
                        # Use BH single decision with threshold fallback
                        rejected, bh_info = args.bh_module.decide_single(total_loss, threshold_fallback=threshold)
                        if bh_info['pvalue'] is not None:
                            print_and_write_log(f'Agent {agent_idx} BH decision: score={total_loss:.4f}, p-value={bh_info["pvalue"]:.4f}, rejected={rejected}')
                        else:
                            print_and_write_log(
                                f'Agent {agent_idx} threshold decision (BH fallback): '
                                f'score={total_loss:.4f}, threshold={threshold:.4f}, rejected={rejected}'
                            )
                    else:
                        # Original threshold-based decision
                        rejected = total_loss > threshold
                        print_and_write_log(
                            f'Agent {agent_idx} threshold decision ({threshold_source}): '
                            f'score={total_loss:.4f}, threshold={threshold:.4f}, rejected={rejected}'
                        )
                    
                    if rejected:
                        print_and_write_log(f'Agent {agent_idx} spatio-temporal check failed, discard current frame')
                        
                        # Update continuous interpolation count
                        if agent_idx not in interpolation_counts:
                            interpolation_counts[agent_idx] = 0
                        interpolation_counts[agent_idx] += 1
                        
                        if interpolation_counts[agent_idx] > args.max_interpolation:
                            # Clear cache
                            args.history_frames[agent_idx] = []
                            interpolation_counts[agent_idx] = 0
                            print_and_write_log(f'Agent {agent_idx} continuous interpolation count too many, clear history cache')
                        else:
                            # Use Kalman Filter to interpolate history frames
                            t_kalman_start = time.time()
                            interpolated_frame = kalman_filter_interpolation(history_frames)
                            t_kalman_end = time.time()
                            print_and_write_log(f"Module Runtime - Kalman Filter Interpolation (Agent {agent_idx}): {t_kalman_end - t_kalman_start:.6f}s")
                            args.history_frames[agent_idx][-1] = interpolated_frame
                            print_and_write_log(f'Kalman Filter interpolation passed, current frame cache, continuous interpolation count: {interpolation_counts[agent_idx]}')
                    else:
                        # Reset continuous interpolation count for this agent
                        interpolation_counts[agent_idx] = 0
                        args.history_frames[agent_idx].append(current_bev_flow)
                        print_and_write_log(f'Agent {agent_idx} spatio-temporal check passed, current frame added to cache, average matching loss: {total_loss:.4f}, unmatched boxes: {unmatched_boxes}')
                        collab_agent_list.append(agent_idx)

                # Use trusted agent for final prediction
                if len(collab_agent_list) > 0:
                    data['collab_agent_list'] = collab_agent_list
                    data['no_fuse'] = False
                    data['pert'] = pert.to(device)
                    loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
                else:
                    # If no trusted agent, use ego only
                    data['pert'] = None
                    data['collab_agent_list'] = None
                    data['no_fuse'] = True
                    loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
                
                
                # Use trusted agent for final prediction
                if len(collab_agent_list) > 0:
                    data['collab_agent_list'] = collab_agent_list
                    data['no_fuse'] = False
                    data['pert'] = pert.to(device)
                    loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
                else:
                    # If no trusted agent, use ego only
                    data['pert'] = None
                    data['collab_agent_list'] = None
                    data['no_fuse'] = True
                    loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
                
                # Update detection results
                det_results_local, annotations_local = local_eval(
                    num_agent, padded_voxel_points, reg_target, anchors_map, 
                    gt_max_iou, result, config, det_results_local, annotations_local
                )
                if args.visualization:
                    # visualize consensus result
                    visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='consensus')
            
            elif args.gcp == 'gated_late_fusion':
                num_sensor = num_agent_list[0][0]
                ego_idx = args.ego_agent
                
                # We need results from all agents, so we run predict_all with no_fuse=True and no collaboration (individual detection)
                data['pert'] = pert.to(device) # Apply perturbation if any
                data['collab_agent_list'] = None
                data['no_fuse'] = True
                
                # Get individual detections from all agents
                loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
                
                # IMPORTANT: Save ego agent's original detection result before voting
                # For fair evaluation, we should evaluate ego agent's own detection capability,
                # not the "borrowed" boxes from other agents after voting.
                # The voting result represents the defense output, but evaluation should be on original detection.
                import copy
                ego_original_result = copy.deepcopy(result[ego_idx])
                
                # Apply Gated Late Fusion (this modifies result[ego_idx] in place with voted boxes)
                # The voted boxes may come from any agent, not just ego agent
                box_color_map = [np.random.rand(3) * 255 for _ in range(num_agent)]
                trans_matrices_np = trans_matrices.cpu().numpy()
                
                gated_late_fusion(ego_idx, num_sensor, result, trans_matrices_np, box_color_map, 
                                  iou_thresh=args.gated_fusion_iou, min_votes=args.gated_fusion_votes)
                
                # For fair evaluation: restore ego agent's original detection result
                # The voting result (result[ego_idx]) is used for final output/visualization,
                # but evaluation should use the original detection to ensure fair comparison with upperbound
                result[ego_idx] = ego_original_result
                
                # Update detection results (evaluating original detections, ensuring fair comparison)
                det_results_local, annotations_local = local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)
                
                if args.visualization:
                     # For visualization, restore the voted result to show the defense output
                     gated_late_fusion(ego_idx, num_sensor, result, trans_matrices_np, box_color_map, 
                                      iou_thresh=args.gated_fusion_iou, min_votes=args.gated_fusion_votes)
                     visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='gated_late_fusion')

            elif args.gcp == 'test':
                data['pert'] = None
                data['collab_agent_list'] = [3]
                data['no_fuse'] = False
                loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
                # Update detection results
                det_results_local, annotations_local = local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)
                
                if args.visualization:
                    # visualize consensus result
                    visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='consensus')
    
    print_and_write_log("\n Ego Agent:{}".format(args.ego_agent))

    if args.gcp == 'probing':
        print_and_write_log("Probing: Evaluated on {} frames".format(frame_seq))
        print_and_write_log("Nth frame of each estimation:{}".format(N_th_frame_of_each_estimation))
        print_and_write_log("Final estimation:{}".format(estimated_attacker_ratio))
        print_and_write_log("Ground Truth:{}".format(args.number_of_attackers/(num_agent-1)))
        print_and_write_log("Error of estimation:{}".format(abs(estimated_attacker_ratio - args.number_of_attackers/(num_agent-1))))
        print_and_write_log("Total sampling steps:{}".format(total_sampling_step))
        print_and_write_log("NTry:{}".format(NTry))
        return
        


    if args.gcp == 'adaptive':
        print_and_write_log("Max Consensus set size:{}".format(np.max(consensus_set_sizes)))
        print_and_write_log("Min Consensus set size:{}".format(np.min(consensus_set_sizes)))
        print_and_write_log("Avg Consensus set size:{}".format(np.mean(consensus_set_sizes)))
        # print_and_write_log("Most common Consensus set size:{}".format(np.argmax(np.bincount(consensus_set_sizes))))
        


    if args.gcp == "robosac_validation":
        # validation of robosac theory
        print_and_write_log("robosac VALIDATION: Evaluated on {} frames".format(frame_seq))
        print_and_write_log("Total Neighbor Agents:{}, Sampling Set Size: {}, Number of Attackers: {}".format(num_agent-1, args.robosac_k, args.number_of_attackers))
        print_and_write_log("Expected at least one successful sampling steps at p=0.99: {}".format(cal_robosac_steps(num_agent, args.robosac_k ,args.number_of_attackers)))
        print_and_write_log("Succeeded {}, Total {}, Success Rate: {}".format(succ, frame_seq, succ / frame_seq))
        print_and_write_log("Sampling STEP: MEAN: {}, MAX: {}, MIN:{}".format(np.mean(steps), np.max(steps), np.min(steps)))



    else:
        if args.gcp != 'lowerbound' or args.gcp != 'upperbound':
            print_and_write_log("robosac VALIDATION: Evaluated on {} frames".format(frame_seq))
            print_and_write_log("Total Neighbor Agents:{}, Sampling Set Size: {}, Number of Attackers: {}".format(num_agent-1, args.robosac_k, args.number_of_attackers))
            if args.robosac_k is None:
                consensus_set_size = cal_robosac_consensus(num_agent, args.step_budget, args.number_of_attackers)
                print_and_write_log("Expected guaranteed Consensus Set Size at p=0.99: {}".format(consensus_set_size))
            else:
                print_and_write_log("Expected at least one successful sampling steps at p=0.99: {}".format(cal_robosac_steps(num_agent, args.robosac_k ,args.number_of_attackers)))
            if frame_seq > 0:
                print_and_write_log("Succeeded {}, Total {}, Success Rate: {}".format(succ, frame_seq, succ / frame_seq))
            else:
                print_and_write_log("No frames matched the specified scene_id. Check your scene_id parameter.")
            print_and_write_log("Sampling STEP MEAN: {}, MAX: {}, MIN:{}".format(np.mean(steps), np.max(steps), np.min(steps)))
            total_steps = steps + ego_steps
            print_and_write_log("Total STEP(including ego only step): MEAN: {}, MAX: {}, MIN:{}".format(np.mean(total_steps), np.max(total_steps), np.min(total_steps)))
            fpss = 1000 / (27*steps+17*ego_steps) # forward time: ego only: 17ms; collaborated: 27ms
            print_and_write_log("FPS: MEAN: {}, MAX: {}, MIN:{}".format(np.mean(fpss), np.max(fpss), np.min(fpss)))
            print_and_write_log(
                "Sampling STEP:{}, Ego STEP:{}, Total STEP:{}, FPS:{}".format(steps, ego_steps, total_steps, fpss))
            print_and_write_log("Box set matching threshold: {}".format(args.box_matching_thresh))
            if args.gcp == "fix_attackers":
                print_and_write_log("Fix attackers total step: {}".format(fix_attackers_total_step))
        # mAP evaluation

        # If has RSU, do not count RSU's output into evaluation
        # eval_start_idx = 0 if args.no_cross_road else 1
        eval_start_idx = 0
        # print(len(det_results_local[2][int(idx)][0]), len(annotations_local[2][int(idx)]['bboxes']))
        
        mean_ap_local = []
        # local mAP evaluation
        det_results_all_local = []
        annotations_all_local = []
        for k in range(eval_start_idx, num_agent):
            print_and_write_log("Local mAP@0.5 from agent {}".format(k))
            mean_ap, _ = eval_map(
                det_results_local[k],
                annotations_local[k],
                scale_ranges=None,
                iou_thr=0.5,
                dataset=None,
                logger=None,
            )
            mean_ap_local.append(mean_ap)
            print_and_write_log("Local mAP@0.7 from agent {}".format(k))

            mean_ap, _ = eval_map(
                det_results_local[k],
                annotations_local[k],
                scale_ranges=None,
                iou_thr=0.7,
                dataset=None,
                logger=None,
            )
            mean_ap_local.append(mean_ap)

            det_results_all_local += det_results_local[k]
            annotations_all_local += annotations_local[k]

        # average local mAP evaluation
        print_and_write_log("Average Local mAP@0.5")

        mean_ap_local_average, _ = eval_map(
            det_results_all_local,
            annotations_all_local,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=None,
            logger=None,
        )
        mean_ap_local.append(mean_ap_local_average)

        print_and_write_log("Average Local mAP@0.7")

        mean_ap_local_average, _ = eval_map(
            det_results_all_local,
            annotations_all_local,
            scale_ranges=None,
            iou_thr=0.7,
            dataset=None,
            logger=None,
        )
        mean_ap_local.append(mean_ap_local_average)

        print_and_write_log(
            "Quantitative evaluation results of model from {}, at epoch {}".format(
                args.resume, start_epoch - 1
            )
        )

        for k in range(eval_start_idx, num_agent):
            print_and_write_log(
                "agent{} mAP@0.5 is {} and mAP@0.7 is {}".format(
                    k, mean_ap_local[k * 2], mean_ap_local[(k * 2) + 1]
                )
            )

        print_and_write_log(
            "average local mAP@0.5 is {} and average local mAP@0.7 is {}".format(
                mean_ap_local[-2], mean_ap_local[-1]
            )
        )

        if need_log:
            saver.close()

def kalman_filter_interpolation(history_frames):
    """
    Use Kalman Filter to interpolate detection boxes in history frames
    Args:
        history_frames: list of arrays, containing detection box sequences from history frames
    Returns:
        interpolated_frame: interpolated detection boxes for next frame
    """
    from filterpy.kalman import KalmanFilter
    import numpy as np
    
    # If history frames are empty, return empty array
    if not history_frames or len(history_frames) < 2:
        return np.array([])
        
    # Get detection boxes in last frame
    last_frame = history_frames[-1]
    if len(last_frame) == 0:
        return np.array([])
    
    # Initialize interpolated result list
    interpolated_boxes = []
    
    # Process each detection box in last frame
    for box_idx in range(len(last_frame)):
        # Try to find trajectory of this box in history frames
        box_trajectory = []
        current_box = last_frame[box_idx]
        
        # Traverse from last frame to first
        prev_box = current_box
        for frame in reversed(history_frames[:-1]):  # Not including last frame
            # Find the best matching box in previous frame
            best_match = None
            best_iou = 0
            
            for box in frame:
                iou = calculate_iou(prev_box, box)  # Need to implement calculate_iou function
                if iou > best_iou:
                    best_iou = iou
                    best_match = box
            
            # If found matching box and IoU is greater than threshold
            if best_match is not None and best_iou > 0.3:
                box_trajectory.insert(0, best_match)
                prev_box = best_match
            else:
                break
                
        # Add current box to trajectory
        box_trajectory.append(current_box)
        
        # If trajectory is too short, skip this box
        if len(box_trajectory) < 2:
            continue
            
        # Initialize Kalman Filter
        kf = KalmanFilter(dim_x=16, dim_z=8)  # State: [x1,y1,x2,y2,x3,y3,x4,y4, vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4]
        
        # State transition matrix F
        kf.F = np.eye(16)
        dt = 1.0  # Time step
        for i in range(8):
            kf.F[i, i+8] = dt
            
        # Measurement matrix H
        kf.H = np.concatenate([np.eye(8), np.zeros((8, 8))], axis=1)
        
        # Process noise covariance Q
        q = 0.1  # Process noise parameter
        kf.Q = np.eye(16) * q
        for i in range(8, 16):
            kf.Q[i, i] *= 10  # Noise for velocity components is larger
            
        # Measurement noise covariance R
        r = 0.1  # Measurement noise parameter
        kf.R = np.eye(8) * r
        
        # Initial state
        kf.x = np.zeros(16)
        kf.x[:8] = box_trajectory[0]  # Position
        if len(box_trajectory) > 1:
            # Initial velocity estimation
            kf.x[8:] = (box_trajectory[1] - box_trajectory[0]) / dt
            
        # Initial state covariance
        kf.P = np.eye(16) * 100
        
        # Update filter with history trajectory
        for box in box_trajectory[1:]:
            kf.predict()
            kf.update(box)
            
        # Predict position of next frame
        kf.predict()
        predicted_box = kf.x[:8]
        
        # Add predicted result
        interpolated_boxes.append(predicted_box)
    
    return np.array(interpolated_boxes)

def calculate_iou(box1, box2):
    """
    Calculate IoU of two detection boxes
    Args:
        box1: coordinates of first detection box [x1,y1,x2,y2,x3,y3,x4,y4]
        box2: coordinates of second detection box [x1,y1,x2,y2,x3,y3,x4,y4]
    Returns:
        iou: IoU value of two boxes
    """
    # Convert four points to rectangle coordinates
    def get_rect(box):
        x = [box[i] for i in range(0, 8, 2)]
        y = [box[i] for i in range(1, 8, 2)]
        return [min(x), min(y), max(x), max(y)]
    
    # Convert four points to rectangle coordinates
    rect1 = get_rect(box1)
    rect2 = get_rect(box2)
    
    # Calculate intersection area
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate areas of each box
    area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    
    # 计算IoU
    iou = intersection / (area1 + area2 - intersection)
    return iou



def calculate_iou_from_corners(corners1, corners2):
    """
    Calculate IoU using corner coordinates
    Args:
        corners1: (4, 2) coordinates of four corners of first box
        corners2: (4, 2) coordinates of four corners of second box
    Returns:
        iou: IoU value
    """
    from shapely.geometry import Polygon

    poly1 = Polygon(corners1)
    poly2 = Polygon(corners2)

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    # Calculate intersection and union
    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - intersection

    if union == 0:
        return 0.0

    return intersection / union

def find_best_match(current_box, frame):
    """
    Find the best matching box in the given frame
    Args:
        current_box: current box (1, 8) - 8 coordinates
        frame: all boxes in the frame to be matched (N, 8)
    Returns:
        best_match: best matching box
        best_iou: IoU value of best matching box
    """
    best_iou = 0
    best_match = None
    
    current_corners = current_box[0, :8].reshape(4, 2)
    
    for box in frame:
        box_corners = box[:8].reshape(4, 2)
        iou = calculate_iou_from_corners(current_corners, box_corners)
        
        if iou > best_iou:
            best_iou = iou
            best_match = box.reshape(1, -1)  # Keep 2D shape
            
    return best_match, best_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-d",
        "--data",
        default="/data2/user2/yihang/GCP/dataset/V2X-Sim/test",
        type=str,
        help="The path to the preprocessed sparse BEV training data",
    )
    parser.add_argument("--batch", default=1, type=int, help="The number of scene")
    parser.add_argument("--nepoch", default=100, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=4, type=int, help="Number of workers")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--log", action="store_true", help="Whether to log")
    parser.add_argument("--logpath", default="/data2/user2/yihang/GCP/coperception/logs", help="The path to the output log file")
    parser.add_argument(
        "--resume",
        # default = "../../ckpt/meanfusion/epoch_advtrain_49.pth" #use this adv epoch 49 trained from scratch
          default="/data2/user2/yihang/GCP/coperception/ckpt/det/meanfusion/epoch_49.pth",
        # default= "/data2/user2/yihang/GCP/coperception/ckpt/det/v2v/with_rsu/epoch_100.pth",
        type=str,
        help="The path to the saved model that is loaded to resume training",
    )
    parser.add_argument(
        "--resume_teacher",
        default="",
        type=str,
        help="The path to the saved teacher model that is loaded to resume training",
    )
    parser.add_argument(
        "--layer",
        default=3,
        type=int,
        help="Communicate which layer in the single layer com mode",
    )
    parser.add_argument(
        "--warp_flag", action="store_true", help="Whether to use pose info for When2com"
    )
    parser.add_argument(
        "--kd_flag",
        default=0,
        type=int,
        help="Whether to enable distillation (only DiscNet is 1 )",
    )
    parser.add_argument("--kd_weight", default=100000, type=int, help="KD loss weight")
    parser.add_argument(
        "--gnn_iter_times",
        default=3,
        type=int,
        help="Number of message passing for V2VNet",
    )
    parser.add_argument(
        "--visualization", action="store_true", help="Visualize validation result"
    )
    parser.add_argument(
        "--com", default="mean", type=str, help="disco/when2com/v2v/sum/mean/max/cat/agent"
    )
    parser.add_argument(
        "--bound",
        type=str,
        default="both",
        help="The input setting: lowerbound -> single-view or upperbound -> multi-view",
    )
    parser.add_argument("--inference", type=str)
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--box_com", action="store_true")
    parser.add_argument(
        "--no_cross_road", action="store_true", help="Do not load data of cross roads"
    )
    # scene_batch => batch size in each scene
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )
    parser.add_argument(
        "--apply_late_fusion",
        default=0,
        type=int,
        help="1: apply late fusion. 0: no late fusion",
    )
    parser.add_argument(
        "--compress_level",
        default=0,
        type=int,
        help="Compress the communication layer channels by 2**x times in encoder",
    )
    parser.add_argument(
        "--pose_noise",
        default=0,
        type=float,
        help="draw noise from normal distribution with given mean (in meters), apply to transformation matrix.",
    )
    parser.add_argument(
        "--only_v2i",
        default=0,
        type=int,
        help="1: only v2i, 0: v2v and v2i",
    )

    # Adversarial perturbation
    parser.add_argument('--pert_alpha', type=float, default=0.1, help='scale of the perturbation')
    parser.add_argument('--adv_method', type=str, default='pgd', help='pgd/bim/cw-l2/simple_bac/bac')
    parser.add_argument('--eps', type=float, default=0.5, help='epsilon of adv attack.')
    parser.add_argument('--adv_iter', type=int, default=15, help='adv iterations of computing perturbation')
    parser.add_argument('--attack_mode', type=str, default='Random', help='Random/Poission/SI')
    parser.add_argument('--attack_ratio', type=float, default=0.6, help='proportion of adv frames')
    parser.add_argument('--time_only', action="store_true", default=False, help='only use temporal loss to detect adv perturbation')
    parser.add_argument('--max_interpolation', type=int, default=3, help='maximum number of interpolation')
    # Scene and frame settings
    parser.add_argument('--scene_id', type=int, nargs='+', default=[8], help='target evaluation scene') #Scene 8, 96, 97 has 6 agents.
    parser.add_argument('--sample_id', type=int, default=None, help='(Legacy) target evaluation sample, same as --sample_start')
    parser.add_argument('--sample_start', type=int, default=None, help='start frame ID (inclusive)')
    parser.add_argument('--sample_end', type=int, default=None, help='end frame ID (inclusive)')
    parser.add_argument('--history_length', type=int, default=5, help='length of history frames')
    parser.add_argument('--reconstruction_threshold', type=float, default=0.8, help='reconstruction error threshold for temporal check')
    # GCP modes and parameters
    parser.add_argument('--gcp', type=str, default='', help='upperbound/lowerbound/no_defense/robosac_validation/robosac_mAP/adaptive/fix_attackers/performance_eval/probing/made/gcp/test')
    parser.add_argument('--bac_eps', type=float, default=0.3, help='Threshold for BAC attack')
    parser.add_argument('--simple_bac_eps', type=float, default=0.3, help='Output threshold for Simple BAC (output-space weighting)')
    parser.add_argument('--ego_agent', type=int, default=1, help='id of ego agent')
    parser.add_argument('--robosac_k', type=int, default=None, help='specify consensus set size if needed (for robosac_mAP mode)')
    parser.add_argument('--ego_loss_only', action="store_true", help='only use ego loss to compute adv perturbation')
    parser.add_argument('--step_budget', type=int, default=3, help='sampling budget in a single frame')
    parser.add_argument('--box_matching_thresh', type=float, default=0.3, help='IoU threshold for validating two detection results')
    parser.add_argument('--number_of_attackers', type=int, default=1, help='number of malicious attackers in the scene')
    parser.add_argument('--fix_attackers', action="store_true", help='if true, attackers will not change in different frames')
    parser.add_argument('--use_history_frame', action="store_true", help='use history frame for computing the consensus, reduce 1 step of forward prop.')
    parser.add_argument('--partial_upperbound', action="store_true", help='use with specifying ransan_k, to perform clean collaboration with a subset of teammates')
    # BH Decision Module parameters
    parser.add_argument('--use_bh', action="store_true", help='use Benjamini-Hochberg procedure for decision making instead of fixed threshold')
    parser.add_argument('--alpha_bh', type=float, default=0.2, help='FDR control level for BH procedure (default: 0.2)')
    parser.add_argument('--calibration_path', type=str, default=None, help='path to calibration set file for BH decision')
    parser.add_argument('--fixed_decision_threshold', type=float, default=None,
                        help='manual fixed threshold used when BH is disabled or calibration is unavailable')
    
    # Gated Late Fusion parameters
    parser.add_argument('--gated_fusion_iou', type=float, default=0.3, help='IoU threshold for gated late fusion')
    parser.add_argument('--gated_fusion_votes', type=int, default=2, help='Minimum votes for gated late fusion')

    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    print(args)
    main(args)
