"""
BAC (Blind Area Confusion) Attack Implementation
Based on paper: GCP - Guarded Collaborative Perception

Implements:
1. Differential Detection: Identify victim's unique detections
2. Blind Region Segmentation (BRS): Algorithm 1 from paper
3. BAC Perturbation Optimization: Equation 4 from paper

Note: BAC mask generation uses slow update strategy (0.5 FPS) as mentioned in paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import deque
from scipy.optimize import linear_sum_assignment

# Global cache for BAC masks (slow update strategy)
_bac_mask_cache = {}
_bac_mask_frame_counter = {}


def extract_detection_boxes_from_result(result, config, padded_voxel_points, 
                                         reg_target, anchors_map, gt_max_iou, agent_idx):
    """
    Extract detection boxes from model output
    Use the same method as local_eval
    
    Args:
        result: Model output [agent_idx][0][0] is the detection result
        config: Configuration object
        padded_voxel_points: BEV voxel data
        reg_target: Regression target
        anchors_map: Anchor mapping
        gt_max_iou: Ground truth IoU
        agent_idx: Agent index
        
    Returns:
        boxes: Detection box list shape (N, 9) [x1,y1,x2,y2,x3,y3,x4,y4,conf]
    """
    try:
        from coperception.utils.detection_util import cal_local_mAP
        
        # Check if result is valid
        if result is None or agent_idx >= len(result):
            return np.array([])
        
        result_temp = result[agent_idx]
        if result_temp is None or len(result_temp) == 0:
            return np.array([])
        
        # Construct data dictionary (same format as local_eval)
        data_agents = {
            "bev_seq": torch.unsqueeze(padded_voxel_points[agent_idx, :, :, :, :], 1),
            "reg_targets": torch.unsqueeze(reg_target[agent_idx, :, :, :, :, :], 0),
            "anchors": torch.unsqueeze(anchors_map[agent_idx, :, :, :, :], 0),
        }
        
        # Process gt_max_iou
        temp_gt = gt_max_iou[agent_idx]
        if len(temp_gt[0]["gt_box"]) == 0:
            data_agents["gt_max_iou"] = []
        else:
            data_agents["gt_max_iou"] = temp_gt[0]["gt_box"][0, :, :]
        
        # Construct temp dictionary for cal_local_mAP
        temp = {
            "bev_seq": data_agents["bev_seq"][0, -1].cpu().numpy(),
            "result": [] if len(result_temp) == 0 else result_temp[0][0],
            "reg_targets": data_agents["reg_targets"].cpu().numpy()[0],
            "anchors_map": data_agents["anchors"].cpu().numpy()[0],
            "gt_max_iou": data_agents["gt_max_iou"],
        }
        
        # Call cal_local_mAP to extract detection boxes
        det_results, _ = cal_local_mAP(config, temp, [], [])
        
        # Return detection boxes
        if len(det_results) > 0 and len(det_results[0]) > 0:
            boxes = det_results[0][0]  # Shape: (N, 9)
            return boxes
        
        return np.array([])
        
    except Exception as e:
        print(f"[extract_detection_boxes] Error: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])


def normalize_box_array(boxes):
    """Return detection boxes as a 2D numpy array with shape [N, >=8]."""
    if boxes is None:
        return np.empty((0, 9), dtype=np.float32)

    boxes = np.asarray(boxes)
    if boxes.size == 0:
        return np.empty((0, 9), dtype=np.float32)

    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)

    return boxes.astype(np.float32, copy=False)


def calculate_iou_from_boxes(box1, box2):
    """Calculate IoU between two detection boxes represented by 4 BEV corners."""
    from shapely.geometry import Polygon

    box1 = np.asarray(box1).reshape(-1)
    box2 = np.asarray(box2).reshape(-1)
    if box1.shape[0] < 8 or box2.shape[0] < 8:
        return 0.0

    poly1 = Polygon(box1[:8].reshape(4, 2))
    poly2 = Polygon(box2[:8].reshape(4, 2))
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    union = poly1.area + poly2.area - poly1.intersection(poly2).area
    if union <= 0:
        return 0.0

    return poly1.intersection(poly2).area / union


def differential_detection(single_boxes, cp_boxes, iou_threshold=0.3):
    """
    Partition detections into matched, victim-only, and collaborative-only groups.

    `single_boxes` correspond to the victim's standalone perception.
    `cp_boxes` correspond to collaborative perception with the malicious sender.
    """
    single_boxes = normalize_box_array(single_boxes)
    cp_boxes = normalize_box_array(cp_boxes)

    if len(single_boxes) == 0 and len(cp_boxes) == 0:
        empty = np.empty((0, 9), dtype=np.float32)
        return empty, empty, empty, empty
    if len(single_boxes) == 0:
        empty = np.empty((0, cp_boxes.shape[1]), dtype=np.float32)
        return empty, cp_boxes, empty, empty
    if len(cp_boxes) == 0:
        empty = np.empty((0, single_boxes.shape[1]), dtype=np.float32)
        return single_boxes, empty, empty, empty

    iou_matrix = np.zeros((len(single_boxes), len(cp_boxes)), dtype=np.float32)
    for i, s_box in enumerate(single_boxes):
        for j, c_box in enumerate(cp_boxes):
            iou_matrix[i, j] = calculate_iou_from_boxes(s_box, c_box)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_pairs = [
        (row_idx, col_idx)
        for row_idx, col_idx in zip(row_ind, col_ind)
        if iou_matrix[row_idx, col_idx] >= iou_threshold
    ]

    matched_single_idx = {row_idx for row_idx, _ in matched_pairs}
    matched_cp_idx = {col_idx for _, col_idx in matched_pairs}

    victim_only = single_boxes[
        [idx for idx in range(len(single_boxes)) if idx not in matched_single_idx]
    ]
    collaborative_only = cp_boxes[
        [idx for idx in range(len(cp_boxes)) if idx not in matched_cp_idx]
    ]
    matched_single = single_boxes[sorted(matched_single_idx)] if matched_single_idx else np.empty((0, single_boxes.shape[1]), dtype=np.float32)
    matched_cp = cp_boxes[sorted(matched_cp_idx)] if matched_cp_idx else np.empty((0, cp_boxes.shape[1]), dtype=np.float32)

    return victim_only, collaborative_only, matched_single, matched_cp


def box_to_grid(box, grid_size, map_range=(-32, 32)):
    """
    Convert detection box coordinates to grid coordinates
    
    Args:
        box: Detection box [x1,y1,x2,y2,x3,y3,x4,y4,conf] (9 dimensions)
        grid_size: Grid size (H, W)
        map_range: BEV map range (min, max), default (-32, 32) meters
        
    Returns:
        grid_coords: Grid coordinates list [(i1,j1), (i2,j2), ...]
    """
    try:
        H, W = grid_size
        min_coord, max_coord = map_range
        map_size = max_coord - min_coord  # 64米
        
        # Ensure box is numpy array
        if not isinstance(box, np.ndarray):
            box = np.array(box)
        
        # Extract 4 corners
        box_flat = box.flatten()  # Ensure 1D
        if box_flat.shape[0] < 8:
            return [(H//2, W//2)]
        
        corners = box_flat[:8].reshape(4, 2)  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
        # Calculate bounding box
        x_min, y_min = float(corners[:, 0].min()), float(corners[:, 1].min())
        x_max, y_max = float(corners[:, 0].max()), float(corners[:, 1].max())
        
        # Convert world coordinates to grid coordinates [-32, 32] -> [0, H-1]
        grid_x_min = int(np.clip((x_min - min_coord) / map_size * W, 0, W-1))
        grid_x_max = int(np.clip((x_max - min_coord) / map_size * W, 0, W-1))
        grid_y_min = int(np.clip((y_min - min_coord) / map_size * H, 0, H-1))
        grid_y_max = int(np.clip((y_max - min_coord) / map_size * H, 0, H-1))
        
        # Generate grid coordinates list
        coords = []
        for i in range(grid_y_min, grid_y_max + 1):
            for j in range(grid_x_min, grid_x_max + 1):
                if 0 <= i < H and 0 <= j < W:
                    coords.append((i, j))
        
        return coords if len(coords) > 0 else [(H//2, W//2)]
    
    except Exception as e:
        print(f"[box_to_grid] Error: {e}, returning center")
        H, W = grid_size
        return [(H//2, W//2)]


def cluster_victim_grid(Y_vic, grid_size=(32, 32), map_range=(-32, 32)):
    """
    Find the grid point farthest from all victim detection boxes as victim grid e
    
    Args:
        Y_vic: Victim detection boxes (N, 9)
        grid_size: BEV grid size (H, W)
        map_range: BEV map range
        
    Returns:
        e: Grid coordinates of victim center (i, j)
    """
    if not isinstance(Y_vic, np.ndarray) or Y_vic.ndim != 2 or Y_vic.shape[0] == 0:
        return (grid_size[0] // 2, grid_size[1] // 2)
    
    H, W = grid_size
    min_coord, max_coord = map_range
    map_size = max_coord - min_coord
    
    # Convert all detection box centers to grid coordinates
    box_centers_grid = []
    for box in Y_vic:
        corners = box[:8].reshape(4, 2)
        cx, cy = corners.mean(axis=0)
        # Convert world coordinates to grid coordinates
        gi = int(np.clip((cy - min_coord) / map_size * H, 0, H-1))
        gj = int(np.clip((cx - min_coord) / map_size * W, 0, W-1))
        box_centers_grid.append((gi, gj))
    
    # Find the grid point farthest from all detection box centers
    max_dist = -1
    victim_grid = (H // 2, W // 2)
    
    for i in range(H):
        for j in range(W):
            total_dist = 0
            for gi, gj in box_centers_grid:
                dist = np.sqrt((i - gi)**2 + (j - gj)**2)
                total_dist += dist
            
            if total_dist > max_dist:
                max_dist = total_dist
                victim_grid = (i, j)
    
    return victim_grid


def get_adaptive_neighbors(s, e, K_base=6, gamma_d=0.3, D_norm=45.25):
    """
    Adaptive neighborhood selection (Equation 3 in paper)
    K_s = K_base * exp(-gamma_d * d(s,e) / D_norm)
    
    Args:
        s: Current grid point (i, j)
        e: Victim grid center (i, j)
        K_base: Base number of neighbors (default 6)
        gamma_d: Decay rate (default 0.3)
        D_norm: Normalization factor sqrt(H^2 + W^2)
        
    Returns:
        K_s: Number of neighbors for grid s
    """
    dist = np.sqrt((s[0] - e[0])**2 + (s[1] - e[1])**2)
    K_s = int(K_base * np.exp(-gamma_d * dist / D_norm))
    return max(K_s, 1)  # At least 1 neighbor


def get_neighbors(point, H, W, K=6):
    """
    Get neighbors of a grid point (8-connected)
    
    Args:
        point: Current grid point (i, j)
        H, W: Grid dimensions
        K: Number of neighbors to return
        
    Returns:
        neighbors: List of neighbor coordinates
    """
    i, j = point
    
    # 8-connected neighbors
    all_neighbors = [
        (i-1, j-1), (i-1, j), (i-1, j+1),
        (i, j-1),             (i, j+1),
        (i+1, j-1), (i+1, j), (i+1, j+1)
    ]
    
    # Filter valid neighbors
    valid_neighbors = []
    for ni, nj in all_neighbors:
        if 0 <= ni < H and 0 <= nj < W:
            valid_neighbors.append((ni, nj))
    
    # Return K nearest neighbors (or all if less than K)
    return valid_neighbors[:min(K, len(valid_neighbors))]


def blind_region_segmentation(detection_map, Y_vic, Y_nvic, grid_size=(32, 32), 
                               K_base=6, gamma_d=0.3, map_range=(-32, 32),
                               confidence_threshold=0.3):
    """
    Algorithm 1: Blind Region Segmentation (BRS)
    
    Args:
        detection_map: BEV detection confidence map [H, W]
        Y_vic: Victim-only detections (N, 9)
        Y_nvic: Non-victim detections (M, 9)
        grid_size: Grid size (H, W)
        K_base: Base neighbor count (higher = faster CA expansion)
        gamma_d: Distance decay rate (lower = less decay, more neighbors far from victim)
        map_range: BEV map range
        confidence_threshold: Threshold for CA vs BA (lower = more CA)
        
    Returns:
        M_i: Binary confidence mask [H, W] (1=confident, 0=blind)
    """
    print(f"[BRS] Starting blind region segmentation, grid={grid_size}")
    H, W = grid_size
    D_norm = np.sqrt(H**2 + W**2)
    
    # Initialize confidence mask (all zeros)
    M_i = np.zeros((H, W), dtype=np.float32)
    print(f"[BRS] Initialized mask: {H}x{W}")
    
    # Step 1: Find victim grid e (Algorithm 1, Line 2)
    print(f"[BRS] Step 1: Finding victim grid center...")
    e = cluster_victim_grid(Y_vic, grid_size, map_range)
    print(f"[BRS] Victim grid center: {e}")
    
    # Step 2: Initialize seed grids (Line 3-7)
    print(f"[BRS] Step 2: Initializing seed grids...")
    Q_ca = deque()  # Confident Area queue
    Q_ba = deque()  # Blind Area queue
    
    # Mark grids containing victim detections as confident (CA seeds)
    if isinstance(Y_vic, np.ndarray) and Y_vic.ndim == 2 and Y_vic.shape[0] > 0:
        print(f"[BRS] Processing {Y_vic.shape[0]} victim boxes for CA seeds...")
        for box in Y_vic:
            grid_coords = box_to_grid(box, grid_size, map_range)
            for coord in grid_coords:
                ci, cj = int(coord[0]), int(coord[1])
                if M_i[ci, cj] == 0:  # Not yet marked
                    M_i[ci, cj] = 1  # Confident
                    Q_ca.append((ci, cj))
        print(f"[BRS] Added {len(Q_ca)} CA seed grids")
    
    # If no victim detections, use center region as CA seed
    if len(Q_ca) == 0:
        print(f"[BRS] No victim detections, using center as CA seed")
        center_i, center_j = H // 2, W // 2
        for di in range(-1, 2):
            for dj in range(-1, 2):
                i, j = center_i + di, center_j + dj
                if 0 <= i < H and 0 <= j < W:
                    M_i[i, j] = 1
                    Q_ca.append((i, j))
    
    # Mark grids containing non-victim detections as blind (BA seeds)
    # Note: Don't mark all nvic boxes as blind, only use differential ones
    # In practice, nvic boxes often overlap with vic boxes, so we're more conservative
    if isinstance(Y_nvic, np.ndarray) and Y_nvic.ndim == 2 and Y_nvic.shape[0] > 0:
        print(f"[BRS] Processing {Y_nvic.shape[0]} non-victim boxes for BA seeds...")
        # Only mark edges as BA seeds to avoid over-marking
        for box in Y_nvic[:min(3, len(Y_nvic))]:  # Limit to first 3 boxes
            grid_coords = box_to_grid(box, grid_size, map_range)
            # Only mark a few representative grids per box
            for coord in grid_coords[:min(2, len(grid_coords))]:
                ci, cj = int(coord[0]), int(coord[1])
                if M_i[ci, cj] == 0:  # Not yet marked
                    M_i[ci, cj] = -1  # Blind
                    Q_ba.append((ci, cj))
        print(f"[BRS] Added {len(Q_ba)} BA seed grids")
    
    # Step 3: Region growing (Line 8-15)
    print(f"[BRS] Step 3: Region growing (CA queue: {len(Q_ca)}, BA queue: {len(Q_ba)})...")
    processed = set()
    max_iterations = H * W * 2  # Prevent infinite loop
    iteration_count = 0
    
    while (len(Q_ca) > 0 or len(Q_ba) > 0) and iteration_count < max_iterations:
        iteration_count += 1
        
        if iteration_count % 50 == 0:
            labeled = np.sum(M_i != 0)
            print(f"[BRS] Iter {iteration_count}: Q_ca={len(Q_ca)}, Q_ba={len(Q_ba)}, processed={len(processed)}, labeled={labeled}/{H*W}")
        
        # Process confident area queue
        if len(Q_ca) > 0:
            s = Q_ca.popleft()
            if s in processed:
                continue
            processed.add(s)
            
            # Get adaptive neighbors
            K_s = get_adaptive_neighbors(s, e, K_base, gamma_d, D_norm)
            neighbors = get_neighbors(s, H, W, K_s)
            
            for neighbor in neighbors:
                ni, nj = int(neighbor[0]), int(neighbor[1])
                neighbor_tuple = (ni, nj)
                
                # Skip if already processed or already labeled
                if neighbor_tuple in processed or M_i[ni, nj] != 0:
                    continue
                
                # Check if should be CA (based on confidence)
                try:
                    confidence_val = detection_map[ni, nj]
                    if hasattr(confidence_val, 'item'):
                        confidence_val = confidence_val.item()
                    else:
                        confidence_val = float(np.array(confidence_val).flatten()[0])
                except:
                    confidence_val = 0.5
                
                if confidence_val > confidence_threshold:
                    M_i[ni, nj] = 1
                    Q_ca.append(neighbor_tuple)
                else:
                    M_i[ni, nj] = -1
                    Q_ba.append(neighbor_tuple)
        
        # Process blind area queue
        if len(Q_ba) > 0:
            s = Q_ba.popleft()
            if s in processed:
                continue
            processed.add(s)
            
            # Get adaptive neighbors
            K_s = get_adaptive_neighbors(s, e, K_base, gamma_d, D_norm)
            neighbors = get_neighbors(s, H, W, K_s)
            
            for neighbor in neighbors:
                ni, nj = int(neighbor[0]), int(neighbor[1])
                neighbor_tuple = (ni, nj)
                
                # Skip if already processed or already labeled
                if neighbor_tuple in processed or M_i[ni, nj] != 0:
                    continue
                
                M_i[ni, nj] = -1
                Q_ba.append(neighbor_tuple)
    
    final_labeled = np.sum(M_i != 0)
    print(f"[BRS] Completed after {iteration_count} iterations, labeled {final_labeled}/{H*W} grids")
    
    # Convert to binary mask (Line 16)
    # 1 for confident areas, 0 for blind areas
    print(f"[BRS] Converting to binary mask...")
    M_i_binary = (M_i == 1).astype(np.float32)
    confident_count = np.sum(M_i_binary == 1)
    blind_count = np.sum(M_i_binary == 0)
    print(f"[BRS] Final mask: {confident_count} confident grids, {blind_count} blind grids")
    
    return M_i_binary


def generate_bac_attack(fafmodule, data, num_agent, ego_agent=1, 
                        attacker_list=[0, 2], num_iterations=15, 
                        alpha=0.1, eps=0.5, device='cuda', 
                        config=None, reg_target=None, anchors_map=None, 
                        gt_max_iou=None, padded_voxel_points=None,
                        scene_id=0, frame_id=0, mask_update_rate=10):
    """
    Generate complete BAC attack
    
    Steps:
    1. Differential Detection (only when updating mask)
    2. Blind Region Segmentation (slow update: 0.5 FPS, ~10 frames)
    3. BAC Perturbation Optimization (every frame)
    
    Args:
        fafmodule: FaFModule with model
        data: Input data
        num_agent: Number of agents
        ego_agent: Ego agent index
        attacker_list: List of attacker indices
        num_iterations: PGD iterations
        alpha: Step size
        eps: Perturbation bound
        device: Device
        config, reg_target, anchors_map, gt_max_iou, padded_voxel_points: For box extraction
        scene_id: Scene ID for caching
        frame_id: Frame ID for caching
        mask_update_rate: Mask update interval (frames), default 10 for 0.5 FPS
        
    Returns:
        pert: BAC perturbation
        blind_mask: Generated blind area mask
    """
    
    global _bac_mask_cache, _bac_mask_frame_counter
    
    # Validate attacker_list
    if attacker_list is None or not isinstance(attacker_list, (list, tuple)):
        attacker_list = [0, 2]  # Default
        print(f"[BAC] Warning: Invalid attacker_list, using default {attacker_list}")
    
    cache_key = f"scene_{scene_id}_ego_{ego_agent}"
    should_update_mask = True
    
    print(f"[BAC] Cache check: scene={scene_id}, frame={frame_id}, key={cache_key}")
    print(f"[BAC] Cache status: key_exists={cache_key in _bac_mask_cache}")
    
    # Check if we can reuse cached mask (slow update strategy)
    if cache_key in _bac_mask_cache:
        last_frame = _bac_mask_frame_counter.get(cache_key, 0)
        frames_since_update = frame_id - last_frame
        print(f"[BAC] Last update at frame {last_frame}, current frame {frame_id}, delta={frames_since_update}")
        
        if frames_since_update < mask_update_rate:
            print(f"[BAC] Reusing cached mask (frame {frames_since_update}/{mask_update_rate})")
            blind_mask_full = _bac_mask_cache[cache_key]
            should_update_mask = False
        else:
            print(f"[BAC] Cache expired, regenerating mask")
    
    # Generate new mask if needed
    if should_update_mask:
        print(f"[BAC] Generating new mask (update cycle)")
        blind_mask_full = _generate_bac_mask(
            fafmodule, data, num_agent, ego_agent, attacker_list, device,
            config, reg_target, anchors_map, gt_max_iou, padded_voxel_points
        )
        
        # Update cache
        _bac_mask_cache[cache_key] = blind_mask_full
        _bac_mask_frame_counter[cache_key] = frame_id
        print(f"[BAC] Mask cached for scene {scene_id}, frame {frame_id}")
    
    # Step 3: BAC Perturbation Optimization (every frame)
    print(f"[BAC] Step 3: Perturbation with {num_iterations} iterations...")
    
    # Initialize perturbation
    pert = torch.randn(num_agent, 256, 32, 32).to(device) * 0.01
    pert.requires_grad = True
    
    # Convert blind mask to tensor
    blind_mask_tensor = torch.from_numpy(blind_mask_full).float().to(device)
    M_delta = 1.0 - blind_mask_tensor  # Invert: 1=blind, 0=confident
    
    # Get pseudo ground truth
    try:
        data_clean = {k: v for k, v in data.items()}
        data_clean['pert'] = None
        data_clean['no_fuse'] = True
        cls_result = fafmodule.cls_predict(data_clean, 1, no_fuse=True)
        mean = torch.mean(cls_result, dim=2)
        cls_result[:,:,0] = cls_result[:,:,0] > mean
        cls_result[:,:,1] = cls_result[:,:,1] > mean
        pseudo_gt = cls_result.clone().detach()
    except:
        pseudo_gt = None
    
    # PGD optimization
    for iteration in range(num_iterations):
        if pert.grad is not None:
            pert.grad.zero_()
        
        # Apply perturbation and set required data fields
        data['pert'] = pert
        data['attacker_list'] = attacker_list
        data['eps'] = eps
        data['ego_agent'] = ego_agent
        data['unadv_pert'] = None
        
        # Forward pass to get result
        try:
            result = fafmodule.model(
                data['bev_seq'], data['trans_matrices'], data['num_agent'],
                batch_size=1, pert=pert, no_fuse=False,
                unadv_pert=None, attacker_list=attacker_list,
                eps=eps, ego_agent=ego_agent
            )
            
            # Compute classification loss manually
            labels = pseudo_gt if pseudo_gt is not None else data['labels']
            labels = labels.view(result['cls'].shape[0], -1, result['cls'].shape[-1])
            
            # Invert labels for adversarial attack
            inverted_labels = labels.clone()
            inverted_labels[:, :, 0] = 1 - inverted_labels[:, :, 0]
            inverted_labels[:, :, 1] = 1 - inverted_labels[:, :, 1]
            
            # Compute BCE loss
            cls_loss = F.binary_cross_entropy_with_logits(
                result['cls'], inverted_labels, reduction='mean'
            )
            
            # Backward (without optimizer step)
            cls_loss.backward()
            
            # PGD update with blind mask guidance
            with torch.no_grad():
                if pert.grad is not None:
                    grad = pert.grad
                    
                    # Apply blind mask weighting to attackers
                    for idx in attacker_list:
                        # Expand mask [32,32] -> [256,32,32]
                        mask_expanded = M_delta[idx].unsqueeze(0).expand(256, -1, -1)
                        # Weight gradient (emphasize blind areas)
                        grad[idx] = grad[idx] * (1.0 + mask_expanded * 1.0)
                    
                    # PGD step
                    pert = pert - alpha * grad.sign()
                    pert = torch.clamp(pert, -eps, eps)
                    
                    # Zero non-attackers
                    for idx in range(num_agent):
                        if idx not in attacker_list:
                            pert[idx] = 0
                
                pert = pert.detach()
                pert.requires_grad = True
        
        except Exception as e:
            print(f"[BAC] Iteration {iteration} error: {e}")
            break
    
    print(f"[BAC] Optimization completed, pert norm: {pert.detach().norm().item():.4f}")
    
    return pert.detach(), blind_mask_full


def _generate_bac_mask(fafmodule, data, num_agent, ego_agent, attacker_list, device,
                       config, reg_target, anchors_map, gt_max_iou, padded_voxel_points):
    """
    Internal function to generate BAC mask (slow update)
    """
    print("[BAC] Step 1: Differential Detection...")
    
    # Get victim-only result
    data_vic = {k: v for k, v in data.items()}
    data_vic['pert'] = None
    data_vic['no_fuse'] = True
    data_vic['collab_agent_list'] = None
    _, _, _, result_vic = fafmodule.predict_all(data_vic, 1, num_agent=num_agent)
    
    # Get CP result (with one collaborator)
    data_cp = {k: v for k, v in data.items()}
    data_cp['pert'] = None
    data_cp['collab_agent_list'] = [attacker_list[0]] if len(attacker_list) > 0 else None
    data_cp['no_fuse'] = False
    _, _, _, result_cp = fafmodule.predict_all(data_cp, 1, num_agent=num_agent)
    
    # Extract detection boxes from standalone victim perception and collaborative perception.
    Y_single_boxes_raw = extract_detection_boxes_from_result(
        result_vic, config, padded_voxel_points, 
        reg_target, anchors_map, gt_max_iou, ego_agent
    )
    
    Y_cp_boxes_raw = extract_detection_boxes_from_result(
        result_cp, config, padded_voxel_points,
        reg_target, anchors_map, gt_max_iou, ego_agent
    )

    Y_single_boxes = normalize_box_array(Y_single_boxes_raw)
    Y_cp_boxes = normalize_box_array(Y_cp_boxes_raw)
    Y_vic_boxes, Y_nvic_boxes, Y_matched_single, Y_matched_cp = differential_detection(
        Y_single_boxes, Y_cp_boxes, iou_threshold=0.3
    )

    print(
        "[BAC] Differential detection: "
        f"single={len(Y_single_boxes)}, cp={len(Y_cp_boxes)}, "
        f"matched={len(Y_matched_single)}, victim_only={len(Y_vic_boxes)}, "
        f"collaborative_only={len(Y_nvic_boxes)}"
    )
    
    # Step 2: Blind Region Segmentation
    print("[BAC] Step 2: Blind Region Segmentation...")
    
    # Generate confidence map from BEV features
    if padded_voxel_points is not None:
        bev_ego = padded_voxel_points[ego_agent, :, :, :].cpu().numpy()
        # bev_ego shape: [C, H, W] where C=256, H=32, W=32
        # Or could be [H, W, C] depending on format
        
        # Handle different shapes
        if bev_ego.ndim == 3:
            # Take max over channel dimension
            if bev_ego.shape[0] == 256:  # [C, H, W]
                detection_map = np.max(bev_ego, axis=0)  # [H, W]
            elif bev_ego.shape[2] >= 13:  # [H, W, C]
                detection_map = np.max(bev_ego, axis=2)  # [H, W]
            else:
                detection_map = bev_ego[:, :, 0] if bev_ego.shape[2] > 0 else bev_ego.squeeze()
        else:
            detection_map = bev_ego.squeeze()
        
        # Ensure 2D - force if needed
        print(f"[BAC] Detection map shape after processing: {detection_map.shape}")
        if detection_map.ndim > 2:
            # If still >2D, take max over extra dimensions
            while detection_map.ndim > 2:
                detection_map = np.max(detection_map, axis=-1)
            print(f"[BAC] Forced to 2D: {detection_map.shape}")
        
        if detection_map.ndim != 2:
            print(f"[BAC] Warning: detection_map dim {detection_map.ndim}, using fallback")
            detection_map = np.ones((32, 32), dtype=np.float32) * 0.5
        else:
            # Normalize
            if detection_map.max() > detection_map.min():
                detection_map = (detection_map - detection_map.min()) / (detection_map.max() - detection_map.min() + 1e-6)
            else:
                detection_map = np.ones_like(detection_map) * 0.5
            
            # Resize to 32x32 if needed
            if detection_map.shape != (32, 32):
                detection_map = cv2.resize(detection_map, (32, 32))
            
            detection_map = detection_map.astype(np.float32)
    else:
        # Fallback: center-focused
        detection_map = np.ones((32, 32), dtype=np.float32) * 0.5
        detection_map[12:20, 12:20] = 0.8
    
    # Run BRS algorithm
    print(f"[BAC] Running BRS algorithm...")
    try:
        blind_mask = blind_region_segmentation(
            detection_map, Y_vic_boxes, Y_nvic_boxes, 
            grid_size=(32, 32), 
            K_base=8,        # Increased from 6 to expand confident areas faster
            gamma_d=0.15,    # Decreased from 0.3 to reduce distance decay
            confidence_threshold=0.2  # New parameter: lower threshold for CA
        )
        print(f"[BAC] BRS completed successfully")
    except Exception as e:
        import traceback
        print(f"[BAC] Error in BRS: {e}")
        traceback.print_exc()
        # Use fallback mask
        print(f"[BAC] Using fallback mask")
        blind_mask = np.ones((32, 32), dtype=np.float32) * 0.5
        blind_mask[:16, :] = 1.0
    
    # Expand to all agents
    blind_mask_full = np.tile(blind_mask, (num_agent, 1, 1))
    
    blind_ratio = 1.0 - blind_mask.mean()
    confident_ratio = blind_mask.mean()
    
    # Sanity check
    if blind_ratio > 0.95 or blind_ratio < 0.05:
        print(f"[BAC] Warning: Extreme blind ratio ({blind_ratio:.2%}), using balanced mask")
        blind_mask = np.ones((32, 32), dtype=np.float32) * 0.5
        blind_mask[:16, :] = 1.0  # Front half confident
        blind_mask_full = np.tile(blind_mask, (num_agent, 1, 1))
        blind_ratio = 1.0 - blind_mask.mean()
        confident_ratio = blind_mask.mean()
    
    print(f"[BAC] Generated blind mask: blind={blind_ratio:.2%}, confident={confident_ratio:.2%}")
    
    return blind_mask_full
