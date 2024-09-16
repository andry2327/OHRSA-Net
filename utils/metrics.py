import numpy as np
from scipy.spatial import procrustes

def reproject_keypoints3d_to_2d(kps3d, add_visibility=False):
    K = np.array([[1198.4395, 0., 960.], [0., 1198.4395, 175.2], [0., 0., 1.]])
    
    kps3d = kps3d @ K.T
    p2d = kps3d[:, :2] / kps3d[:, [2]]

    # Clip coordinates to image dimensions
    new_p2d = np.zeros_like(p2d)
    new_p2d[:, 1] = np.clip(p2d[:, 0], 0, 1919)  # Image width - 1
    new_p2d[:, 0] = np.clip(p2d[:, 1], 0, 1079)  # Image height - 1

    if add_visibility:
        # add visibility flag
        visibility = np.ones((new_p2d.shape[0], 1))
        new_p2d = np.hstack((new_p2d, visibility))
    
    return new_p2d # IT RETURNS (y, x) coordinates

def compute_metrics(targets, results, right_hand_faces=None):
    """
    Compute the metrics for a batch of validation data.
    
    Args:
    - targets: Ground truth data for the batch (including 2D and 3D keypoints, mesh)
    - results: Predicted data for the batch (including 3D keypoints, mesh)
    - right_hand_faces: Mesh connectivity for the hand (optional for PVE)
    
    Returns:
    - metrics: Dictionary containing metrics for the batch
    """
    
    def compute_kps2d_dist(gt_2d, pred_2d):
        dist = np.linalg.norm(gt_2d - pred_2d, axis=1)
        return np.mean(dist)
    
    def compute_2d_error(gt_2d, kps3d, img_size=(1920, 1080)): # in pixels
        # Project 3d points in 2D
        p2d = reproject_keypoints3d_to_2d(kps3d) # returns (y, x points)
        p2d[:, [0, 1]] = p2d[:, [1, 0]] # Swap (y, x) -> (x, y)
        # p2d[:, 0] = p2d[:, 0] / img_size[0] # Normalize x values
        # p2d[:, 1] = p2d[:, 1] / img_size[1] # Normalize y values
        
        gt_2d[:, 0] = gt_2d[:, 0] * img_size[0]
        gt_2d[:, 1] = gt_2d[:, 1] * img_size[1]

        error = np.linalg.norm(gt_2d - p2d, axis=1)
        return np.mean(error)

    def compute_mpjpe(gt_3d, pred_3d):
        error = np.linalg.norm(gt_3d - pred_3d, axis=1)
        return np.mean(error)
    
    def compute_pve(gt_vertices, pred_vertices):
        error = np.linalg.norm(gt_vertices - pred_vertices, axis=1)
        return np.mean(error)
    
    def compute_procrustes_aligned_error(gt, pred):
        mtx1, mtx2, disparity = procrustes(gt, pred)
        error = np.linalg.norm(mtx1 - mtx2, axis=1)
        return np.mean(error)

    batch_metrics = {
        'D2d': 0.0,
        'P2d': 0.0,
        'MPJPE': 0.0,
        'PVE': 0.0,
        'PA-MPJPE': 0.0,
        'PA-PVE': 0.0
    }

    num_samples = len(targets)
    
    for i in range(num_samples):
        # Extract ground truth and predictions
        gt_2d = targets[i]['keypoints'].squeeze().cpu().numpy()
        gt_2d[:, [0, 1]] = gt_2d[:, [1, 0]] # Swap from (y, x) to (x, y)
        gt_2d = gt_2d[:, :2] # Remove visibility flag
        pred_2d = results['keypoint2d'][i].detach().cpu().numpy()
        
        gt_3d = targets[i]['keypoints3d'].squeeze().cpu().numpy()
        pred_3d = results['keypoint3d'][i].detach().cpu().numpy()
        
        gt_vertices = targets[i]['mesh3d'].squeeze().cpu().numpy()
        pred_vertices = results['mesh3d'][i][:, :3].detach().cpu().numpy()
        
        # Compute metrics
        batch_metrics['D2d'] += compute_kps2d_dist(gt_2d, pred_2d)
        batch_metrics['P2d'] += compute_2d_error(gt_2d, pred_3d) # in pixels
        batch_metrics['MPJPE'] += compute_mpjpe(gt_3d, pred_3d)
        batch_metrics['PVE'] += compute_pve(gt_vertices, pred_vertices)
        batch_metrics['PA-MPJPE'] += compute_procrustes_aligned_error(gt_3d, pred_3d)
        batch_metrics['PA-PVE'] += compute_procrustes_aligned_error(gt_vertices, pred_vertices)

    # Average metrics over the batch
    for key in batch_metrics:
        batch_metrics[key] /= num_samples

    return batch_metrics

# Accumulate metrics across batches during validation
def accumulate_metrics(metrics, batch_metrics, num_batches):
    """
    Accumulate metrics over multiple batches.
    
    Args:
    - metrics: Accumulated metrics over the validation set
    - batch_metrics: Metrics from the current batch
    - num_batches: Number of batches processed so far
    
    Returns:
    - Updated metrics with batch_metrics added and averaged
    """
    for key in metrics:
        metrics[key] += batch_metrics[key]
    
    # Return averaged metrics across the batches processed so far
    return {key: metrics[key] / num_batches for key in metrics}