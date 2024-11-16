import numpy as np


def calculate_metric_on_batch(output_np, target_np, segment_length=1.0):
    """
    output_np: numpy array of shape [batch_size, seq_len, 4], predicted x, y, yaw
    target_np: numpy array of same shape, ground truth x, y, yaw

    Returns:
        metric: float, the average metric over the batch
    """
    x_pred, y_pred, yaw_pred = output_np[..., 0], output_np[..., 1], output_np[..., 2]
    x_gt, y_gt, yaw_gt = target_np[..., 0], target_np[..., 1], target_np[..., 2]

    # Compute c1 and c2 for predicted
    c1_pred = np.stack([x_pred, y_pred], axis=-1)
    c2_pred = c1_pred + segment_length * np.stack(
        [np.cos(yaw_pred), np.sin(yaw_pred)], axis=-1
    )

    # Compute c1 and c2 for ground truth
    c1_gt = np.stack([x_gt, y_gt], axis=-1)
    c2_gt = c1_gt + segment_length * np.stack([np.cos(yaw_gt), np.sin(yaw_gt)], axis=-1)

    # Compute distances between corresponding points
    dist_c1 = np.linalg.norm(c1_pred - c1_gt, axis=-1)
    dist_c2 = np.linalg.norm(c2_pred - c2_gt, axis=-1)

    # Compute pose metric
    pose_metric = np.sqrt((dist_c1**2 + dist_c2**2) / 2.0)
    metric = np.mean(pose_metric)

    return metric
