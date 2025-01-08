import numpy as np
from scipy.ndimage import label, distance_transform_edt, gaussian_gradient_magnitude

def compute_mse_loss(pred, target, trimap):
    error_map = (pred - target) / 255.0
    loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)

    return loss


def compute_sad_loss(pred, target, trimap):
    error_map = np.abs((pred - target) / 255.0)
    loss = np.sum(error_map * (trimap == 128))

    return loss / 1000, np.sum(trimap == 128) / 1000


def compute_connectivity_error(pred, target, trimap, step=0.1):
    """
    Compute the connectivity error given a prediction, a ground truth, and a trimap.

    Parameters:
        pred (np.ndarray): The predicted alpha matte.
        target (np.ndarray): The ground truth alpha matte.
        trimap (np.ndarray): The trimap (0: background, 128: unknown, 255: foreground).
        step (float): The step size for thresholding (default: 0.1).

    Returns:
        float: The connectivity error.
    """
    pred = pred.astype(np.float32) / 255.0
    target = target.astype(np.float32) / 255.0
    dimy, dimx = pred.shape

    # Threshold steps
    thresh_steps = np.arange(0, 1 + step, step)
    l_map = np.full_like(pred, -1.0, dtype=np.float32)
    dist_maps = np.zeros((dimy, dimx, len(thresh_steps)), dtype=np.float32)

    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = pred >= thresh_steps[i]
        target_alpha_thresh = target >= thresh_steps[i]

        # Connected components
        intersection = pred_alpha_thresh & target_alpha_thresh
        labeled, num_features = label(intersection)

        if num_features == 0:
            continue

        # Find the largest connected component
        sizes = np.array([np.sum(labeled == j) for j in range(1, num_features + 1)])
        max_id = np.argmax(sizes) + 1  # +1 because labels start from 1
        omega = (labeled == max_id).astype(np.float32)

        # Update l_map
        flag = (l_map == -1) & (omega == 0)
        l_map[flag] = thresh_steps[i - 1]

        # Compute distance map and normalize
        dist_maps[:, :, i] = distance_transform_edt(1 - omega)
        if np.max(dist_maps[:, :, i]) > 0:
            dist_maps[:, :, i] /= np.max(dist_maps[:, :, i])

    l_map[l_map == -1] = 1

    # Compute phi values
    pred_d = pred - l_map
    target_d = target - l_map

    pred_phi = 1 - pred_d * (pred_d >= 0.15)
    target_phi = 1 - target_d * (target_d >= 0.15)

    # Compute connectivity error
    loss = np.sum(np.abs(pred_phi - target_phi) * (trimap == 128).astype(np.float32))
    return loss


def compute_gradient_loss(pred, target, trimap, sigma=1.4):
    """
    Compute the gradient error given a prediction, a ground truth, and a trimap.

    Parameters:
        pred (np.ndarray): The predicted alpha matte (grayscale image, 0-255).
        target (np.ndarray): The ground truth alpha matte (grayscale image, 0-255).
        trimap (np.ndarray): The trimap (0: background, 128: unknown, 255: foreground).
        sigma (float): The standard deviation for Gaussian kernel (default: 1.4).

    Returns:
        float: The gradient error.
    """
    # Normalize the input images to the range [0, 1]
    pred = pred.astype(np.float32) / 255.0
    target = target.astype(np.float32) / 255.0

    # Compute gradient magnitude using Gaussian smoothing
    pred_gradient = gaussian_gradient_magnitude(pred, sigma=sigma)
    target_gradient = gaussian_gradient_magnitude(target, sigma=sigma)

    # Compute error map (difference between gradients)
    error_map = (pred_gradient - target_gradient) ** 2

    # Compute loss only in the unknown region (trimap == 128)
    loss = np.sum(error_map * (trimap == 128).astype(np.float32))

    return loss