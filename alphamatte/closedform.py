import os
import cv2
import numpy as np
from PIL import Image
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg
from numpy.lib.stride_tricks import as_strided


def _rolling_block(A, block=(3, 3)):
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides, writeable=False)

def compute_laplacian(img, mask=None, eps=1e-7, win_rad=1):
    win_size = (win_rad * 2 + 1) ** 2
    h, w, d = img.shape
    win_diam = win_rad * 2 + 1

    indsM = np.arange(h * w).reshape((h, w))
    ravelImg = img.reshape(h * w, d)
    win_inds = _rolling_block(indsM, block=(win_diam, win_diam)).reshape(-1, win_size)

    if mask is not None:
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((win_diam, win_diam), np.uint8)).astype(bool)
        win_mask = np.sum(mask.ravel()[win_inds], axis=1) > 0
        win_inds = win_inds[win_mask]

    winI = ravelImg[win_inds]
    win_mu = np.mean(winI, axis=1, keepdims=True)
    win_var = np.einsum('...ji,...jk ->...ik', winI, winI) / win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)
    inv = np.linalg.pinv(win_var + (eps / win_size) * np.eye(3))
    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    vals = np.eye(win_size) - (1.0 / win_size) * (1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))

    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h * w, h * w))
    return L

def closed_form_matting_with_trimap(image, trimap, trimap_confidence=100.0):
    consts_map = (trimap < 0.1) | (trimap > 0.9)
    laplacian = compute_laplacian(image, ~consts_map)
    confidence = scipy.sparse.diags((trimap_confidence * consts_map).ravel())
    solution = scipy.sparse.linalg.spsolve(laplacian + confidence, trimap.ravel() * trimap_confidence * consts_map.ravel())
    alpha = np.clip(solution.reshape(trimap.shape), 0, 1)
    return alpha

def compute_mse_loss(pred, target, trimap):
    error_map = (pred - target) / 255.0
    loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)
    return loss

def compute_sad_loss(pred, target, trimap):
    error_map = np.abs((pred - target) / 255.0)
    loss = np.sum(error_map * (trimap == 128))
    return loss / 1000, np.sum(trimap == 128) / 1000

def compute_gradient_loss(pred, target, trimap, sigma=1.4):
    pred = pred.astype(np.float32) / 255.0
    target = target.astype(np.float32) / 255.0
    pred_gradient = scipy.ndimage.gaussian_gradient_magnitude(pred, sigma=sigma)
    target_gradient = scipy.ndimage.gaussian_gradient_magnitude(target, sigma=sigma)
    error_map = (pred_gradient - target_gradient) ** 2
    loss = np.sum(error_map * (trimap == 128).astype(np.float32))
    return loss

def compute_connectivity_error(pred, target, trimap, step=0.1):
    pred = pred.astype(np.float32) / 255.0
    target = target.astype(np.float32) / 255.0
    dimy, dimx = pred.shape

    thresh_steps = np.arange(0, 1 + step, step)
    l_map = np.full_like(pred, -1.0, dtype=np.float32)

    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = pred >= thresh_steps[i]
        target_alpha_thresh = target >= thresh_steps[i]
        intersection = pred_alpha_thresh & target_alpha_thresh
        labeled, num_features = scipy.ndimage.label(intersection)
        if num_features == 0:
            continue
        sizes = np.array([np.sum(labeled == j) for j in range(1, num_features + 1)])
        max_id = np.argmax(sizes) + 1
        omega = (labeled == max_id).astype(np.float32)
        flag = (l_map == -1) & (omega == 0)
        l_map[flag] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15)
    target_phi = 1 - target_d * (target_d >= 0.15)
    loss = np.sum(np.abs(pred_phi - target_phi) * (trimap == 128).astype(np.float32))
    return loss

def process_image():
    print("\n=== Closed-Form Matting 처리 시작 ===\n")

    DATASET_PATH = "./CHASE_DB1/train"
    INPUT_PATH = "./train_output"
    OUTPUT_PATH = "./closedform_output"
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    image_path = os.path.join(DATASET_PATH, "images", "01_test.tif")
    trimap_path = os.path.join(INPUT_PATH, "01_train_trimap3_4.png")
    gt_path = os.path.join(DATASET_PATH, "1st_manual", "01_manual1.tif")

    print("\n이미지를 불러오는 중...")
    try:
        img = cv2.cvtColor(np.array(Image.open(image_path).convert('RGB')), cv2.COLOR_RGB2BGR) / 255.0
        trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE) / 255.0
        gt_alpha = np.array(Image.open(gt_path).convert('L'))
    except Exception as e:
        print(f"이미지 로드 중 오류 발생: {e}")
        return

    print("\n알파 매트를 생성하는 중...")
    alpha = closed_form_matting_with_trimap(img, trimap)
    alpha_uint8 = (alpha * 255).astype(np.uint8)
    print(f"생성된 알파 매트 shape: {alpha_uint8.shape}")

    print("\n평가 지표를 계산하는 중...")
    mse = compute_mse_loss(alpha_uint8, gt_alpha, trimap * 255)
    sad, _ = compute_sad_loss(alpha_uint8, gt_alpha, trimap * 255)
    grad = compute_gradient_loss(alpha_uint8, gt_alpha, trimap * 255)
    conn = compute_connectivity_error(alpha_uint8, gt_alpha, trimap * 255)

    print("\n=== 평가 결과 ===")
    print(f"MSE: {mse:.4f}")
    print(f"SAD: {sad:.4f}")
    print(f"Grad: {grad:.4f}")
    print(f"Conn: {conn:.4f}")

    print("\n결과를 저장하는 중...")
    h, w, _ = img.shape
    blended = (alpha.reshape(h, w, 1).repeat(3, axis=2) * (img * 255)).astype(np.uint8)

    trimap_num = os.path.basename(trimap_path).split('_trimap')[-1].split('.')[0]
    try:
        num = int(trimap_num)
        suffix = f"tri{num:02d}"
    except ValueError:
        suffix = "tri01"

    output_paths = {
        f"alpha_matte_{suffix}.png": alpha_uint8,
        f"blended_result_{suffix}.png": blended,
        f"visualization_{suffix}.png": np.hstack([
            (img * 255).astype(np.uint8),
            cv2.cvtColor(alpha_uint8, cv2.COLOR_GRAY2BGR),
            blended
        ])
    }

    for filename, image in output_paths.items():
        output_path = os.path.join(OUTPUT_PATH, filename)
        cv2.imwrite(output_path, image)
        print(f"저장 완료: {output_path}")

if __name__ == "__main__":
    process_image()
