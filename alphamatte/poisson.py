from __future__ import division
import numpy as np
import scipy.ndimage
import imageio.v2 as imageio
import cv2
import matplotlib.pyplot as plt
from numba import jit
from PIL import Image
import os

# 평가 지표 함수들
def compute_mse_loss(pred, target, trimap):
    """알 수 없는 영역에서의 평균 제곱 오차(MSE) 계산"""
    error_map = (pred - target) / 255.0
    loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)
    return loss

def compute_sad_loss(pred, target, trimap):
    """알 수 없는 영역에서의 절대 차이 합계(SAD) 계산"""
    error_map = np.abs((pred - target) / 255.0)
    loss = np.sum(error_map * (trimap == 128))
    return loss / 1000, np.sum(trimap == 128) / 1000

def compute_gradient_loss(pred, target, trimap, sigma=1.4):
    """알 수 없는 영역에서의 그래디언트 손실 계산"""
    pred = pred.astype(np.float32) / 255.0
    target = target.astype(np.float32) / 255.0
    pred_gradient = scipy.ndimage.gaussian_gradient_magnitude(pred, sigma=sigma)
    target_gradient = scipy.ndimage.gaussian_gradient_magnitude(target, sigma=sigma)
    error_map = (pred_gradient - target_gradient) ** 2
    loss = np.sum(error_map * (trimap == 128).astype(np.float32))
    return loss

def compute_connectivity_error(pred, target, trimap, step=0.1):
    """알 수 없는 영역에서의 연결성 오차 계산"""
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
        labeled, num_features = scipy.ndimage.label(intersection)

        if num_features == 0:
            continue

        # Find the largest connected component
        sizes = np.array([np.sum(labeled == j) for j in range(1, num_features + 1)])
        max_id = np.argmax(sizes) + 1
        omega = (labeled == max_id).astype(np.float32)

        # Update l_map
        flag = (l_map == -1) & (omega == 0)
        l_map[flag] = thresh_steps[i - 1]

        # Compute distance map and normalize
        dist_maps[:, :, i] = scipy.ndimage.distance_transform_edt(1 - omega)
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

@jit(nopython=True)
def computeAlphaJit(alpha, b, unknown):
    """야코비 반복법을 사용하여 알파 매트 계산"""
    h, w = unknown.shape
    alphaNew = alpha.copy()
    alphaOld = np.zeros(alphaNew.shape)
    threshold = 0.1
    n = 1
    # 수렴할 때까지 반복
    while n < 50 and np.sum(np.abs(alphaNew - alphaOld)) > threshold:
        alphaOld = alphaNew.copy()
        for i in range(1, h-1):
            for j in range(1, w-1):
                if unknown[i, j]:
                    # 포아송 방정식의 이산화된 형태
                    alphaNew[i, j] = 1/4 * (alphaNew[i-1, j] + alphaNew[i, j-1] + 
                                          alphaOld[i, j+1] + alphaOld[i+1, j] - b[i, j])
        n += 1
    return alphaNew

def poisson_matte(gray_img, trimap):
    """포아송 매팅을 사용하여 알파 매트 생성"""
    h, w = gray_img.shape
    # 영역 분류
    fg = trimap == 255  # 전경
    bg = trimap == 0    # 배경
    unknown = np.logical_not(np.logical_or(fg, bg))  # 알 수 없는 영역

    # 전경과 배경 이미지 초기화
    fg_img = gray_img * fg
    bg_img = gray_img * bg
    alphaEstimate = fg + 0.5 * unknown  # 초기 알파 추정값

    # 전경과 배경 근사화
    approx_bg = cv2.inpaint(bg_img.astype(np.uint8), 
                           (unknown + fg).astype(np.uint8) * 255, 
                           3, cv2.INPAINT_TELEA) * (np.logical_not(fg)).astype(np.float32)
    approx_fg = cv2.inpaint(fg_img.astype(np.uint8), 
                           (unknown + bg).astype(np.uint8) * 255, 
                           3, cv2.INPAINT_TELEA) * (np.logical_not(bg)).astype(np.float32)

    # F-B 이미지 스무딩
    approx_diff = approx_fg - approx_bg
    approx_diff = scipy.ndimage.gaussian_filter(approx_diff, 0.9)

    # 0으로 나누기 방지
    approx_diff = np.where(approx_diff == 0, 1e-6, approx_diff)
    approx_diff = np.nan_to_num(approx_diff, nan=1e-6)

    # 그래디언트 계산
    dy, dx = np.gradient(gray_img)
    d2y, _ = np.gradient(dy / approx_diff)
    _, d2x = np.gradient(dx / approx_diff)

    b = d2y + d2x

    # 포아송 방정식 해결
    alpha = computeAlphaJit(alphaEstimate, b, unknown)
    alpha = np.clip(alpha, 0, 1).reshape(h, w)
    return alpha

def process_image():
    print("\n=== Poisson Matting 처리 시작 ===\n")
    
    # 경로 설정
    DATASET_PATH = "./CHASE_DB1/train"
    INPUT_PATH = "./train_output"
    OUTPUT_PATH = "./poisson_matting"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"출력 디렉토리 생성: {OUTPUT_PATH}")

    # 파일 경로
    image_path = os.path.join(DATASET_PATH, "images", "01_test.tif")
    trimap_path = os.path.join(INPUT_PATH, "01_train_trimap3_4.png")
    gt_path = os.path.join(DATASET_PATH, "1st_manual", "01_manual1.tif")

    # 이미지 로드
    print("\n이미지를 불러오는 중...")
    try:
        # RGB로 로드 후 BGR로 변환
        img = cv2.cvtColor(np.array(Image.open(image_path).convert('RGB')), cv2.COLOR_RGB2BGR)
        # print(f"원본 이미지 shape: {img.shape}")
        gray_img = np.array(Image.open(image_path).convert('L'))
        # print(f"grayscale 이미지 shape: {gray_img.shape}")
        trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)
        # print(f"trimap shape: {trimap.shape}")
        gt_alpha = np.array(Image.open(gt_path).convert('L'))
        # print(f"ground truth shape: {gt_alpha.shape}")
    except Exception as e:
        print(f"이미지 로드 중 오류 발생: {e}")
        return

    # 알파 매트 생성
    print("\n알파 매트를 생성하는 중...")
    alpha = poisson_matte(gray_img, trimap)
    alpha_uint8 = (alpha * 255).astype(np.uint8)
    print(f"생성된 알파 매트 shape: {alpha_uint8.shape}")

    # 평가 지표 계산
    print("\n평가 지표를 계산하는 중...")
    mse = compute_mse_loss(alpha_uint8, gt_alpha, trimap)
    sad, _ = compute_sad_loss(alpha_uint8, gt_alpha, trimap)
    grad = compute_gradient_loss(alpha_uint8, gt_alpha, trimap)
    conn = compute_connectivity_error(alpha_uint8, gt_alpha, trimap)

    print("\n=== 평가 결과 ===")
    print(f"MSE: {mse:.4f}")
    print(f"SAD: {sad:.4f}")
    print(f"Grad: {grad:.4f}")
    print(f"Conn: {conn:.4f}")

    # 결과 저장
    print("\n결과를 저장하는 중...")
    h, w, _ = img.shape
    blended = (alpha.reshape(h, w, 1).repeat(3, axis=2) * img).astype(np.uint8)
    
    # trimap 번호 추출 및 포맷팅
    trimap_num = os.path.basename(trimap_path).split('_trimap')[-1].split('.')[0]
    try:
        num = int(trimap_num)
        suffix = f"tri{num:02d}"
    except ValueError:
        suffix = "tri01"
    
    # 결과 저장
    output_paths = {
        f"alpha_matte_{suffix}.png": alpha_uint8,
        f"blended_result_{suffix}.png": blended,
        f"visualization_{suffix}.png": np.hstack([img, cv2.cvtColor(alpha_uint8, cv2.COLOR_GRAY2BGR), blended])
    }

    for filename, image in output_paths.items():
        output_path = os.path.join(OUTPUT_PATH, filename)
        cv2.imwrite(output_path, image)
        print(f"저장 완료: {output_path}")

if __name__ == "__main__":
    process_image()