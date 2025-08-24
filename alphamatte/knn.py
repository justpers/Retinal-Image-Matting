import numpy as np
import sklearn.neighbors
import scipy.sparse
import scipy.ndimage
import warnings
import cv2
from PIL import Image
import os

# 평가 지표 함수들
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
    dist_maps = np.zeros((dimy, dimx, len(thresh_steps)), dtype=np.float32)

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

        dist_maps[:, :, i] = scipy.ndimage.distance_transform_edt(1 - omega)
        if np.max(dist_maps[:, :, i]) > 0:
            dist_maps[:, :, i] /= np.max(dist_maps[:, :, i])

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map

    pred_phi = 1 - pred_d * (pred_d >= 0.15)
    target_phi = 1 - target_d * (target_d >= 0.15)

    loss = np.sum(np.abs(pred_phi - target_phi) * (trimap == 128).astype(np.float32))
    return loss

# knn_matte 함수만 수정
def knn_matte(img, trimap, mylambda=100):
    [m, n, c] = img.shape
    img = img / 255.0
    trimap = trimap / 255.0
    
    # trimap을 2D로 유지
    foreground = (trimap > 0.99).astype(int)
    background = (trimap < 0.01).astype(int)
    all_constraints = foreground + background

    print('최근접 이웃을 찾는 중...')
    a, b = np.unravel_index(np.arange(m * n), (m, n))
    feature_vec = np.column_stack([img.reshape(m * n, c), 
                                 a / np.sqrt(m * m + n * n),
                                 b / np.sqrt(m * m + n * n)])

    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(feature_vec)
    knns = nbrs.kneighbors(feature_vec)[1]

    print('희소 행렬 A 계산 중...')
    row_inds = np.repeat(np.arange(m * n), 10)
    col_inds = knns.reshape(m * n * 10)
    vals = 1 - np.linalg.norm(feature_vec[row_inds] - feature_vec[col_inds], axis=1) / (c + 2)
    A = scipy.sparse.coo_matrix((vals, (row_inds, col_inds)), shape=(m * n, m * n))

    D_script = scipy.sparse.diags(np.ravel(A.sum(axis=1)))
    L = D_script - A
    D = scipy.sparse.diags(np.ravel(all_constraints))  # 2D trimap 사용
    v = np.ravel(foreground)  # 2D foreground 사용
    const = 2 * mylambda * np.transpose(v)
    H = 2 * (L + mylambda * D)

    print('알파 값에 대한 선형 시스템 해결 중...')
    warnings.filterwarnings('error')
    try:
        alpha = np.clip(scipy.sparse.linalg.spsolve(H, const), 0, 1).reshape(m, n)
    except (scipy.sparse.linalg.MatrixRankWarning, np.linalg.LinAlgError):
        print('경고: spsolve 실패, lsqr 방법으로 전환')
        x = scipy.sparse.linalg.lsqr(H, const)[0]
        alpha = np.clip(x, 0, 1).reshape(m, n)
    return alpha

def process_image():
    print("\n=== KNN Matting 처리 시작 ===\n")
    
    # 경로 설정
    DATASET_PATH = "./CHASE_DB1/train"
    INPUT_PATH = "./train_output"
    OUTPUT_PATH = "./knn_output"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"출력 디렉토리 생성: {OUTPUT_PATH}")

    # 파일 경로
    image_path = os.path.join(DATASET_PATH, "images", "01_test.tif")
    trimap_path = os.path.join(INPUT_PATH, "trimap_resized.png")
    gt_path = os.path.join(DATASET_PATH, "1st_manual", "01_manual1.tif")

    # 이미지 로드
    print("\n이미지를 불러오는 중...")
    try:
        img = cv2.cvtColor(np.array(Image.open(image_path).convert('RGB')), cv2.COLOR_RGB2BGR)
        trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)
        gt_alpha = np.array(Image.open(gt_path).convert('L'))
    except Exception as e:
        print(f"이미지 로드 중 오류 발생: {e}")
        return

    # 알파 매트 생성
    print("\n알파 매트를 생성하는 중...")
    alpha = knn_matte(img, trimap)
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