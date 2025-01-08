import os
from PIL import Image
import numpy as np
import cv2

def read_image(file_path, mode='RGB'):
    """이미지를 불러오는 함수"""
    try:
        with Image.open(file_path) as img:
            if mode == 'RGB':
                img = img.convert('RGB')  # 원본 색상 유지
            elif mode == 'L':
                img = img.convert('L')  # Grayscale 모드
            return np.array(img)
    except Exception as e:
        print(f"[ERROR] Cannot open file {file_path}: {e}")
        return None

def create_background_image(image, mask):
    """배경 이미지 생성"""
    mask_3d = np.stack([mask] * 3, axis=-1)  # Grayscale 마스크를 3채널로 변환
    background = np.where(mask_3d == 0, image, 0)
    return background

def create_alpha_matte(mask):
    """Alpha Matte 생성"""
    alpha_matte = (mask > 0).astype(np.uint8) * 255
    return alpha_matte

def create_compose_image(image, background, alpha_matte):
    """합성 이미지 생성"""
    alpha = alpha_matte.astype(float) / 255.0
    composed_image = (alpha[..., np.newaxis] * image + (1 - alpha[..., np.newaxis]) * background).astype(np.uint8)
    return composed_image

def create_trimap(alpha_matte, kernel_size=5):
    """Trimap 생성"""
    dilated = cv2.dilate(alpha_matte, np.ones((kernel_size, kernel_size), np.uint8))
    eroded = cv2.erode(alpha_matte, np.ones((kernel_size, kernel_size), np.uint8))
    trimap = np.zeros_like(alpha_matte)
    trimap[dilated == 255] = 255
    trimap[eroded == 0] = 0
    trimap[(dilated != 255) & (eroded != 0)] = 128
    return trimap

def process_dataset(input_dir, output_dir):
    """데이터셋 생성"""
    categories = ["background", "image", "alpha_matte", "compose_image", "trimap", "gt_alpha_matte"]
    for category in categories:
        os.makedirs(os.path.join(output_dir, category), exist_ok=True)

    images_dir = os.path.join(input_dir, "images")
    masks_dir = os.path.join(input_dir, "mask")
    gt_alpha_dir = os.path.join(input_dir, "1st_manual")

    image_files = os.listdir(images_dir)
    mask_files = os.listdir(masks_dir)
    gt_alpha_files = os.listdir(gt_alpha_dir)

    for image_file, mask_file, gt_alpha_file in zip(sorted(image_files), sorted(mask_files), sorted(gt_alpha_files)):
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, mask_file)
        gt_alpha_path = os.path.join(gt_alpha_dir, gt_alpha_file)

        # 이미지 읽기
        image = read_image(image_path, mode='RGB')
        mask = read_image(mask_path, mode='L')  # Grayscale
        gt_alpha = read_image(gt_alpha_path, mode='L')

        if image is None or mask is None or gt_alpha is None:
            print(f"[WARN] Skipping: {image_file} (failed to read data)")
            continue

        # 데이터 생성
        background_image = create_background_image(image, mask)
        alpha_matte = create_alpha_matte(mask)
        composed_image = create_compose_image(image, background_image, alpha_matte)
        trimap = create_trimap(alpha_matte)

        base_name = os.path.splitext(image_file)[0]
        cv2.imwrite(os.path.join(output_dir, "background", f"{base_name}_background.png"), cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, "image", f"{base_name}_image.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, "alpha_matte", f"{base_name}_alpha_matte.png"), alpha_matte)
        cv2.imwrite(os.path.join(output_dir, "compose_image", f"{base_name}_compose_image.png"), cv2.cvtColor(composed_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, "trimap", f"{base_name}_trimap.png"), trimap)
        cv2.imwrite(os.path.join(output_dir, "gt_alpha_matte", f"{base_name}_gt_alpha_matte.png"), gt_alpha)

        print(f"[INFO] Processed: {image_file}")

if __name__ == "__main__":
    base_input_dir = "Retinal-Vessel-Dataset/CHASE_DB1"
    base_output_dir = "Retinal-Vessel-Dataset/CHASE_DB1/data"

    process_dataset(os.path.join(base_input_dir, "train"), os.path.join(base_output_dir, "train"))
    process_dataset(os.path.join(base_input_dir, "test"), os.path.join(base_output_dir, "test"))
