import cv2
import numpy as np
import os

class RetinalVesselTrimap:
    def create_trimap(self, image, gt_mask, roi_mask=None):
        """개선된 트리맵 생성 함수"""
        # Ground truth 이진화 (임계값 낮춤)
        _, gt_binary = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)  # 임계값 수정
        
        # 1. 전경(확실한 혈관) 영역 생성
        kernel_vessel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # 커널 크기 수정
        sure_vessels = cv2.erode(gt_binary, kernel_vessel, iterations=2)  # iterations 수정
        
        # 혈관 보존을 위한 추가 처리
        thin_vessels = cv2.adaptiveThreshold(
            gt_mask,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        # 가는 혈관 보존
        sure_vessels = cv2.bitwise_or(sure_vessels, 
                                    cv2.bitwise_and(thin_vessels, gt_binary))
        
        # 2. Unknown 영역 생성
        # 외부 경계 생성 (커널 크기 증가)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 커널 크기 수정
        dilated = cv2.dilate(gt_binary, kernel_dilate, iterations=1)
        
        # 내부 경계 생성
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(gt_binary, kernel_erode, iterations=1)
        
        # Unknown 영역 = 팽창 영역 - 침식 영역
        unknown = cv2.subtract(dilated, eroded)
        
        # 혈관 경계 강화
        edges = cv2.Canny(gt_binary, 100, 200)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel_edge, iterations=1)  # iterations 수정
        unknown = cv2.bitwise_or(unknown, edges_dilated)
        
        # 트리맵 초기화
        trimap = np.zeros_like(image, dtype=np.uint8)
        
        # 영역 설정
        trimap[unknown > 0] = 128      # Unknown 영역
        trimap[sure_vessels > 0] = 255 # 확실한 혈관
        
        # ROI 마스크 적용
        if roi_mask is not None:
            trimap[roi_mask == 0] = 0
        
        # 후처리
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # 전경 영역 정리
        fg_mask = (trimap == 255).astype(np.uint8)
        fg_cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_clean)
        
        # Unknown 영역 정리
        unknown_mask = (trimap == 128).astype(np.uint8)
        unknown_cleaned = cv2.morphologyEx(unknown_mask, cv2.MORPH_CLOSE, kernel_clean)
        
        # 작은 고립 영역 제거
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(unknown_cleaned, 8)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 20:  # 면적 임계값 수정
                unknown_cleaned[labels == i] = 0
        
        # 최종 trimap 생성
        trimap = np.zeros_like(image, dtype=np.uint8)
        trimap[unknown_cleaned > 0] = 128
        trimap[fg_cleaned > 0] = 255
        
        # 마지막으로 ROI 마스크 다시 적용
        if roi_mask is not None:
            trimap[roi_mask == 0] = 0
            
        return trimap

def calculate_statistics(gt_mask, trimap, roi_mask):
    """통계 계산 함수"""
    valid_pixels = np.sum(roi_mask > 0)
    
    gt_vessel_ratio = np.sum((gt_mask > 127) & (roi_mask > 0)) / valid_pixels * 100
    trimap_fg_ratio = np.sum((trimap == 255) & (roi_mask > 0)) / valid_pixels * 100
    trimap_unknown_ratio = np.sum((trimap == 128) & (roi_mask > 0)) / valid_pixels * 100
    
    return gt_vessel_ratio, trimap_fg_ratio, trimap_unknown_ratio

def process_single_image():
    # 경로 설정
    DATASET_PATH = "./CHASE_DB1/train"
    OUTPUT_PATH = "./train_output"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # 첫 번째 이미지 파일 경로
    image_path = os.path.join(DATASET_PATH, "images", "01_test.tif")
    gt_path = os.path.join(DATASET_PATH, "1st_manual", "01_manual1.tif")
    mask_path = os.path.join(DATASET_PATH, "mask", "mask_01L.png")
    
    # 이미지 로드
    print("Loading images...")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    roi_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None or gt is None or roi_mask is None:
        print("Error: Failed to load images. Check file paths:")
        print(f"Image path: {image_path}")
        print(f"GT path: {gt_path}")
        print(f"Mask path: {mask_path}")
        return
    
    # 트리맵 생성
    print("Generating trimap...")
    trimap_generator = RetinalVesselTrimap()
    trimap = trimap_generator.create_trimap(image, gt, roi_mask)
    
    # 결과 저장
    print("Saving results...")
    # trimap 저장
    output_trimap_path = os.path.join(OUTPUT_PATH, "01_train_trimap1.png")
    cv2.imwrite(output_trimap_path, trimap)

    # ground truth 저장 (참고용)
    cv2.imwrite(os.path.join(OUTPUT_PATH, "ground_truth.png"), gt)
    
    # 통계 계산
    gt_ratio, fg_ratio, unknown_ratio = calculate_statistics(gt, trimap, roi_mask)
    
    # 통계 출력
    print("\n=== 통계 정보 ===")
    print(f"Ground Truth 혈관 비율: {gt_ratio:.2f}%")
    print(f"Trimap 전경 비율: {fg_ratio:.2f}%")
    print(f"Trimap Unknown 비율: {unknown_ratio:.2f}%")
    
    # Trimap 픽셀값 분포
    unique_values, counts = np.unique(trimap, return_counts=True)
    print("\n=== Trimap 픽셀값 분포 ===")
    for value, count in zip(unique_values, counts):
        print(f"값 {value}: {count}개 픽셀")
    
    print("\nProcessing complete!")
    print(f"Results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    process_single_image()