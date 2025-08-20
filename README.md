# Retinal Image Matting (망막 이미지 매팅)

본 프로젝트는 **망막 영상에서 혈관 구조를 정밀하게 분리**하기 위해 **Matting 기법**을 적용한 연구입니다. 기존의 단순한 세그멘테이션(segmentation) 방식보다 더 세밀한 **alpha matte**를 생성하여, 혈관 주변 경계 부분까지 정확히 추출하는 것을 목표로 하였습니다.  

---

## 📌 Problem
- 망막 영상 분석에서 혈관의 경계 부분은 흐릿하게 표현되어 **기존 segmentation 방식으로는 정밀 분리가 어려움**  
- 세그멘테이션은 픽셀 단위 0/1 이진 분류만 제공 → 경계선 정보 손실  
- 정밀 진단 및 후속 분석을 위해 **alpha matte 기반의 부드러운 경계 표현** 필요  

---

## 🔎 Approach
1. **데이터 준비**
   - Retinal 이미지 수집 (Fundus 사진 기반)
   - Photoshop을 활용해 **수작업 GT alpha matte** 생성
   - Trimap 생성 (FG/BG/Unknown 영역 분리)

2. **모델 선정 및 학습**
   - IndexNet Matting, Information Flow Matting(IFM) 등 기존 matting 모델 검토  
   - Retinal dataset에 맞춰 **Fine-tuning** 진행

3. **실험**
   - 원본 Retinal 이미지 + Trimap 입력 → Alpha Matte 출력  
   - Baseline segmentation과 성능 비교  

---

## 🛠 Tools & Tech
- Python, PyTorch  
- OpenCV, Pillow  
- Pretrained Matting Models (IndexNet, IFM)  
- Adobe Photoshop (GT 제작)  

---

## 📈 Result
- Retinal 영상에 Matting 기법을 적용하여 **혈관 경계 추출 실험**을 진행함
- IndexNet, AEMatter, MatteFormer 등의 모델을 파인튜닝하여 초기 결과를 확보
- 그러나 **의료 영상 전문가 피드백**에 따르면,
  - 망막 영상은 binary segmentation이 연구 목적에 더 적합하고
  - Matting 기법은 의료적 활용성 측면에서 한계가 있다는 점이 지적됨
- 이에 따라 본 프로젝트는 **중간 단계에서 종료**하였음

---

## ✨ Reflection
- **의의**
  - Retinal 영상 분석의 어려움과 기존 segmentation 연구의 의의를 다시 확인
- **한계 / 교훈**
  - 모든 최신 AI 기법이 의료 영상에 그대로 맞아떨어지는 것은 아님
  - 임상적 타당성과 현장 전문가 피드백을 함께 고려하는 것이 필수적임
- 본 프로젝트는 종료되었으나, 과정에서 얻은 교훈을 바탕으로 이후 새로운 연구 주제를 설정하여 진행 중

