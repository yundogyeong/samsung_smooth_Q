# Quantization Practice

**Affiliation:** 서울대학교 머신러닝 시스템 연구실  

## 실습 단계

### Practice 1: Collect Activation Maximums
- 각 **Hidden Dimension**에 대해 Activation의 최대값 수집
- Sequence Length 차원을 펼친 뒤, `per-Hidden Dim` 최대값 계산
- `act_scale = {}` 딕셔너리에 저장

---

### Practice 2: Compute Smoothing Factors
- **Weight Scale**: 각 Input Dimension별 최대값 계산
- **Smoothing Factor 계산**:
  - Activation과 Weight의 스케일을 조합하여 최적화
  - 예시: Activation에 `×0.1`, Weight에 `×10` 적용

---

### Practice 3: Quantize Real Value to INT8
- **사전 준비 단계 (Done in advance)**:
  - 수집한 스케일을 기반으로 Weight 사전 변환
- **런타임 단계 (Done in runtime)**:
  - 입력 Activation을 스케일링 후 int8로 변환
  - Quantized Weight와 연산 수행

---