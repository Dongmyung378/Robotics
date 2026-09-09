<div align="right">
  <a href="./README.md">English</a> | <strong>한국어</strong>
</div>

# 지식 증류 기반 교통 표지판 인식 모델 경량화

MobileNetV2 Teacher 모델의 지식을 3.4만 개 파라미터의 경량 모델로 전달해 German Traffic Sign Recognition Benchmark(GTSRB)를 분류하는 TensorFlow 컴퓨터 비전 프로젝트입니다.

[노트북 열기](./notebooks/knowledge_distillation_gtsrb.ipynb) · [기술 리포트 읽기](./docs/knowledge-distillation-report.pdf) · [Kaggle에서 보기](https://www.kaggle.com/code/dongmyungpark/gtsrb)

## 프로젝트 요약

이 프로젝트는 추론 복잡도를 늘리지 않으면서 작은 합성곱 신경망이 강력한 사전학습 모델의 정확도를 얼마나 회복할 수 있는지 검증합니다.

MobileNetV2 Teacher 모델, 독립적으로 학습한 경량 기준 모델, 9개의 지식 증류 모델을 비교합니다. 세 가지 temperature와 세 가지 hard-label weight 조합을 탐색하고 정확도, 파라미터 수, 단일 이미지 지연 시간, 클래스별 오류 패턴을 분석했습니다.

| 지표 | 결과 |
| --- | ---: |
| 최고 지식 증류 정확도 | **80.97%** |
| 경량 기준 모델 정확도 | 54.61% |
| Teacher 모델 정확도 | 88.37% |
| 경량 기준 모델 대비 향상 | **+26.37%p** |
| Teacher 정확도 유지율 | **91.6%** |
| Teacher 대비 파라미터 감소 | **67.2배** |
| 최고 조합 | `temperature=10`, `alpha=0.3` |

지식 증류 모델은 경량 기준 모델과 동일한 34,406개 파라미터 구조와 유사한 측정 지연 시간을 유지합니다. 따라서 성능 향상은 배포 모델의 확장이 아니라 학습 신호에서 발생합니다.

![Temperature와 alpha 탐색에 따른 정확도 비교](./assets/figures/accuracy-comparison.png)

## 실험 설계

### 데이터

- 43개 교통 표지판 클래스로 구성된 GTSRB
- 학습 이미지 39,209장, 평가 이미지 12,630장
- 이미지를 96 × 96으로 조정하고 `[0, 1]` 범위로 정규화
- 클래스 불균형을 보정하기 위한 역빈도 가중치 적용
- 반복 가능한 결과를 위한 고정 난수 시드 사용

### 모델

- **Teacher:** ImageNet 사전학습 MobileNetV2의 마지막 60개 backbone layer를 미세 조정하고 global average pooling, dropout, 43-class output layer를 결합
- **경량 모델:** 세 개의 depthwise-separable convolution block, global average pooling, 작은 dense classifier로 구성
- **모델 크기:** Teacher 2,313,067개, 경량 모델 34,406개 파라미터

### 지식 증류 목적 함수

경량 모델은 정답 레이블과 Teacher의 temperature-scaled 확률 분포를 함께 학습합니다.

```text
loss = alpha * hard_label_loss
     + (1 - alpha) * temperature^2 * distillation_loss
```

노트북은 다음 값의 모든 조합을 평가합니다.

- Temperature: `3`, `5`, `10`
- Hard-label weight (`alpha`): `0.1`, `0.3`, `0.5`

![전체 지식 증류 조합의 검증 정확도](./assets/figures/hyperparameter-grid.png)

## 핵심 인사이트

- 9개 지식 증류 조합 모두 경량 기준 모델보다 최소 21.82%p 높은 정확도를 기록했습니다.
- Temperature와 `alpha`는 상호작용합니다. 두 값을 함께 탐색했을 때 temperature만 분리해 탐색한 최고 결과보다 1.59%p 높은 조합을 찾았습니다.
- 속도 제한 표지판 클래스의 평균 향상 폭은 나머지 클래스보다 컸습니다(+33.11%p 대 +23.44%p). 시각적 유사성이 의미 있는 클래스 구조를 반영할 때 soft target이 특히 유용함을 보여줍니다.
- 일부 방향 표지판 클래스에서는 성능이 낮아졌습니다. 형태가 비슷해도 의미상 명확히 구분해야 하는 클래스에서는 negative transfer가 발생할 수 있습니다.

## 저장소 구조

```text
.
├── assets/
│   └── figures/                  # 주요 결과 시각화
├── docs/
│   ├── knowledge-distillation-report.pdf
│   └── project-brief.pdf
├── notebooks/
│   └── knowledge_distillation_gtsrb.ipynb
├── README.md                     # 영문 문서
├── README.ko.md                  # 국문 문서
└── requirements.txt
```

기술 리포트에는 전체 방법론, 관련 연구 맥락, 결과 및 논의가 담겨 있습니다. 원본 프로젝트 브리프는 작업 출처를 보존하기 위해서만 유지합니다.

## 실행 방법

저장된 실행 결과는 Python 3.12, TensorFlow 2.19, Tesla P100 GPU 환경에서 생성되었습니다. 노트북이 `kagglehub`로 GTSRB를 내려받고 저장된 출력도 GPU 기반 Kaggle 환경에서 만들어졌으므로 Kaggle에서 실행하는 방법이 가장 간단합니다.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
jupyter lab notebooks/knowledge_distillation_gtsrb.ipynb
```

로컬 실행 시 GTSRB와 MobileNetV2 사전학습 가중치를 내려받을 인터넷 연결이 필요합니다. CUDA 지원 GPU 사용을 권장하며, 기록된 전체 실행 시간은 약 56분입니다.

## 평가 시 참고 사항

- 요약 정확도는 저장된 학습 이력에서 각 모델이 기록한 최고 검증 정확도입니다.
- 노트북은 공식 GTSRB test split을 모델 선택 과정의 `val_ds`로 사용합니다. 따라서 결과는 별도로 보존한 최종 test set의 일반화 추정치가 아니라 통제된 실험 비교로 해석해야 합니다.
- 후반부 confusion matrix 셀에서 경량 기준 모델의 마지막 메모리 가중치는 54.12%를 기록하지만, 학습 이력의 최고 epoch는 54.61%입니다. 요약표에는 실험 표 전체에서 일관되게 사용한 최고 epoch 값을 표시했습니다.
- 지연 시간은 저장된 Kaggle 환경에서 batch size 1로 측정했으며 프레임워크의 prediction overhead를 포함합니다. 하드웨어와 무관한 벤치마크 값은 아닙니다.
