# 과제2 — 시계열 데이터 분석 및 모델링

> 일별 시계열(`value`)의 EDA + 예측 모델 개발.  
> 개발 가이드라인은 `CLAUDE.md`를 참조 — 본 README는 **사용자 관점**의 실행/결과 문서.

---

## 1. 프로젝트 개요

- **데이터**: 2018-01-01 ~ 2022-03-31 (1,551일 / 결측 11건). `date, holiday, event, value`.
- **목표**: 가설 기반 EDA → 베이스라인·트리·HPO를 거쳐 일별 `value` 예측 모델 도출.
- **차별 포인트**: HPO에서 **TuRBO** (NeurIPS 2019, Uber-Research) 메인 채택, Optuna는 비교군. M4 가설("9차원에서 두 옵티마이저 유사")을 데이터로 검증.

```
.
├── CLAUDE.md                 # 개발 가이드 (작업자용)
├── README.md                 # 본 문서 (사용자용)
├── Dockerfile / .dockerignore
├── requirements.txt
├── run.py                    # 전체 파이프라인 진입점
├── config/                   # 모든 하이퍼파라미터 (재현성)
│   ├── train.yaml
│   ├── hpo_turbo.yaml
│   └── hpo_optuna.yaml
├── data/raw/dataset.csv      # 정규 입력 경로 (CLAUDE.md §1)
├── scripts/                  # 한-줄 실행 wrapper
├── src/{data,features,evaluation,train,predict,tuning}/
└── outputs/{analytics,models,hpo,predictions}/
```

---

## 2. EDA 핵심 인사이트 5가지

`outputs/analytics/notebook.ipynb` 실행 결과 (가설 9개 검증, 자세한 수치는 노트북·`hypothesis_log.md`).

1. **H1 채택 ✅** — `holiday=1`은 평균 value를 평일 대비 **약 47%** 수준으로 낮춤 (Welch's t, p≈0). CLAUDE.md 사전 관찰 "1/3"은 약간 과대 추정이었다.
2. **H4 채택 ✅** — 월별 평균이 **강한 상승 추세** (Mann-Kendall τ=+0.88, p≈5.5e-20). 시간 인덱스 피처(`t_index`, `year`) 필수.
3. **H5 채택 ✅** — `autocorr(lag=7)=+0.74`. 강한 주간 계절성. → `value_lag_{1,7,14,28}`, `rolling_mean_{7,28}` 피처로 반영.
4. **H6 부분 기각 ⚠️** — skew=+0.69. CLAUDE.md 사전 통과 기준(>1.0) 미달. Shapiro-Wilk p≈0이지만 분포가 강하게 우편향이라 보긴 어렵다 → log1p는 *기본 OFF*, 옵션 토글로만.
6. **데이터 무결성 발견** — `event` 컬럼 스펙은 `{0,1}`이지만 실제로 `2`가 1건. preprocess에서 `event_flag = (event > 0)`으로 정규화.

> 가설 채택/기각 전체 표는 `outputs/analytics/hypothesis_log.md` 참조.

---

## 3. 모델 선정 근거

| 결정 | 근거 |
|---|---|
| **메인 모델 = LightGBM** | M5 시계열 우승 표준. 본 데이터의 비선형성·상호작용(H8) 자동 포착. 1,500행 규모에 적합. |
| **메인 지표 = RMSE** | value 분포가 우편향+이상치 존재(H6, H7) → 큰 오차에 민감해야 비즈니스 리스크 감지 가능. HPO surrogate(GP) 학습에도 미분가능 형태 유리. |
| **메인 HPO = TuRBO** | 차별 포인트 + 연구 알고리즘 이식 경험. 9차원에서 우위가 없을 가능성을 *정직*하게 비교 (M4 가설). |
| **베이스라인 2종 필수** | Naive seasonal(`y(t-7)`), 그룹 평균. **이 둘을 못 이기면 모델 가치 없음** (CLAUDE.md §9-2). |
| **시간순 split (고정)** | train=2018-01~2021-09, val=2021-10~2021-12, test=2022-01~2022-03. **결과 보고 재조정 금지** (§9-1). |

---

## 4. 실행 방법

### 4-1. Docker (권장)

```bash
# 빌드 (~5분, TuRBO git 설치 포함)
docker build -t kakaobank2 .

# EDA 노트북만 실행
docker run --rm -v "$(pwd)/outputs:/app/outputs" kakaobank2 bash scripts/run_eda.sh

# 학습 + 예측 (HPO 제외 빠른 경로)
docker run --rm -v "$(pwd)/outputs:/app/outputs" -v "$(pwd)/data:/app/data" kakaobank2 \
    python run.py all_no_hpo

# 전체 (EDA → preprocess → train → predict → HPO Optuna → HPO TuRBO → compare)
docker run --rm -v "$(pwd)/outputs:/app/outputs" -v "$(pwd)/data:/app/data" kakaobank2 \
    python run.py all
```

### 4-2. 로컬 (Python 3.11 권장)

```bash
pip install -r requirements.txt
pip install "git+https://github.com/uber-research/TuRBO.git@master"

bash scripts/run_eda.sh             # EDA notebook 실행 (in-place)
bash scripts/run_train.sh           # preprocess → train (5개 모델)
bash scripts/run_predict.sh         # 잔차 + SHAP 분석
bash scripts/run_hpo_optuna.sh      # Optuna 100 trials
bash scripts/run_hpo_turbo.sh       # TuRBO 100 evals
bash scripts/run_hpo_compare.sh     # 비교 리포트 생성
```

또는 단일 명령:

```bash
python run.py all_no_hpo  # EDA + 학습 + 예측만
python run.py all         # HPO 포함 전체
python run.py train       # 단일 단계만
```

---

## 5. 결과 요약 (test set, 2022-01 ~ 2022-03)

CV 평균이 아닌 **실제 test 윈도우** 성능. 마지막 컬럼은 `vs Naive seasonal` 개선율.

| Model | HPO | RMSE | MAE | MAPE | sMAPE | R² | vs Naive |
|---|---|---:|---:|---:|---:|---:|---:|
| Naive seasonal `y(t-7)` | — | 803.5 | 431.0 | 12.9% | 11.9% | 0.445 | 0.0% |
| Group mean (dow×holiday) | — | 821.0 | 782.5 | 21.5% | 24.3% | 0.421 | -2.2% |
| Ridge                   | default | 267.2 | 227.9 |  7.7% |  7.2% | 0.939 | 66.7% |
| LightGBM                | default | 223.3 | 179.8 |  5.1% |  5.1% | 0.957 | 72.2% |
| XGBoost                 | default | 294.2 | 253.5 |  8.0% |  7.7% | 0.926 | 63.4% |
| LightGBM | Optuna 100t | 246.3 | 182.2 |  4.6% |  4.7% | 0.948 | 69.3% |
| **LightGBM**            | **TuRBO 100t** | **204.3** | **146.7** |  **3.8%** |  **3.8%** | **0.964** | **74.6%** |

> **최종 선정 모델**: LightGBM + TuRBO HPO. test RMSE 204.3 (default 대비 **8.5% 추가 개선**), R²=0.964.  
> Group mean이 Naive에 *지는* 점이 흥미: 강한 추세(H4) 때문에 작년 평균보다 7일 전 값이 낫다.

전체 표는 `outputs/predictions/metrics.csv`. HPO 결과는 `outputs/hpo/comparison/report.md`.

---

## 6. TuRBO vs Optuna 비교

전체 100 trial 실행 결과 (`outputs/hpo/comparison/report.md`):

| Optimizer | best CV-RMSE | walltime | test RMSE | test R² |
|---|---:|---:|---:|---:|
| TuRBO 100 evals  | **456.81** | 33s  | **204.25** | **0.964** |
| Optuna 100 trials | 471.74 | 254s | 246.34 | 0.948 |

**M4 가설** ("9차원에서 TuRBO와 Optuna 유사") **부분 채택**:
- CV-RMSE 차이 3.27% (5% 미만 → 부분 채택), test set에서는 TuRBO가 17% 우세
- TuRBO의 trust-region 전략이 본 데이터(작고 잘 정렬된 9d 문제)에서도 효과적
- Optuna가 walltime이 7배 길었던 이유: TPE의 acquisition + medianpruner overhead. 본 데이터 규모에서는 GP 기반 TuRBO가 오히려 빠름

> 일반화 결론: TuRBO의 본 강점은 20-200차원 고차원 BlackBox 최적화 (NeurIPS 2019 원논문). 본 9차원 LightGBM HPO에서도 우위가 있었지만, 일반적으로는 산업 표준 Optuna(TPE)가 충분한 영역. 본 결과를 "TuRBO 절대 우세"로 일반화하지 말 것.

---

## 7. 한계와 향후 개선

- **외생 변수 부족**: holiday/event 외에 마케팅·날씨·외부 이벤트 데이터를 결합하면 잔차 자기상관(`outputs/predictions/residual_summary.csv`의 Ljung-Box) 추가 감소 여지.
- **코로나 효과**: 2020-03 이후 분포 이동 (강한 추세 H4의 일부일 수 있음). change-point 분석은 미실시.
- **단일 시계열**: 본 데이터는 단일 채널. 다채널이라면 hierarchical/global 모델(N-BEATS, TFT 등) 검토 가치.
- **예측 구간 미제공**: §6-4 분위수 회귀는 옵션 단계 — 현재 미구현.
- **Prophet 미포함**: 정성적 분해 검증용 베이스라인. 패키지 무게 대비 가치 판단으로 일단 제외.

---

## 8. 재현성 체크리스트

| 항목 | 상태 |
|---|---|
| `random_state`, Optuna seed, TuRBO `np.random.seed`+`torch.manual_seed` 모두 42 고정 | ✅ |
| 모든 하이퍼파라미터 외부화 (`config/*.yaml`) | ✅ |
| `requirements.txt` 버전 핀 | ✅ |
| Docker `python:3.11-slim` 명시 (latest 미사용) | ✅ |
| TuRBO 라이선스 명시 (Uber Non-Commercial) | ✅ (본 §9) |
| `scripts/run_*.sh` 한 줄 실행 가능 | ✅ |
| seed 고정으로 2회 실행 시 동일 결과 | ✅ (CV·split·모델 모두 결정론적) |

### 9. TuRBO 라이선스 주의

본 프로젝트는 [Uber-Research/TuRBO](https://github.com/uber-research/TuRBO)를 사용한다. 라이선스는 **Uber Non-Commercial License** — 학술/연구 용도만 허용, 상용 사용 금지. 상용 환경에 배포할 경우 TuRBO를 제거하고 Optuna 결과만 사용하거나 동등 BO 알고리즘(BoTorch, Ax 등)으로 교체해야 한다.

---

## 10. 산출물 위치

```
outputs/
├── analytics/
│   ├── notebook.ipynb            # 실행된 EDA 노트북
│   ├── hypothesis_log.md         # 가설 채택/기각 표
│   └── feature_importance.md     # LightGBM gain importance
├── models/                       # 학습된 모델 pickle
├── hpo/
│   ├── turbo/{trials.csv,best_params.json}
│   ├── optuna/{trials.csv,best_params.json,study.db}
│   └── comparison/{report.md,convergence.png}
└── predictions/
    ├── metrics.csv               # 8개 모델 통합 표
    ├── predictions_*.csv         # 모델별 train/val/test 예측
    ├── residual_plots.png        # 잔차 3-panel
    ├── shap_summary.png          # LightGBM SHAP
    └── top10_errors_lightgbm.csv # 가장 큰 오차 일자
```
