# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 과제2 수행 가이드 (v4) — Claude Code가 본 프로젝트를 일관되게 수행하도록 돕는 작업 지침서.

---

## Current repo state

본 저장소는 아직 **scaffold 전 상태**이다. §4의 디렉터리 트리는 *목표 구조*이며, 현재 실재하는 파일은 아래 셋뿐이다:

- `dataset.csv` — 원본 데이터 (저장소 루트, **`data/raw/dataset.csv`가 아님**)
- `과제2.md` — 과제 원문 (한글 파일명)
- `CLAUDE.md` — 본 문서

`src/`, `scripts/`, `outputs/`, `config/`, `data/` 등은 모두 **앞으로 만들어야 하는** 디렉터리다. 작업 시작 시 §4 트리대로 scaffold 생성을 첫 단계로 수행할 것.

### 한글 파일명 주의 (`과제2.md`)
일부 셸 환경에서 `cd … && cat 과제2.md`가 escape code로 깨질 수 있다. 그 경우:
- Read 툴로 **절대경로** 사용, 또는
- `python -c "print(open('/home/.../과제2.md', encoding='utf-8').read())"`

---

## Common commands (scaffold 후 사용)

```bash
bash scripts/run_eda.sh                                  # EDA 노트북 실행
bash scripts/run_train.sh --config config/train.yaml     # 학습
bash scripts/run_hpo_turbo.sh                            # HPO (TuRBO, 메인)
bash scripts/run_hpo_optuna.sh                           # HPO (Optuna, 비교)
bash scripts/run_hpo_compare.sh                          # 비교 리포트 생성
bash scripts/run_predict.sh                              # 예측

# 단일 가설 노트북 재실행 (한 셀씩 디버깅 후 재현 검증)
jupyter nbconvert --execute --to notebook --inplace outputs/analytics/notebook.ipynb
```

Docker:
```bash
docker build -t kakaobank2 .
docker run --rm -v "$(pwd)/outputs:/app/outputs" kakaobank2 bash scripts/run_train.sh
```

---

## 0. 과제 원문 요지 (반드시 충족해야 할 평가 기준)

> **시계열 데이터 분석 및 모델링** — 주어진 시계열 데이터를 분석하여 인사이트를 도출하고 모델링을 수행한다.

**평가 기준 (과제 설명서에서 명시)**:
1. ✅ **모든 주장 및 해석은 객관적 데이터에 근거**해야 함
2. ✅ **가설 수립 → 검증 → 모델 선정 → 결과 도출**의 논리적 과정을 명확히 서술
3. ✅ EDA 수행
4. ✅ 타겟 예측 모델 생성 + 성능 확인
5. ✅ 알고리즘·평가 지표는 **과제 목적에 부합하도록 자율 선정** (선정 근거 필수)

→ 이는 곧 **"왜 이 결정을 내렸는가"의 근거 체인**이 산출물 전체를 관통해야 함을 의미한다. 수치 없는 주장, 검증 없는 가설, 근거 없는 모델 선택은 감점 요인이다.

---

## 1. 데이터 개요

- **파일**: `dataset.csv` (1,551행 / 2018-01-01 ~ 2022-03-31, 일별)
  - `date`: 일자 (yyyy-mm-dd)
  - `holiday`: 비영업일 1, 영업일 0
  - `event`: 매월 반복 이벤트 1, 그 외 0
  - `value`: **타겟 변수** (결측 11건)
- **사전 관찰**: holiday=1일 때 value가 평일의 약 1/3, value 분포 우편향, 최댓값 13,740 (평균 3,080)
- **경로 규약**: 전처리 파이프라인은 `data/raw/dataset.csv`에서 읽는다. scaffold 첫 단계로 루트의 `dataset.csv`를 `data/raw/`로 이동/복사할 것 — **코드의 read 경로를 루트로 바꾸지 말 것**.

---

## 2. 가설 기반 분석 프레임워크 ★

EDA와 모델링을 **막연한 탐색이 아닌 가설 검증 활동**으로 진행한다. 노트북·코드·README에 다음 4단계를 일관되게 표기한다.

### 2-1. 작업할 가설 목록 (사전 등록)

| ID | 가설 | 검증 방법 | 통과 기준 |
|---|---|---|---|
| H1 | holiday=1은 value를 유의하게 낮춘다 | 그룹별 평균 + Welch's t-test | p < 0.05 |
| H2 | event=1은 value를 유의하게 높인다 | 그룹별 평균 + Welch's t-test | p < 0.05 |
| H3 | 요일 효과가 존재한다 (월~일 차이) | one-way ANOVA + 사후검정 | p < 0.05 |
| H4 | 연도별 추세(상승/하락)가 존재한다 | 연도별 평균 + Mann-Kendall | p < 0.05 |
| H5 | 강한 주간 계절성 (lag=7)이 존재한다 | ACF/PACF, lag-7 자기상관 | \|ρ\| > 0.3 |
| H6 | value 분포는 비대칭/우편향이다 | skewness, Shapiro-Wilk | skew > 1 |
| H7 | 이상치가 특정 일자에 집중되어 있다 | 3σ/IQR 룰 + 일자 패턴 | 패턴 식별 |
| H8 | holiday × dayofweek 상호작용 효과 존재 | 2-way ANOVA 상호작용항 | p < 0.05 |
| H9 | 결측 11건은 완전 무작위(MCAR)가 아니다 | Little's MCAR test 또는 패턴 분석 | 비랜덤 시 보고 |

각 가설은 EDA 노트북에서 **"가설 → 코드 → 수치 → 결론(채택/기각)"** 순서로 셀을 구성한다.

### 2-2. 모델링 가설

| ID | 가설 | 검증 방법 |
|---|---|---|
| M1 | 트리 부스팅 > 선형 모델 (비선형성 존재 시) | RMSE 비교 |
| M2 | lag/rolling 피처 추가가 성능을 개선한다 | ablation 실험 |
| M3 | HPO가 default 대비 유의한 개선을 만든다 | HPO 전/후 비교 |
| M4 | TuRBO와 Optuna는 본 9차원 문제에서 유사 성능 (TuRBO 강점은 고차원) | 두 옵티마이저 결과 비교 |

---

## 3. 평가 지표 선정과 근거 ★

과제는 **"평가 지표를 자율 선정"** 하라고 명시. 따라서 **선정 근거를 명문화** 해야 함.

### 3-1. 채택 지표

| 지표 | 채택 사유 | 비고 |
|---|---|---|
| **RMSE** | 큰 오차에 페널티, 메인 최적화 지표로 사용 | HPO 목적함수 |
| **MAE** | 이상치 영향 작음, 실무 해석 용이 | 보고용 |
| **MAPE** | 상대 오차, 비즈니스 친화적 | value > 0이라 사용 가능 |
| **sMAPE** | MAPE의 비대칭 문제 보정 | 보고용 |
| **R²** | 분산 설명력 | 보고용 |

### 3-2. 메인 지표 = RMSE 선정 근거 (README에 기록)

1. value 분포가 우편향·이상치 존재 → 큰 오차에 민감해야 비즈니스 리스크 감지
2. M5 등 시계열 대회 표준 지표
3. HPO surrogate 학습에 미분 가능한 형태가 유리

### 3-3. 베이스라인 대비 평가

단일 지표 값보다 **베이스라인 대비 개선율**이 중요. 모든 모델은 다음과 함께 보고:
- vs Naive seasonal (`y(t-7)`)
- vs 그룹 평균 baseline

---

## 4. 디렉터리 구조

```
.
├── CLAUDE.md
├── README.md
├── Dockerfile
├── requirements.txt
├── run.py
├── config/
│   ├── train.yaml
│   ├── hpo_turbo.yaml
│   └── hpo_optuna.yaml
├── data/
│   ├── raw/dataset.csv
│   └── data_preprocessed.csv
├── scripts/
│   ├── run_eda.sh
│   ├── run_train.sh
│   ├── run_predict.sh
│   ├── run_hpo_turbo.sh
│   ├── run_hpo_optuna.sh
│   └── run_hpo_compare.sh
├── src/
│   ├── data/preprocess.py
│   ├── features/build.py
│   ├── train/run_train.py
│   ├── predict/run_predict.py
│   ├── tuning/
│   │   ├── __init__.py
│   │   ├── param_spec.py
│   │   ├── cv_objective.py
│   │   ├── turbo_runner.py
│   │   ├── optuna_runner.py
│   │   └── compare.py
│   └── evaluation/metrics.py
└── outputs/
    ├── analytics/
    │   ├── notebook.ipynb       # 가설 검증 중심 EDA
    │   └── hypothesis_log.md   # ★ 가설 채택/기각 결과 요약
    ├── models/
    ├── hpo/
    │   ├── turbo/
    │   ├── optuna/
    │   └── comparison/
    └── predictions/
```

> **TuRBO 설치 방식**: pip-install only (`pip install git+https://github.com/uber-research/TuRBO.git`). git submodule + pip install 동시 사용은 import shadowing을 유발하므로 **submodule(`third_party/turbo/`) 사용 금지**.

---

## 5. 작업 순서 (논리적 흐름 강제)

### STEP 1 — EDA: 가설 검증 형식 (`outputs/analytics/notebook.ipynb`)

**노트북 셀 구조 강제**:
```
[Section 1] 데이터 로딩 & 기본 통계
[Section 2] 가설 H1~H9 검증
  - 각 가설마다: 가설 진술 → 시각화 → 통계 검정 → 결론
[Section 3] STL 분해 (trend/seasonality/residual)
[Section 4] EDA 종합 인사이트 (Top 5)
[Section 5] EDA 결과가 모델링 결정에 미친 영향 (decision log)
```

**최소 포함 분석**:
1. 시계열 plot (전체 + 연/월별)
2. holiday/event 그룹별 value 분포 (boxplot, 평균/중앙값 표, **t-test 결과**)
3. 요일/월/연도 효과 (boxplot + ANOVA)
4. 결측 11건 위치/패턴 분석
5. ACF/PACF (lag 7, 30 주목)
6. 이상치 일자 식별 (3σ, IQR — 일자 리스트 출력 필수)
7. STL 분해
8. **가설 채택/기각 요약표**를 마지막 셀에

→ 결과는 `hypothesis_log.md`에도 별도 저장 (의사결정 추적용).

### STEP 2 — 전처리 (`src/data/preprocess.py`)

**EDA 결과로 결정되는 것들**:
- 결측 처리 방식 — H9(MCAR 여부)에 따라 선형보간 vs 요일평균 보정 결정
- 이상치 처리 — H7에서 식별된 일자가 특수일이면 보존, 아니면 winsorize 또는 로그변환
- log 변환 여부 — H6 결과에 따라 결정

**모든 결정에 코드 주석으로 EDA 근거 명시**:
```python
# H6 채택(skew=2.3, p<0.001) → log1p 변환 적용
y_train = np.log1p(y_train_raw)
```

### STEP 3 — 피처 엔지니어링 (`src/features/build.py`)

**모든 피처는 가설 또는 도메인 지식 기반 (목적 명시)**:
- 캘린더 피처: `dayofweek`, `month`, `day`, `weekofyear`, `is_weekend`, `is_month_start/end`
  - 근거: H3 (요일 효과)
- 주기 인코딩 (sin/cos): `dayofweek`, `month`
  - 근거: 트리 모델은 불필요하지만 선형 모델용
- Lag: `value_lag_{1,7,14,28}` (반드시 `shift(1)` 후)
  - 근거: H5 (lag-7 자기상관)
- Rolling: `mean_{7,28}`, `std_7`
  - 근거: 단기 변동성 포착
- 상호작용: `holiday × dayofweek`
  - 근거: H8

**피처별 ablation 실험 결과**를 `outputs/analytics/feature_importance.md`에 기록.

### STEP 4 — 모델 후보 선정 + 근거

**시간순 split**: train = 2018-01 ~ 2021-09 (45개월), val = 2021-10 ~ 2021-12 (3개월), test = 2022-01 ~ 2022-03 (3개월)

> **고정 split 원칙**: 위 윈도우는 사전 등록(pre-registered)된 것으로 **결과를 본 뒤 재조정하지 말 것**. test 성능을 보고 split을 옮기면 곧 test-set leakage이며 §9-1 위반. 재조정이 필요하다고 판단되면 별도 결정 로그(`outputs/analytics/decision_log.md`)에 사유와 함께 기록한 뒤 모든 모델을 새 split으로 재실행.

| 모델 | 역할 | 선정 근거 |
|---|---|---|
| Naive seasonal `y(t-7)` | 베이스라인 | 강한 주간 계절성(H5) 활용한 단순 모델 |
| 그룹 평균 (요일×holiday) | 베이스라인 | 가장 단순한 휴리스틱 — 어떤 모델이든 이걸 못 이기면 안 됨 |
| Ridge | 선형 베이스라인 | 피처 효과 해석, 비선형성 필요성 검증(M1) |
| **LightGBM** | **메인** | 시계열 회귀 표준 (M5 우승 레시피), 비선형/상호작용 자동 포착 |
| XGBoost | 보조 비교 | LightGBM과 다른 트리 분할 전략 — 앙상블 후보 |
| Prophet | 해석용 베이스라인 | 휴일·계절성 분해 결과를 정성 검증에 활용 |

검증: **TimeSeriesSplit(n_splits=5)**

### STEP 5 — HPO (TuRBO 메인 + Optuna 비교) ★

**(이전 v3에서 정의한 내용 그대로 — 핵심만 요약)**

#### 5-1. 설계 원칙
- TuRBO를 메인 (NeurIPS 2019 알고리즘 — 차별 포인트)
- Optuna를 비교군 (산업 표준)
- **두 옵티마이저는 동일한 `cv_objective` 호출** (공정 비교)

#### 5-2. 공통 파라미터 명세 (`src/tuning/param_spec.py`)
모든 파라미터를 `[0,1]^d`로 정규화한 명세 dict 정의. TuRBO는 어댑터로 디코딩, Optuna는 동일 명세로 `suggest_*` 호출.

#### 5-3. 공통 목적함수 (`src/tuning/cv_objective.py`)
TimeSeriesSplit CV로 평균 RMSE 반환. seed 고정으로 결정론적.

#### 5-4. TuRBO Runner
```python
from turbo import Turbo1
turbo = Turbo1(f=objective, lb=np.zeros(d), ub=np.ones(d),
               n_init=20, max_evals=100, batch_size=5,
               use_ard=True, device="cpu", dtype="float64")
turbo.optimize()
```
**주의사항**:
- noise-free 가정 위반 보정: CV seed 고정
- 정수 plateau: `n_init ≥ 2d`
- 영속화 부재: 매 trial csv flush
- 라이선스: Uber Non-Commercial (README 명시)

#### 5-5. Optuna Runner
TPESampler + MedianPruner + sqlite 영속화.

#### 5-6. 비교 리포트 (`outputs/hpo/comparison/report.md`) ★
1. 수렴 곡선 (한 그래프)
2. best 파라미터 표
3. walltime 비교
4. test set 최종 성능
5. 정성 분석 (M4 가설 검증)

#### 5-7. 동일 예산 (100 evals) — 공정 비교 보장

### STEP 6 — 평가 (`src/evaluation/metrics.py`)

**필수 산출물**:

#### 6-1. 통합 성능 표
| Model | HPO | MAE | RMSE | MAPE | sMAPE | R² | vs Naive 개선율 |
|---|---|---|---|---|---|---|---|
| Naive seasonal | — | ... | ... | ... | ... | ... | 0% (기준) |
| 그룹 평균 | — | ... | ... | ... | ... | ... | ...% |
| Ridge | default | ... | ... | ... | ... | ... | ...% |
| LightGBM | default | ... | ... | ... | ... | ... | ...% |
| LightGBM | Optuna | ... | ... | ... | ... | ... | ...% |
| **LightGBM** | **TuRBO** | ... | ... | ... | ... | ... | ...% |
| XGBoost | TuRBO | ... | ... | ... | ... | ... | ...% |
| Prophet | — | ... | ... | ... | ... | ... | ...% |

#### 6-2. 잔차 분석
- 잔차 시계열 plot (자기상관 잔존 여부)
- 잔차 vs 예측값 (이분산성)
- 잔차 분포 (정규성, Q-Q plot)
- **가장 큰 오차 Top 10 일자** — 모델이 못 잡는 패턴 식별

#### 6-3. 피처 중요도 + SHAP
- LightGBM gain importance
- SHAP summary plot (글로벌 해석)
- SHAP force plot (개별 예측 해석 — 1~2개 케이스)

#### 6-4. 예측 구간 (선택)
- 분위수 회귀(LightGBM quantile) 또는 부트스트랩으로 80% 예측 구간 추정
- 비즈니스 의사결정에 신뢰도 정보 제공

### STEP 7 — 재현성 패키징

- 모든 하이퍼파라미터 외부화 (`config/*.yaml`)
- `random_state`, Optuna seed, TuRBO `np.random.seed`+`torch.manual_seed` 모두 42 고정
- `requirements.txt`: pandas, numpy, scikit-learn, lightgbm, xgboost, matplotlib, seaborn, statsmodels, scipy, optuna≥3.5, torch≥1.9, gpytorch≥1.6, pyyaml, shap, prophet (선택)
- TuRBO: `pip install git+https://github.com/uber-research/TuRBO.git`
- 라이선스 명시: TuRBO Uber Non-Commercial
- Dockerfile: python:3.11-slim
- scripts/*.sh로 한 줄 실행

---

## 6. 코딩·문서화 규칙

- type hint + docstring 필수
- 출력은 항상 `outputs/` 하위
- print 대신 `logging`
- lag/rolling은 **반드시 shift(1) 후** 계산
- **모든 의사결정에 EDA·실험 근거를 코드 주석/문서로 남길 것** (과제 평가 핵심)
- 한국어 주석 OK, 변수명은 영어

---

## 7. README.md 필수 항목

1. **프로젝트 개요** + 디렉터리 설명
2. **EDA 핵심 인사이트 5가지** (가설 채택 결과 요약)
3. **모델 선정 근거** — 왜 LightGBM, 왜 TuRBO, 왜 RMSE
4. Docker 빌드/실행 명령
5. TuRBO 설치 방법 + 라이선스 주의사항
6. 실행 방법: `run_eda.sh`, `run_train.sh`, `run_hpo_*.sh`, `run_predict.sh`
7. **결과 요약 표** (8개 모델 × 5개 지표)
8. **TuRBO vs Optuna 비교 결론** — 9차원 문제에서 정직한 결론
9. **한계와 향후 개선** — 정직 평가 포인트
10. **재현성 체크리스트** — seed, version, command

---

## 8. 노트북 작성 원칙 ★

`outputs/analytics/notebook.ipynb`는 **이야기처럼 읽혀야 함**:

```
서론: "이 데이터는 무엇인가, 우리는 무엇을 알고 싶은가"
  ↓
가설: "다음 9개 가설을 검증한다"
  ↓
검증: "H1: holiday는 value를 낮춘다 → 코드 → p=0.000 → 채택"
  ↓
종합: "EDA에서 얻은 5가지 핵심 인사이트"
  ↓
결정: "이 인사이트는 모델링에 다음과 같이 반영된다"
```

**반-패턴**:
- ❌ 그래프 10개 나열 후 결론 없음
- ❌ "흥미롭다" "특이하다" 같은 주관적 표현
- ❌ p-value 보여주지 않고 "차이가 있다"고 단정
- ❌ 코드만 있고 해석 없음

---

## 9. Claude가 피해야 할 실수

### 9-1. 데이터 누수
- ❌ `train_test_split(shuffle=True)`
- ❌ 미래 데이터로 lag/rolling 계산
- ❌ HPO를 전체 데이터로 돌리고 test로 또 평가
- ❌ TimeSeriesSplit 외 random fold

### 9-2. 평가 기준 위반 (과제 핵심)
- ❌ "value가 holiday에 따라 다르다" → **수치/p-value 없으면 무효**
- ❌ "이 모델이 더 좋다" → **개선율·통계적 유의성 없으면 무효**
- ❌ "결측을 보간으로 처리했다" → **왜 그 방법인지 EDA 근거 없으면 감점**
- ❌ 모델 비교 없이 단일 모델 제출

### 9-3. HPO 함정
- ❌ TuRBO와 Optuna에 다른 cv_objective
- ❌ TuRBO trial 결과를 csv 저장 안 함
- ❌ Optuna study DB 저장 누락
- ❌ TuRBO `n_init` 너무 작음 (< 2d)
- ❌ "TuRBO가 무조건 좋다"고 결론 — 9차원에서는 우위 없을 가능성

### 9-4. 라이선스/설치
- ❌ TuRBO를 `pip install turbo`로 시도 (PyPI 미배포)
- ❌ TuRBO Uber Non-Commercial 명시 누락
- ❌ Docker `latest` 태그

### 9-5. 결측·이상치
- ❌ value 결측을 0으로 채움
- ❌ 이상치 무조건 제거 (특수일 가능성 — 일자 먼저 확인)

---

## 10. 면접/심사 대비 — 예상 질문

| 질문 | 핵심 답변 포인트 |
|---|---|
| 가장 영향력 큰 인사이트는? | EDA 가설 9개 중 채택된 것 + 모델링에 반영된 방식 |
| 왜 RMSE를 메인 지표로? | 우편향 분포·이상치 존재 → 큰 오차에 민감해야 함 |
| 왜 LightGBM? | M5 표준, 비선형/상호작용 자동, 본 데이터 규모에 적합 |
| 왜 Optuna가 아닌 TuRBO? | 차별성 + 연구 알고리즘 이식 경험. 9차원에선 우위 없음을 정직 보고 |
| 결측 처리 근거? | H9 검증 결과 + 채택한 방법의 robustness |
| 이상치 처리 근거? | H7에서 식별된 일자 확인 → 보존/제거 결정 근거 |
| 한계는? | 단일 시계열, 외생변수 부족, 코로나 영향 분리 어려움 등 |

---

## 11. 최종 체크리스트 (제출 전)

> 평가 기준 자체는 §0 참조. 본 절은 **산출물 완전성 점검** 목록.

- [ ] 노트북 9개 가설 모두 검증 셀 완료 (§5 STEP 1)
- [ ] 모든 의사결정에 데이터 근거 명시 — 코드 주석 또는 README (§0 평가기준 1)
- [ ] 8개 모델 × 5개 지표 통합 표 완성 (§6-1)
- [ ] HPO 전/후 + TuRBO vs Optuna 비교 표 완성 (§5-6)
- [ ] 잔차 분석 + SHAP 분석 완료 (§6-2, §6-3)
- [ ] Docker 빌드·실행 검증 (clean 환경에서)
- [ ] seed 고정으로 2회 실행 시 동일 결과 확인 (§5 STEP 7)
- [ ] README에 EDA 인사이트 5가지 + 모델 선정 근거 + 한계 기록 (§7)
- [ ] TuRBO 라이선스 명시 (§5-4, §5 STEP 7)
- [ ] `bash scripts/run_*.sh` 한 줄 실행 모두 동작 (Common commands 섹션 참조)
