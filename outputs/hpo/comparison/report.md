# HPO Comparison — TuRBO vs Optuna

**Date**: 2026-05-10 09:10

## 1. 수렴 곡선

![](convergence.png)

## 2. best CV-RMSE

| Optimizer | n_trials | best CV-RMSE | walltime (s) |
|---|---|---|---|
| TuRBO  | 100  | 463.423  | 28.8 |
| Optuna | 200 | 471.249 | 371.8 |

## 3. Test set 최종 성능

| Optimizer | RMSE | MAE | MAPE | sMAPE | R² |
|---|---|---|---|---|---|
| TuRBO | 433.84 | 357.79 | 10.87 | 11.03 | 0.838 |
| Optuna | 378.38 | 311.63 | 9.07 | 9.51 | 0.877 |

## 4. Best params

### TuRBO
```json
{
  "learning_rate": 0.11568174427378175,
  "num_leaves": 61,
  "max_depth": 3,
  "min_child_samples": 17,
  "feature_fraction": 0.8708499327319221,
  "bagging_fraction": 0.5405709850625559,
  "bagging_freq": 2,
  "lambda_l1": 0.0026128499587609933,
  "lambda_l2": 0.00043074400794777253
}
```

### Optuna
```json
{
  "learning_rate": 0.1278257054128034,
  "num_leaves": 195,
  "max_depth": 4,
  "min_child_samples": 25,
  "feature_fraction": 0.944637007405818,
  "bagging_fraction": 0.529980591136961,
  "bagging_freq": 1,
  "lambda_l1": 0.0003839698658641854,
  "lambda_l2": 0.04323678460172011
}
```

## 5. 정성 분석 (M4 가설 검증)

**M4**: 본 9차원 문제에서 TuRBO와 Optuna는 유사 성능 (TuRBO 강점은 고차원).

- best CV-RMSE 차이: **7.826** (1.69%)
- **M4 부분 채택 ⚠️** — 차이는 있으나 5% 미만.

> 결론: 9차원 정도의 LightGBM HPO에서는 TuRBO와 Optuna가 유사 성능을 보인다는 것이 본 실험에서도 재확인됨. 
> TuRBO의 강점은 고차원(20~200d) 문제에서 두드러짐 (NeurIPS 2019 원논문).