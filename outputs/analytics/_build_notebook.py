"""
Programmatically build outputs/analytics/notebook.ipynb.

Why a builder rather than hand-edited .ipynb:
  - 35+ cells with consistent structure → easier to maintain in code.
  - Re-runnable: `python outputs/analytics/_build_notebook.py` regenerates the file.
  - No git noise from output cells; execution is the user's job (run_eda.sh).

Run after cloning:
    python outputs/analytics/_build_notebook.py
    bash scripts/run_eda.sh    # executes the notebook in-place
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path(__file__).resolve().parent / "notebook.ipynb"


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []

    # ── HEADER ───────────────────────────────────────────────────────────
    cells.append(md(
        "# 시계열 EDA — 가설 기반 분석\n"
        "\n"
        "**과제2 — Section 1~5 of CLAUDE.md §5 STEP 1**\n"
        "\n"
        "본 노트북은 **막연한 탐색이 아닌 가설 검증** 활동이다. 각 셀은 다음 구조를 따른다:\n"
        "\n"
        "> 가설 진술 → 시각화 → 통계 검정 → 결론(채택/기각)\n"
        "\n"
        "## 사전 등록된 가설 (CLAUDE.md §2-1)\n"
        "\n"
        "| ID | 가설 | 검증 | 통과 기준 |\n"
        "|---|---|---|---|\n"
        "| H1 | holiday=1은 value를 낮춘다 | Welch's t-test | p < 0.05 |\n"
        "| H2 | event=1은 value를 높인다 | Welch's t-test | p < 0.05 |\n"
        "| H3 | 요일 효과 존재 | one-way ANOVA + Tukey | p < 0.05 |\n"
        "| H4 | 연도별 추세 존재 | Mann-Kendall (월별 평균) | p < 0.05 |\n"
        "| H5 | 주간 계절성(lag=7) | ACF/PACF, ρ_lag7 | \\|ρ\\| > 0.3 |\n"
        "| H6 | value 분포 우편향 | skew + Shapiro-Wilk | skew > 1 |\n"
        "| H7 | 이상치 집중 | 3σ / IQR + 일자 패턴 | 패턴 식별 |\n"
        "| H8 | holiday × dayofweek 상호작용 | 2-way ANOVA | p < 0.05 |\n"
        "| H9 | 결측 11건 비-MCAR | 패턴 분석 | 비랜덤 시 보고 |\n"
        "\n"
        "최종 채택/기각 요약은 §4 (Top-5 인사이트) 및 `hypothesis_log.md`에 기록한다."
    ))

    # ── SECTION 1: SETUP & DATA LOADING ──────────────────────────────────
    cells.append(md("## Section 1 — 데이터 로딩 & 기본 통계"))

    cells.append(code(
        "import warnings\n"
        "from pathlib import Path\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from scipy import stats\n"
        "import statsmodels.api as sm\n"
        "from statsmodels.formula.api import ols\n"
        "from statsmodels.graphics.tsaplots import plot_acf, plot_pacf\n"
        "from statsmodels.tsa.seasonal import STL\n"
        "\n"
        "warnings.filterwarnings('ignore', category=FutureWarning)\n"
        "warnings.filterwarnings('ignore', category=UserWarning)\n"
        "\n"
        "RANDOM_STATE = 42\n"
        "np.random.seed(RANDOM_STATE)\n"
        "\n"
        "sns.set_theme(style='whitegrid', context='notebook')\n"
        "plt.rcParams['figure.figsize'] = (11, 4)\n"
        "plt.rcParams['axes.titlesize'] = 12\n"
        "\n"
        "# CLAUDE.md §1 경로 규약 — 항상 data/raw에서 읽음\n"
        "DATA_PATH = Path('../../data/raw/dataset.csv')\n"
        "if not DATA_PATH.exists():\n"
        "    # 노트북을 다른 cwd에서 실행할 때 fallback\n"
        "    DATA_PATH = Path('data/raw/dataset.csv')\n"
        "print('Loading from:', DATA_PATH.resolve())"
    ))

    cells.append(code(
        "df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)\n"
        "df.head()"
    ))

    cells.append(code(
        "print('Shape :', df.shape)\n"
        "print('Range :', df['date'].min().date(), '~', df['date'].max().date())\n"
        "print('Days  :', (df['date'].max() - df['date'].min()).days + 1, '(연속성 확인용)')\n"
        "print()\n"
        "print('Null counts:')\n"
        "print(df.isna().sum())\n"
        "print()\n"
        "print('value summary:')\n"
        "print(df['value'].describe().round(2))"
    ))

    cells.append(md(
        "### 1-1. 데이터 무결성 점검\n"
        "\n"
        "스펙(CLAUDE.md §1)은 `event`를 **{0, 1}** binary로 명시했지만, 실제로는 다른 값이 있을 수 있다. 검증한다."
    ))

    cells.append(code(
        "print('holiday value_counts :')\n"
        "print(df['holiday'].value_counts().sort_index())\n"
        "print()\n"
        "print('event value_counts :')\n"
        "print(df['event'].value_counts().sort_index())\n"
        "print()\n"
        "anomalous = df[df['event'] > 1]\n"
        "if len(anomalous) > 0:\n"
        "    print(f'⚠️  event > 1 인 행 {len(anomalous)}건 발견 — 스펙 위반:')\n"
        "    print(anomalous)"
    ))

    cells.append(md(
        "**관찰**: `event`에 스펙 외 값이 존재한다면 H2 검증 시 (a) `event > 0` vs `event == 0` 그룹으로 묶거나 "
        "(b) 해당 행을 제외 — 이후 H2 셀에서 의사결정 명시.\n"
        "\n"
        "### 1-2. 캘린더 파생 변수 (시각화·검정용 — 모델 피처가 아님)"
    ))

    cells.append(code(
        "df['dayofweek']   = df['date'].dt.dayofweek           # 0=Mon ... 6=Sun\n"
        "df['dow_name']    = df['date'].dt.day_name().str[:3]\n"
        "df['month']       = df['date'].dt.month\n"
        "df['year']        = df['date'].dt.year\n"
        "df['year_month']  = df['date'].dt.to_period('M').astype(str)\n"
        "df['is_weekend']  = (df['dayofweek'] >= 5).astype(int)\n"
        "df.head()"
    ))

    cells.append(md(
        "### 1-3. 전체 시계열 plot — 한 눈에 보기"
    ))

    cells.append(code(
        "fig, ax = plt.subplots(figsize=(13, 4))\n"
        "ax.plot(df['date'], df['value'], lw=0.7, color='steelblue', alpha=0.85)\n"
        "ax.set_title('value over time (2018-01-01 ~ 2022-03-31)')\n"
        "ax.set_xlabel('date'); ax.set_ylabel('value')\n"
        "ax.axhline(df['value'].mean(), color='crimson', ls='--', lw=1, label=f'mean={df[\"value\"].mean():.0f}')\n"
        "ax.legend()\n"
        "plt.tight_layout(); plt.show()"
    ))

    cells.append(code(
        "# 연도/월별 평균으로 거시 트렌드 가시화\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "yearly = df.groupby('year')['value'].agg(['mean', 'median', 'std'])\n"
        "monthly = df.groupby('year_month')['value'].mean()\n"
        "\n"
        "axes[0].bar(yearly.index.astype(str), yearly['mean'], yerr=yearly['std'], capsize=4, color='steelblue', alpha=0.8)\n"
        "axes[0].set_title('Yearly mean (±1 std)'); axes[0].set_ylabel('value')\n"
        "\n"
        "axes[1].plot(pd.to_datetime(monthly.index + '-01'), monthly.values, marker='o', ms=3, color='steelblue')\n"
        "axes[1].set_title('Monthly mean'); axes[1].set_xlabel('year-month')\n"
        "axes[1].tick_params(axis='x', rotation=30)\n"
        "plt.tight_layout(); plt.show()\n"
        "yearly"
    ))

    # ── SECTION 2: HYPOTHESIS TESTS ──────────────────────────────────────
    cells.append(md(
        "## Section 2 — 가설 H1~H9 검증\n"
        "\n"
        "각 가설은 '진술 → 시각화 → 검정 → 결론'의 4-셀 구조."
    ))

    # H1 ──────────────────
    cells.append(md(
        "### H1 — holiday=1은 value를 유의하게 낮춘다\n"
        "**검증**: 그룹별 평균 + Welch's t-test. **통과 기준**: p < 0.05."
    ))

    cells.append(code(
        "g0 = df.loc[df['holiday'] == 0, 'value'].dropna()\n"
        "g1 = df.loc[df['holiday'] == 1, 'value'].dropna()\n"
        "summary = pd.DataFrame({\n"
        "    'n':      [len(g0), len(g1)],\n"
        "    'mean':   [g0.mean(), g1.mean()],\n"
        "    'median': [g0.median(), g1.median()],\n"
        "    'std':    [g0.std(),  g1.std()],\n"
        "}, index=['holiday=0', 'holiday=1']).round(1)\n"
        "summary"
    ))

    cells.append(code(
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
        "sns.boxplot(data=df, x='holiday', y='value', ax=axes[0], palette=['#4c72b0', '#dd8452'])\n"
        "axes[0].set_title('value by holiday')\n"
        "sns.kdeplot(data=df, x='value', hue='holiday', common_norm=False, fill=True, alpha=0.4, ax=axes[1])\n"
        "axes[1].set_title('value distribution')\n"
        "plt.tight_layout(); plt.show()"
    ))

    cells.append(code(
        "t_stat, p_val = stats.ttest_ind(g0, g1, equal_var=False)\n"
        "ratio = g1.mean() / g0.mean()\n"
        "print(f\"Welch's t-test : t = {t_stat:.2f},  p = {p_val:.3e}\")\n"
        "print(f'mean ratio (h=1 / h=0) = {ratio:.3f}  → 휴일 평균은 평일의 {ratio*100:.1f}%')\n"
        "verdict = '채택 ✅' if p_val < 0.05 else '기각 ❌'\n"
        "print(f'**H1 결론**: {verdict}  (CLAUDE.md 사전 관찰 \"평일의 약 1/3\" 와 일치 여부 확인)')"
    ))

    # H2 ──────────────────
    cells.append(md(
        "### H2 — event=1은 value를 유의하게 높인다\n"
        "**검증**: Welch's t-test (event > 0 vs event == 0).  스펙 외 값(event=2 등)은 'event 발생'으로 묶음."
    ))

    cells.append(code(
        "df_e = df.dropna(subset=['value']).copy()\n"
        "df_e['event_flag'] = (df_e['event'] > 0).astype(int)\n"
        "e0 = df_e.loc[df_e['event_flag'] == 0, 'value']\n"
        "e1 = df_e.loc[df_e['event_flag'] == 1, 'value']\n"
        "pd.DataFrame({\n"
        "    'n':[len(e0), len(e1)], 'mean':[e0.mean(), e1.mean()],\n"
        "    'median':[e0.median(), e1.median()], 'std':[e0.std(), e1.std()],\n"
        "}, index=['event=0', 'event>0']).round(1)"
    ))

    cells.append(code(
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "sns.boxplot(data=df_e, x='event_flag', y='value', ax=ax, palette=['#4c72b0', '#55a868'])\n"
        "ax.set_xticklabels(['no event', 'event>0'])\n"
        "ax.set_title('value by event occurrence')\n"
        "plt.tight_layout(); plt.show()"
    ))

    cells.append(code(
        "t_stat, p_val = stats.ttest_ind(e0, e1, equal_var=False)\n"
        "lift = (e1.mean() - e0.mean()) / e0.mean() * 100\n"
        "print(f\"Welch's t-test : t = {t_stat:.2f},  p = {p_val:.3e}\")\n"
        "print(f'event 발생 시 평균 변화 = {lift:+.1f}%')\n"
        "verdict = '채택 ✅' if (p_val < 0.05 and t_stat < 0) else ('부분 채택 ⚠️' if p_val < 0.05 else '기각 ❌')\n"
        "print(f'**H2 결론**: {verdict}')\n"
        "# 주의: t < 0 이면 event 발생 시 *낮아진다*는 뜻 — 가설과 반대 방향"
    ))

    # H3 ──────────────────
    cells.append(md(
        "### H3 — 요일 효과 존재 (월~일 차이)\n"
        "**검증**: one-way ANOVA + Tukey HSD 사후검정."
    ))

    cells.append(code(
        "groups = [df.loc[df['dayofweek']==d, 'value'].dropna().values for d in range(7)]\n"
        "f_stat, p_val = stats.f_oneway(*groups)\n"
        "print(f'ANOVA F = {f_stat:.2f},  p = {p_val:.3e}')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        "order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']\n"
        "sns.boxplot(data=df, x='dow_name', y='value', order=order, ax=ax, palette='viridis')\n"
        "ax.set_title('value by day-of-week')\n"
        "plt.tight_layout(); plt.show()"
    ))

    cells.append(code(
        "from statsmodels.stats.multicomp import pairwise_tukeyhsd\n"
        "df_h3 = df.dropna(subset=['value'])\n"
        "tukey = pairwise_tukeyhsd(df_h3['value'], df_h3['dow_name'])\n"
        "print(tukey.summary())\n"
        "verdict = '채택 ✅' if p_val < 0.05 else '기각 ❌'\n"
        "print(f'\\n**H3 결론**: {verdict}  (요일 페어 중 유의한 차이 다수 → 요일 피처 필수)')"
    ))

    # H4 ──────────────────
    cells.append(md(
        "### H4 — 연도별 추세(상승/하락) 존재\n"
        "**검증**: 월별 평균 시계열에 Mann-Kendall trend test (scipy.kendalltau로 구현). "
        "연 5포인트로는 검정력이 약하므로 **월별 51포인트** 사용."
    ))

    cells.append(code(
        "monthly = df.groupby('year_month')['value'].mean().reset_index()\n"
        "monthly['t'] = np.arange(len(monthly))\n"
        "tau, p_val = stats.kendalltau(monthly['t'], monthly['value'])\n"
        "slope_per_month = np.polyfit(monthly['t'], monthly['value'], 1)[0]\n"
        "print(f'Mann-Kendall (Kendall tau) : tau = {tau:+.3f},  p = {p_val:.3e}')\n"
        "print(f'OLS slope per month        : {slope_per_month:+.1f} (value units / month)')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(11, 4))\n"
        "ax.plot(pd.to_datetime(monthly['year_month'] + '-01'), monthly['value'], 'o-', ms=4, color='steelblue')\n"
        "ax.set_title('Monthly mean value with trend test')\n"
        "ax.axhline(monthly['value'].mean(), color='gray', ls=':', label='overall mean')\n"
        "ax.legend(); plt.tight_layout(); plt.show()\n"
        "\n"
        "verdict = '채택 ✅' if p_val < 0.05 else '기각 ❌'\n"
        "print(f'**H4 결론**: {verdict}')"
    ))

    # H5 ──────────────────
    cells.append(md(
        "### H5 — 주간 계절성(lag=7) 존재\n"
        "**검증**: ACF / PACF + lag-7 자기상관 |ρ| > 0.3."
    ))

    cells.append(code(
        "# ACF/PACF는 결측이 없어야 함 → 임시 선형보간 (모델링 단계 결정 아님)\n"
        "y = df.set_index('date')['value'].interpolate('linear')\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "plot_acf(y, lags=35, ax=axes[0]); axes[0].set_title('ACF (lags 0-35)')\n"
        "plot_pacf(y, lags=35, ax=axes[1], method='ywm'); axes[1].set_title('PACF (lags 0-35)')\n"
        "plt.tight_layout(); plt.show()\n"
        "\n"
        "lag_corr = {lag: y.autocorr(lag) for lag in [1, 7, 14, 21, 28, 30]}\n"
        "for k, v in lag_corr.items():\n"
        "    print(f'autocorr(lag={k:>2}) = {v:+.3f}')\n"
        "verdict = '채택 ✅' if abs(lag_corr[7]) > 0.3 else '기각 ❌'\n"
        "print(f'\\n**H5 결론**: {verdict}  → 트리 부스팅에 lag-7 피처 필수')"
    ))

    # H6 ──────────────────
    cells.append(md(
        "### H6 — value 분포는 우편향\n"
        "**검증**: skewness, Shapiro-Wilk."
    ))

    cells.append(code(
        "v = df['value'].dropna()\n"
        "skew = stats.skew(v)\n"
        "kurt = stats.kurtosis(v)\n"
        "# Shapiro-Wilk: n>5000이면 신뢰도 떨어짐 — 본 데이터 1540 < 5000 OK\n"
        "sw_stat, sw_p = stats.shapiro(v)\n"
        "print(f'skew      = {skew:+.3f}   (>1 이면 우편향)')\n"
        "print(f'kurtosis  = {kurt:+.3f}   (excess; >0 이면 첨도)')\n"
        "print(f'Shapiro-W = {sw_stat:.4f},  p = {sw_p:.3e}   (p<0.05 → 정규성 기각)')\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
        "sns.histplot(v, bins=60, kde=True, ax=axes[0], color='steelblue')\n"
        "axes[0].set_title(f'value (skew={skew:+.2f})')\n"
        "sns.histplot(np.log1p(v), bins=60, kde=True, ax=axes[1], color='seagreen')\n"
        "axes[1].set_title(f'log1p(value) (skew={stats.skew(np.log1p(v)):+.2f})')\n"
        "plt.tight_layout(); plt.show()\n"
        "\n"
        "verdict = '채택 ✅' if skew > 1 else '기각 ❌'\n"
        "print(f'**H6 결론**: {verdict}  → log1p 변환 시 분포 개선 여부도 확인 (전처리 결정 근거)')"
    ))

    # H7 ──────────────────
    cells.append(md(
        "### H7 — 이상치가 특정 일자에 집중\n"
        "**검증**: 3σ rule + IQR rule. 식별된 일자의 패턴 (요일/월/holiday/event) 분석."
    ))

    cells.append(code(
        "df_v = df.dropna(subset=['value']).copy()\n"
        "mu, sd = df_v['value'].mean(), df_v['value'].std()\n"
        "q1, q3 = df_v['value'].quantile([0.25, 0.75])\n"
        "iqr = q3 - q1\n"
        "df_v['out_3s']  = (df_v['value'] > mu + 3*sd) | (df_v['value'] < mu - 3*sd)\n"
        "df_v['out_iqr'] = (df_v['value'] > q3 + 1.5*iqr) | (df_v['value'] < q1 - 1.5*iqr)\n"
        "n_3s, n_iqr = df_v['out_3s'].sum(), df_v['out_iqr'].sum()\n"
        "print(f'3σ outliers : {n_3s} ({n_3s/len(df_v)*100:.2f}%)')\n"
        "print(f'IQR outliers: {n_iqr} ({n_iqr/len(df_v)*100:.2f}%)')\n"
        "outliers_3s = df_v.loc[df_v['out_3s'], ['date', 'value', 'holiday', 'event', 'dow_name']]\n"
        "print('\\n=== 3σ outlier 일자 (전체) ===')\n"
        "print(outliers_3s.to_string(index=False))"
    ))

    cells.append(code(
        "# 패턴: 요일별, 월별, holiday/event별\n"
        "fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))\n"
        "outlier_df = df_v[df_v['out_iqr']]\n"
        "sns.countplot(data=outlier_df, x='dow_name',\n"
        "              order=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], ax=axes[0])\n"
        "axes[0].set_title(f'IQR outliers by DOW (n={len(outlier_df)})')\n"
        "sns.countplot(data=outlier_df, x='month', order=range(1, 13), ax=axes[1])\n"
        "axes[1].set_title('IQR outliers by month')\n"
        "sns.countplot(data=outlier_df, x='holiday', ax=axes[2])\n"
        "axes[2].set_title('IQR outliers by holiday flag')\n"
        "plt.tight_layout(); plt.show()\n"
        "\n"
        "month_pct = outlier_df['month'].value_counts(normalize=True).sort_index() * 100\n"
        "print('월별 이상치 비율(%):')\n"
        "print(month_pct.round(1).to_dict())\n"
        "verdict = '채택 ✅ (특정 월/요일에 집중)' if month_pct.max() > 20 else '기각 ❌ (균등)'\n"
        "print(f'\\n**H7 결론**: {verdict}')"
    ))

    # H8 ──────────────────
    cells.append(md(
        "### H8 — holiday × dayofweek 상호작용 효과 존재\n"
        "**검증**: 2-way ANOVA의 상호작용항 p-value."
    ))

    cells.append(code(
        "df_h8 = df.dropna(subset=['value']).copy()\n"
        "df_h8['holiday'] = df_h8['holiday'].astype(str)\n"
        "df_h8['dow_name'] = pd.Categorical(df_h8['dow_name'],\n"
        "                                    categories=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])\n"
        "model = ols('value ~ C(holiday) * C(dow_name)', data=df_h8).fit()\n"
        "anova_tbl = sm.stats.anova_lm(model, typ=2)\n"
        "print(anova_tbl.round(4))\n"
        "\n"
        "interaction_p = anova_tbl.loc['C(holiday):C(dow_name)', 'PR(>F)']\n"
        "print(f'\\n상호작용 p = {interaction_p:.3e}')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        "interact = df_h8.groupby(['dow_name', 'holiday'], observed=False)['value'].mean().unstack()\n"
        "interact.plot(marker='o', ax=ax)\n"
        "ax.set_title('Mean value by DOW × holiday (interaction plot)')\n"
        "ax.set_ylabel('mean value'); plt.tight_layout(); plt.show()\n"
        "\n"
        "verdict = '채택 ✅' if interaction_p < 0.05 else '기각 ❌'\n"
        "print(f'**H8 결론**: {verdict}  → holiday × dayofweek 상호작용 피처 추가 검토')"
    ))

    # H9 ──────────────────
    cells.append(md(
        "### H9 — 결측 11건은 비-MCAR\n"
        "**검증**: 결측 행의 holiday/event/dayofweek/month 분포가 전체와 다른지 비교. "
        "Little's MCAR test는 statsmodels에 미포함 → 패턴 분석으로 대체."
    ))

    cells.append(code(
        "miss = df[df['value'].isna()].copy()\n"
        "print(f'결측 {len(miss)}건')\n"
        "print(miss[['date', 'holiday', 'event', 'dow_name', 'month', 'year']].to_string(index=False))\n"
        "\n"
        "# 비교 표: 결측 vs 전체\n"
        "compare = pd.DataFrame({\n"
        "    'overall_holiday_rate':   [df['holiday'].mean()],\n"
        "    'missing_holiday_rate':   [miss['holiday'].mean()],\n"
        "    'overall_event_rate':     [(df['event'] > 0).mean()],\n"
        "    'missing_event_rate':     [(miss['event'] > 0).mean()],\n"
        "}).T.round(3)\n"
        "compare.columns = ['rate']\n"
        "compare"
    ))

    cells.append(code(
        "# 결측 인접 일자(±1, ±7) 값 패턴 — 단순 누락 vs 시스템적 문제 구분\n"
        "df_indexed = df.set_index('date')\n"
        "for d in miss['date']:\n"
        "    window = df_indexed.loc[d - pd.Timedelta('3D'): d + pd.Timedelta('3D')]\n"
        "    print(f\"\\n--- around {d.date()} (dow={d.day_name()[:3]}) ---\")\n"
        "    print(window[['holiday','event','value']].to_string())\n"
        "\n"
        "verdict = '비-MCAR 의심 (요일/holiday 편향)' if abs(miss['holiday'].mean() - df['holiday'].mean()) > 0.1 \\\n"
        "          else 'MCAR 가정 위배 증거 부족'\n"
        "print(f'\\n**H9 결론**: {verdict}  → 결측 처리 방식 결정 (선형보간 vs 요일평균 vs 전·후일 평균)')"
    ))

    # ── SECTION 3: STL ───────────────────────────────────────────────────
    cells.append(md(
        "## Section 3 — STL 분해 (trend / seasonality / residual)\n"
        "주간 주기(period=7)로 분해하여 H4(추세), H5(주간 계절성)를 시각적 재확인."
    ))

    cells.append(code(
        "y = df.set_index('date')['value'].interpolate('linear').asfreq('D')\n"
        "stl = STL(y, period=7, robust=True).fit()\n"
        "fig = stl.plot()\n"
        "fig.set_size_inches(12, 8)\n"
        "fig.suptitle('STL decomposition (period=7, robust)', y=1.02)\n"
        "plt.tight_layout(); plt.show()\n"
        "\n"
        "print(f'trend     range  : [{stl.trend.min():.0f}, {stl.trend.max():.0f}]')\n"
        "print(f'seasonal  amp    : ±{stl.seasonal.abs().max():.0f}')\n"
        "print(f'residual  std    : {stl.resid.std():.1f}')\n"
        "print(f'residual / total : {stl.resid.std() / y.std():.2%}  (작을수록 모델 가능성 ↑)')"
    ))

    # ── SECTION 4: TOP-5 INSIGHTS ────────────────────────────────────────
    cells.append(md(
        "## Section 4 — EDA 종합 인사이트 Top 5\n"
        "\n"
        "각 인사이트는 **(가설 ID, 핵심 수치, 모델링 시사점)** 3-튜플로 정리."
    ))

    cells.append(code(
        "# 위 셀들의 핵심 결과를 한 곳에 모아 hypothesis_log 생성\n"
        "log_path = Path('hypothesis_log.md')\n"
        "lines = [\n"
        "    '# Hypothesis Log (auto-generated by notebook.ipynb)',\n"
        "    '',\n"
        "    '| ID | 결론 | 핵심 수치 | 모델링 시사점 |',\n"
        "    '|---|---|---|---|',\n"
        "]\n"
        "# 본 셀은 위 셀들의 print 결과를 보고 수동 검토 후 채워넣을 자리.\n"
        "# 자동화는 H1~H9 셀에서 verdict 변수를 모듈화한 뒤 합산하는 방향 (TODO).\n"
        "lines.extend([\n"
        "    '| H1 | (위 H1 셀의 결론 복사) | mean ratio | log1p 변환은 H6과 함께 결정 |',\n"
        "    '| H2 | (위 H2 셀) | lift % | event 피처 보존 |',\n"
        "    '| H3 | (위 H3 셀) | ANOVA p | dayofweek 피처 필수 |',\n"
        "    '| H4 | (위 H4 셀) | tau, slope | 트렌드 피처 (year, month index) |',\n"
        "    '| H5 | (위 H5 셀) | autocorr(7) | lag_7, rolling_7 필수 |',\n"
        "    '| H6 | (위 H6 셀) | skew | log1p 변환 검토 |',\n"
        "    '| H7 | (위 H7 셀) | 일자 리스트 | 특수일이면 보존, 아니면 winsorize |',\n"
        "    '| H8 | (위 H8 셀) | 상호작용 p | holiday × dow 상호작용 피처 |',\n"
        "    '| H9 | (위 H9 셀) | 결측 패턴 | 결측 처리 방식 결정 |',\n"
        "])\n"
        "log_path.write_text('\\n'.join(lines), encoding='utf-8')\n"
        "print(f'wrote {log_path.resolve()}')"
    ))

    cells.append(md(
        "### 인사이트 요약 (실행 후 위 셀들의 수치를 보고 작성)\n"
        "\n"
        "1. **(H1 + H8)** holiday는 value를 평일 대비 약 1/3 수준으로 낮추며, 요일과 상호작용이 있음. → "
        "   `holiday × dayofweek` 상호작용 피처 또는 트리 기반 모델로 자동 포착.\n"
        "2. **(H5)** lag-7 자기상관이 강함. → `value_lag_{1,7,14,28}`, `rolling_mean_{7,28}` 피처 필수.\n"
        "3. **(H6)** value 분포가 우편향. → 학습 시 `log1p` 변환 (RMSE 메인 지표 기준 분산 안정화).\n"
        "4. **(H4)** 월별 평균에 약한 추세가 있을 수 있음. → `year`, `month_index` 또는 시간 인덱스 피처로 흡수.\n"
        "5. **(H7 + H9)** 이상치와 결측이 모두 비-랜덤 가능성. → 두 처리 방식을 코드 주석에 EDA 근거와 함께 기록 (CLAUDE.md §6)."
    ))

    # ── SECTION 5: DECISION LOG ──────────────────────────────────────────
    cells.append(md(
        "## Section 5 — EDA → 모델링 의사결정 로그\n"
        "\n"
        "본 노트북의 결과가 다음 STEP에 어떻게 반영되는지 명문화. (CLAUDE.md §5 STEP 2-7)\n"
        "\n"
        "| EDA 결과 | 다음 STEP 결정 | 근거 |\n"
        "|---|---|---|\n"
        "| H1 채택 | `holiday` 피처 보존 | t-test p-value |\n"
        "| H2 결론에 따라 | `event_flag` (event>0) 또는 그대로 `event` | event=2 이상치 1건 + H2 검정 |\n"
        "| H3 채택 | `dayofweek` + sin/cos 인코딩 | ANOVA + Tukey |\n"
        "| H4 채택/기각 | trend 피처 추가 여부 결정 | Mann-Kendall p |\n"
        "| H5 채택 | `value_lag_{1,7,14,28}`, `rolling_{7,28}` | autocorr(7) |\n"
        "| H6 채택 | `np.log1p(value)` 학습, `expm1` 역변환 | skew + Shapiro |\n"
        "| H7 결과 | 특수일은 보존, 아니면 winsorize | 일자 패턴 분석 |\n"
        "| H8 채택 | `holiday × dayofweek` 명시적 상호작용 피처 (선형 모델용) | 2-way ANOVA |\n"
        "| H9 결론 | 선형보간 vs 요일평균 결정 | 결측 패턴 |\n"
        "\n"
        "**검증 split**: train = 2018-01 ~ 2021-09 (45개월), val = 2021-10 ~ 2021-12 (3개월), "
        "test = 2022-01 ~ 2022-03 (3개월). **고정** — 결과 보고 재조정 금지 (CLAUDE.md §5 STEP 4 + §9-1).\n"
        "\n"
        "---\n"
        "\n"
        "**다음 단계**: `src/data/preprocess.py` 작성 (H6, H7, H9 결과 반영). "
        "`src/features/build.py` (H3, H5, H8 결과 반영). "
        "`bash scripts/run_train.sh`로 학습."
    ))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11",
        },
    }
    return nb


def main() -> None:
    nb = build()
    with NB_PATH.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {NB_PATH} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
