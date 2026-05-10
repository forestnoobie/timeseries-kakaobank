# CLAUDE.md §5 STEP 7 — python:3.11-slim, latest 태그 금지
FROM python:3.11-slim AS base

# 시스템 의존성 (lightgbm/xgboost/torch wheels에 필요한 최소셋)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app

# requirements 먼저 (레이어 캐시)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install "git+https://github.com/uber-research/TuRBO.git@master"
# ↑ TuRBO: PyPI 미배포 → git 직접 설치. Uber Non-Commercial 라이선스(README 명시).

# 소스 복사
COPY . /app

# 산출물 디렉터리 (volume mount 권장)
RUN mkdir -p /app/outputs/analytics /app/outputs/models /app/outputs/hpo /app/outputs/predictions

# 기본 동작: EDA 노트북 실행 (CMD는 scripts/*.sh로 override 가능)
CMD ["bash", "scripts/run_eda.sh"]
