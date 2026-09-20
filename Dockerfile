# 多阶段构建：第一阶段装依赖，第二阶段只带运行需要的东西，镜像更小，也不含构建工具。
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
# 先只拷依赖清单再安装。代码改动时这一层能命中缓存，不用每次重装依赖
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
RUN uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm
WORKDIR /app
# 不用 root 跑服务：容器被攻破时攻击者拿到的也只是普通用户权限
RUN useradd --create-home --uid 1000 relaydesk \
    && mkdir -p /app/data && chown relaydesk:relaydesk /app/data
# 数据目录在镜像里先建好并归属给运行用户。命名卷首次挂载时会沿用镜像里这个目录的属主，
# 否则卷归 root，普通用户写不进去，向量库启动就会报 Permission denied
COPY --from=builder --chown=relaydesk:relaydesk /app/.venv /app/.venv
COPY --chown=relaydesk:relaydesk app ./app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER relaydesk
EXPOSE 8000
# 健康检查用标准库发请求，基础镜像里没有 curl
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
