"""集中读取配置。

所有环境变量只在这里读一次，其他模块 import settings 使用。
这样换模型、换中转站时只改 .env，不用改代码。
"""
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # 把项目根目录 .env 里的键值读进环境变量


@dataclass(frozen=True)  # frozen=True 表示创建后不可修改，防止运行中被误改
class Settings:
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    # 部分模型（如 Qwen3 系列）默认每次回答前先做一段隐藏思考，客服场景下既慢又费 token。
    # None 表示不干预，True/False 表示显式开关。不是所有模型都认这个参数，所以默认不发送。
    llm_enable_thinking: Optional[bool] = None
    embedding_model: str = "text-embedding-v4"
    kb_path: str = "data/chroma"   # 向量库的本地存储目录，已在 .gitignore 里
    # 检索增强开关。依据 scripts/compare_retrieval.py 的实测结果设定默认值，见该脚本
    retrieval_rerank: bool = True
    retrieval_rewrite: bool = False
    # 知识库工具的保护参数
    kb_timeout_s: float = 8.0          # 整次检索的总时限
    kb_rerank_timeout_s: float = 4.0   # 其中重排一步的时限，必须小于总时限
    kb_cache_ttl_s: float = 300.0      # 相同问题 5 分钟内不重复检索
    kb_breaker_failures: int = 3       # 连续失败 3 次后熔断
    kb_breaker_recovery_s: float = 30.0
    redis_url: str = "redis://localhost:6379/0"
    judge_model: str = "gpt-4.1-mini"   # 评测用的评委模型，刻意和主模型不同厂商
    # 分环节选模型。意图分类和重排只输出一个标签或一串序号，不需要最强的模型，用快的。
    # 留空表示沿用 llm_model。
    intent_model: str = ""
    rerank_model: str = ""
    # 意图分类模型那一票由谁投："llm" 用对话模型（配合 intent_model），"jev" 用 TypeSafe 的 jev
    intent_backend: str = "llm"
    rerank_backend: str = "llm"   # 知识库重排由谁做，取值同上
    jev_model: str = "jev-1.13"
    jev_timeout_s: float = 10.0
    # 连接闲置多久后由客户端关闭。httpx 默认 5 秒，比用户两次发消息的间隔还短，
    # 于是几乎每轮都要重新建连接，经代理的冷连接要 6 到 10 秒。实测中转站那一侧闲置 60 秒的连接仍可复用，
    # 120 秒时已被对端关闭。对端先关也不会报错，httpx 会发现并换一条新连接，只是又变回冷连接的耗时
    http_keepalive_s: float = 60.0


def _optional_bool(name: str) -> Optional[bool]:
    raw = os.getenv(name, "").strip().lower()
    if raw in ("true", "1", "yes"):
        return True
    if raw in ("false", "0", "no"):
        return False
    return None


BACKENDS = ("llm", "jev")


def _backend(name: str) -> str:
    value = os.getenv(name, "").strip().lower() or "llm"
    if value not in BACKENDS:
        # 拼错时直接报错。静默退回 llm 的话，你以为在用 jev，评测数字其实是 LLM 的
        raise RuntimeError(f"{name}={value!r} 无效，可选 {BACKENDS}")
    return value


def load_settings() -> Settings:
    return Settings(
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://aihubmix.com/v1").strip(),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        llm_enable_thinking=_optional_bool("LLM_ENABLE_THINKING"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4").strip(),
        kb_path=os.getenv("KB_PATH", "data/chroma").strip(),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0").strip(),
        judge_model=os.getenv("JUDGE_MODEL", "gpt-4.1-mini").strip(),
        intent_model=os.getenv("INTENT_MODEL", "").strip(),
        rerank_model=os.getenv("RERANK_MODEL", "").strip(),
        intent_backend=_backend("INTENT_BACKEND"),
        rerank_backend=_backend("RERANK_BACKEND"),
        jev_model=os.getenv("JEV_MODEL", "").strip() or "jev-1.13",
        retrieval_rerank=_optional_bool("RETRIEVAL_RERANK") is not False,    # 未设置时为 True
        retrieval_rewrite=_optional_bool("RETRIEVAL_REWRITE") is True,       # 未设置时为 False
    )


settings = load_settings()
