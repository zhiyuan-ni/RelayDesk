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


def _optional_bool(name: str) -> Optional[bool]:
    raw = os.getenv(name, "").strip().lower()
    if raw in ("true", "1", "yes"):
        return True
    if raw in ("false", "0", "no"):
        return False
    return None


def load_settings() -> Settings:
    return Settings(
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://aihubmix.com/v1").strip(),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        llm_enable_thinking=_optional_bool("LLM_ENABLE_THINKING"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4").strip(),
        kb_path=os.getenv("KB_PATH", "data/chroma").strip(),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0").strip(),
        retrieval_rerank=_optional_bool("RETRIEVAL_RERANK") is not False,    # 未设置时为 True
        retrieval_rewrite=_optional_bool("RETRIEVAL_REWRITE") is True,       # 未设置时为 False
    )


settings = load_settings()
