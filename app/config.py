"""集中读取配置。

所有环境变量只在这里读一次，其他模块 import settings 使用。
这样换模型、换中转站时只改 .env，不用改代码。
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # 把项目根目录 .env 里的键值读进环境变量


@dataclass(frozen=True)  # frozen=True 表示创建后不可修改，防止运行中被误改
class Settings:
    llm_api_key: str
    llm_base_url: str
    llm_model: str


def load_settings() -> Settings:
    return Settings(
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://aihubmix.com/v1").strip(),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
    )


settings = load_settings()
