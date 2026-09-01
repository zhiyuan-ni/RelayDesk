# RelayDesk

多 Agent 客服编排服务。开发中，当前处于阶段 0。

## 运行

```bash
cp .env.example .env      # 然后编辑 .env 填入 LLM_API_KEY
uv run uvicorn app.main:app --reload
```

打开 http://localhost:8000/docs 调试接口。

## 测试

```bash
uv run pytest -q
```

## 进度

- [x] 阶段 0：骨架、LLM 客户端、最简 /chat
- [ ] 阶段 1：意图识别与实体抽取
- [ ] 阶段 2：多 Agent 路由与工具调用
- [ ] 阶段 3：知识库与 RAG 工具链
- [ ] 阶段 4：分层记忆
- [ ] 阶段 5：评测集与 LLM Judge
- [ ] 阶段 6：可观测性与部署
