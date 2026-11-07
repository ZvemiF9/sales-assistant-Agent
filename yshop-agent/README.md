# 电商智能导购 Agent

基于 FastAPI、LangGraph 和本地中文 RAG 的独立服务，支持商品与 SKU 预算查询、规则来源、多轮会话和确认加购。商城业务通过 HTTP API 接入；配置、依赖安装和启动步骤见 [启动说明](../docs/delivery/startup.md)。

## 目录

| 目录 | 内容 |
| --- | --- |
| `app/` | API、查询与确认图、规则检索、商城适配和状态存储 |
| `knowledge/` | 15 篇带明确演示声明的规则 |
| `tests/` | 应用、接口、持久化与命令行工具的自动化测试 |
| `evals/` | 固定评测用例与数据版本说明 |
| `scripts/` | 建库、联调、评测及本地启动工具 |

## 主要接口

除健康检查外，业务接口均要求当前商城的有效 `Authorization`。

| 接口 | 用途 |
| --- | --- |
| `GET /health` | 检查 Agent 存活 |
| `GET /agent/me` | 读取经过商城校验的身份 |
| `GET /agent/products` | 按关键词、分类与可购 SKU 预算查询 |
| `GET /agent/products/{product_id}` | 读取实时商品与规格 |
| `GET /agent/cart` | 读取购物车公开信息 |
| `GET /agent/policies?question=...` | 检索演示规则并返回来源 |
| `POST /agent/chat` | 多轮查询或准备确认卡 |
| `POST /agent/chat/stream` | 展示处理状态与最终聊天结果 |
| `GET /agent/actions/{action_id}` | 读取当前用户的动作状态 |
| `GET /agent/actions/{action_id}/history` | 读取动作记录 |
| `POST /agent/actions/{action_id}/decision` | 接受明确的 `confirm` 或 `cancel` 决定 |

模型仅能提出只读查询工具。购物车写入经过独立确认流程和实时复核；超时或结果未知时不重放写入。会话和动作保存在本地 SQLite，规则索引使用 Chroma。

## 测试与评测

安装开发依赖，并按启动说明准备本地 embedding 模型后，从本目录运行：

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q -W error --basetemp '.tmp/pytest-local'
& '.\.venv\Scripts\python.exe' -m pip check
```

测试使用隔离状态和测试替身。固定真实场景需要另行准备商城、模型及索引；结果与仍未通过的场景见 [评测说明](../docs/delivery/evaluation.md)。

[系统架构](../docs/delivery/architecture.md) · [商城接入](../docs/delivery/integration.md) · [演示视频与操作步骤](../docs/delivery/demo.md)
