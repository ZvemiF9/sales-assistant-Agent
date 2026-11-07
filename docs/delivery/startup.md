# 本地启动说明

仓库包含完整 Python Agent，以及 Java 商品查询扩展和 H5 导购接入文件。后两个目录是供合入已有商城的代码片段，不是可独立编译的完整商城。先按 [接入说明](integration.md) 准备商城 API、商品数据和有效登录。

## 1. 安装 Agent

使用 Python 3.12。以下 PowerShell 命令从克隆后的仓库根目录开始：

```powershell
Set-Location .\yshop-agent
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-dev.txt -c requirements-lock.txt
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
```

`requirements-lock.txt` 记录开发环境中的依赖版本。复制其他机器的 `.venv` 不能替代安装。后续命令均在 `yshop-agent` 目录执行。

## 2. 配置本地环境

编辑 `.env`，参考 [.env.example](../../yshop-agent/.env.example)：

| 配置 | 用途 |
| --- | --- |
| `YSHOP_API_URL` | 已运行的商城 API，包含 `/api/`，例如 `http://127.0.0.1:8008/api/` |
| `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY` | 支持项目工具协议的模型接口、模型名称和密钥 |
| `LLM_PROXY_URL` | 模型请求可选的本机代理；按实际网络填写 |
| `CORS_ORIGINS` | H5 实际来源的 JSON 数组，例如 `["http://127.0.0.1:8081"]` |
| `SESSION_DB_PATH` | 会话与动作数据库，默认 `data/day5-sessions.sqlite3` |
| `RAG_CHROMA_DIR` | 本地规则向量索引，默认 `data/chroma` |
| `YSHOP_AUTHORIZATION` | 仅供本地 CLI 联调的完整 Bearer 凭据；H5 使用自身登录状态 |

Python 通过商城 HTTP 接口访问业务，不直接读写商城数据库。若已有可运行的商城 JAR，可用 `scripts.start_shop --java <Java路径> --jar <JAR路径>`；JAR、MySQL/Redis 和数据库配置由使用者自行准备。

## 3. 构建演示规则索引

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.build_knowledge
```

首次运行需要下载中文 BGE embedding 模型。构建成功后，服务使用本地模型与索引；仓库不包含模型缓存、Chroma 数据库或聊天记录。修改知识或 embedding 配置后，应显式重新建库。

## 4. 启动服务

```powershell
& '.\.venv\Scripts\python.exe' -m uvicorn app.main:app --host 127.0.0.1 --port 8018 --no-access-log
```

在另一个终端检查：

```powershell
Invoke-RestMethod 'http://127.0.0.1:8018/health'
```

接口文档地址为 `http://127.0.0.1:8018/docs`。健康检查只验证 Agent 存活；业务请求仍需要可用的商城、有效身份，以及对应的模型或检索配置。

## 5. 接入 H5 与恢复会话

将仓库中的 H5 文件合入自己的完整 uni-app 商城，按 [H5 接入步骤](integration.md#h5-接入) 注册页面、配置 Agent 地址并安装所需依赖，再使用商城自身的 H5 启动方式。访问 `/pages/agent/Chat/index`，登录后按 [演示步骤](demo.md) 操作。

H5 地址与 `CORS_ORIGINS` 应使用相同的主机和端口。重启 Agent 后继续使用同一个会话数据库、其旁边的检查点文件及规则索引。刷新页面恢复动作 ID 后，先点击“读取动作状态”；该步骤只读取状态，不自动重放确认。

## 6. 运行自动化测试

先完成第 3 步的本地模型准备：其中一项测试会校验真实 embedding 权重及依赖版本，空模型缓存会使该检查失败。

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q -W error --basetemp '.tmp/pytest-local'
& '.\.venv\Scripts\python.exe' -m pip check
```

Python 测试使用隔离状态和测试替身，不要求运行真实商城或配置模型密钥。Java 与 H5 接入测试的运行前提见 [接入说明](integration.md)。真实模型评测需要另行准备服务、登录、模型与索引，不能用单元测试结果替代，见 [评测说明](evaluation.md)。
