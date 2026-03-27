# BaozhiRAG

面向金融保险场景的 RAG 客服系统后端基础工程，当前版本完成了 `FastAPI + uv` 的基础环境、质量门禁和团队协作规范初始化。

## 技术栈

- Python 3.13
- FastAPI
- uv
- 阿里云百炼 OpenAI 兼容接口
- Ruff
- Mypy
- Pytest
- Pre-commit

## 目录结构

```text
.
|-- .env.example
|-- .gitmessage
|-- .pre-commit-config.yaml
|-- docs/
|   `-- development.md
|-- pyproject.toml
|-- scripts/
|   `-- check_commit_msg.py
|-- src/
|   `-- baozhi_rag/
|       |-- api/
|       |   |-- health.py
|       |   `-- routes.py
|       |-- app/
|       |   `-- main.py
|       |-- core/
|       |   |-- config.py
|       |   `-- logging.py
|       |-- domain/
|       |-- infra/
|       |   |-- llm/
|       |   |-- retrieval/
|       |   `-- storage/
|       |-- schemas/
|       |   `-- system.py
|       `-- services/
`-- tests/
```

## 快速开始

```powershell
uv sync --all-groups
Copy-Item .env.example .env
uv run pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type commit-msg
uv run uvicorn baozhi_rag.app.main:app --reload
```

如果本机安装了 `just`，也可以直接使用项目内置任务命令：

```powershell
just bootstrap
just dev
```

如果使用 VS Code 打开本仓库，工作区已内置 Python 保存自动格式化配置：

- 保存时使用 `Ruff` 作为默认格式化器
- 保存时执行导入整理与可安全应用的自动修复

首次使用请确保已安装 VS Code 的 Python 与 Ruff 扩展。

## VS Code 调试

仓库已内置 `.vscode/launch.json`，可直接在 VS Code 中使用调试面板启动。

首次使用建议按以下步骤操作：

```powershell
uv sync --all-groups
Copy-Item .env.example .env
```

然后在 VS Code 中：

- 使用 `Python: Select Interpreter` 选择工作区内的 `.venv`
- 打开“运行和调试”面板
- 选择以下任一配置后按 `F5`

可用调试配置：

- `Python 调试程序: FastAPI`
  - 适合稳定打断点
  - 启动命令等价于 `uvicorn baozhi_rag.app.main:app --host 127.0.0.1 --port 8000`
- `Python 调试程序: FastAPI（热重载）`
  - 适合一边改代码一边联调
  - 等价于 `uvicorn baozhi_rag.app.main:app --host 127.0.0.1 --port 8000 --reload`
  - 由于热重载会派生子进程，断点稳定性可能略弱于无热重载模式
- `Python 调试程序: Pytest 当前文件`
  - 适合直接调试当前测试文件

调试配置默认会：

- 读取工作区根目录下的 `.env`
- 将 `src` 注入 `PYTHONPATH`
- 在集成终端中输出日志

服务启动后可访问：

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health/live`

服务启动后访问以下地址：

- 文档地址：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health/live`
- 文件上传：`POST http://127.0.0.1:8000/files/upload`

## 常用命令

```powershell
uv run ruff check .
uv run ruff format .
uv run mypy
uv run pytest
```

对应的 `just` 命令如下：

```powershell
just lint
just format
just typecheck
just test
just test-api
just check
just lock
just build
just clean
just run pytest -k health
```

## 文件上传

当前版本提供 Word 文件上传、领域词增强、ES 入库与 chunk 检索闭环。

- 接口：`POST /files/upload`
- 请求类型：`multipart/form-data`
- 字段名：`files`
- 当前支持：`.docx`、`.doc`
- 返回：`file_id`、原始文件名、大小、内容类型、相对存储路径、上传时间、切块状态、切块数量、切块预览
- 切块增强字段：`fmm_terms`、`bmm_terms`、`merged_terms`
- 可选启用百炼 embedding，为 `chunk` 补充 `content_embedding`
- 上传后会自动创建索引并写入 chunk 文档

示例：

```powershell
curl -X POST "http://127.0.0.1:8000/files/upload" `
  -H "accept: application/json" `
  -F "files=@example.docx"
```

上传目录通过 `UPLOAD_ROOT_DIR` 配置，默认值为 `data/uploads`。
切块窗口和旧版 Word 转换临时目录分别通过 `DOC_CHUNK_SIZE`、`DOC_CHUNK_OVERLAP`、`DOC_CONVERT_TEMP_DIR` 配置。
领域词典扩展文件通过 `DOMAIN_DICTIONARY_PATH` 配置。
ES 连接和索引配置通过 `ES_URL`、`ES_INDEX_NAME`、`ES_USERNAME`、`ES_PASSWORD`、`ES_API_KEY`、`ES_VERIFY_CERTS` 配置。
百炼模型配置通过 `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`、`BAILIAN_TIMEOUT_SECONDS`、`BAILIAN_CHAT_MODEL` 配置。
向量化开关与模型参数通过 `CHUNK_EMBEDDING_ENABLED`、`CHUNK_EMBEDDING_MODEL`、`CHUNK_EMBEDDING_DIMENSIONS`、`CHUNK_EMBEDDING_BATCH_SIZE` 配置。

## Chunk 检索

当前版本新增 `GET /search/chunks` 接口，用于基于 `content`、领域词字段和可选查询向量执行混合召回。

- 查询参数：`q`
- 可选参数：`size`
- 默认返回条数：`SEARCH_DEFAULT_SIZE`
- 混合检索字段：`content`、`fmm_terms`、`bmm_terms`、`merged_terms`
- 当 `CHUNK_EMBEDDING_ENABLED=true` 时，会额外启用 `content_embedding` 的语义检索打分

示例：

```powershell
curl "http://127.0.0.1:8000/search/chunks?q=免赔额&size=5"
```

## 本机检索基础设施

仓库提供了本机开发用的 `Milvus + Elasticsearch` 容器编排文件：`docker-compose.search.yml`。

- `Elasticsearch`：`http://127.0.0.1:9200`
- `Milvus gRPC`：`127.0.0.1:19530`
- `Milvus Health`：`http://127.0.0.1:9091/healthz`
- `MinIO`：`http://127.0.0.1:9001`

启动：

```powershell
docker compose -f docker-compose.search.yml up -d
```

查看状态：

```powershell
docker compose -f docker-compose.search.yml ps
docker compose -f docker-compose.search.yml logs -f elasticsearch
docker compose -f docker-compose.search.yml logs -f milvus
```

停止：

```powershell
docker compose -f docker-compose.search.yml down
```

如果需要连同卷数据一起清空：

```powershell
docker compose -f docker-compose.search.yml down -v
```

当前应用已预留 Elasticsearch 配置项，复制 `.env.example` 后请确保本机 ES 可用：

- `ES_URL=http://127.0.0.1:9200`
- `ES_INDEX_NAME=document_chunks`
- `ES_VERIFY_CERTS=false`

如果要启用阿里云百炼向量化，请额外配置：

- `DASHSCOPE_API_KEY=<你的百炼密钥>`
- `CHUNK_EMBEDDING_ENABLED=true`
- `CHUNK_EMBEDDING_MODEL=text-embedding-v4`
- `CHUNK_EMBEDDING_DIMENSIONS=1024`

说明：

- 当前工程通过百炼 OpenAI 兼容接口统一封装 embedding 能力，后续接入聊天模型时可直接复用同一客户端。
- 已存在的 ES 索引如果还没有 `content_embedding` 字段，应用启动时会尝试补充映射；如果维度和当前配置不一致，需要手动重建索引。

说明：

- 该编排仅面向本机开发与联调，不适用于生产环境。
- Elasticsearch 为单节点并关闭安全认证，便于本地调试。
- Milvus 采用官方 standalone 模式，并带上 `etcd` 与 `MinIO` 依赖容器。

## 提交规范

项目采用 Conventional Commits，并通过 `commit-msg` 钩子校验。推荐格式如下：

```text
feat(api): 初始化健康检查接口
fix(config): 修复环境变量读取异常
docs(readme): 补充启动说明
```

详细协作约定见 `docs/development.md`。
项目级代理与协作约定见 `AGENTS.md`。
