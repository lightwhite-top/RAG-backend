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
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
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

当前版本提供 Word 文件上传、领域词增强、ES 文档入库、Milvus 向量入库与 chunk 混合检索闭环。

- 接口：`POST /files/upload`
- 请求类型：`multipart/form-data`
- 字段名：`files`
- 当前支持：`.docx`、`.doc`
- 成功时返回 `200 OK`
- 业务输入错误返回 `4xx`
- 下游依赖或系统故障返回 `5xx`
- 成功响应示例：`{"state": "success", "message": "文件上传成功", "data": {"file_count": 1, "files": []}, "request_id": "7f6f4f9f..."}` 
- 失败响应示例：`{"state": "error", "code": "unsupported_document_type", "message": "暂不支持的文件格式: .txt", "request_id": "7f6f4f9f..."}` 
- 切块增强字段：`merged_terms`
- 上传后会强制执行百炼 embedding，为 `chunk` 补充 `content_embedding`
- 上传后会自动写入 ES 文档索引与 Milvus 向量集合

示例：

```powershell
curl -X POST "http://127.0.0.1:8000/files/upload" `
  -H "accept: application/json" `
  -F "files=@example.docx"
```

上传目录通过 `UPLOAD_ROOT_DIR` 配置，默认值为 `data/uploads`。
切块窗口和旧版 Word 转换临时目录分别通过 `DOC_CHUNK_SIZE`、`DOC_CHUNK_OVERLAP`、`DOC_CONVERT_TEMP_DIR` 配置。
默认领域词典文件位于 `src/baozhi_rag/domain/default_domain_terms.txt`，自定义扩展词典可通过 `DOMAIN_DICTIONARY_PATH` 配置。
ES 连接和索引配置通过 `ES_URL`、`ES_INDEX_NAME`、`ES_USERNAME`、`ES_PASSWORD`、`ES_API_KEY`、`ES_VERIFY_CERTS` 配置。
Milvus 连接和集合配置通过 `MILVUS_URI`、`MILVUS_TOKEN`、`MILVUS_DB_NAME`、`MILVUS_COLLECTION_NAME` 配置。
百炼模型配置通过 `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`、`BAILIAN_TIMEOUT_SECONDS`、`BAILIAN_CHAT_MODEL` 配置。
向量化模型参数通过 `CHUNK_EMBEDDING_MODEL`、`CHUNK_EMBEDDING_DIMENSIONS`、`CHUNK_EMBEDDING_BATCH_SIZE` 配置。

## Chunk 检索

当前版本新增 `GET /search/chunks` 接口，用于基于 ES 文本召回和 Milvus 向量召回执行混合检索。

- 查询参数：`q`
- 可选参数：`size`
- 默认返回条数：`SEARCH_DEFAULT_SIZE`
- ES 检索字段：`content`、`merged_terms`
  - `content` 使用 `ik_max_word` 建索引，`ik_smart` 做查询分析
- Milvus 检索字段：`content_embedding`
- 结果融合策略：基于 ES 和 Milvus 的 Reciprocal Rank Fusion
- 成功响应中的业务结果放在 `data` 字段

示例：

```powershell
curl "http://127.0.0.1:8000/search/chunks?q=免赔额&size=5"
```

响应示例：

```json
{
  "state": "success",
  "message": "检索成功",
  "data": {
    "query": "免赔额",
    "size": 1,
    "hits": [
      {
        "chunk_id": "chunk-1",
        "file_id": "file-1",
        "source_filename": "保险条款.docx",
        "storage_key": "2026/03/28/file-1_保险条款.docx",
        "chunk_index": 0,
        "char_count": 24,
        "content": "本条款包含免赔额和保险责任说明。",
        "merged_terms": ["免赔额", "保险责任"],
        "score": 0.032786
      }
    ]
  },
  "request_id": "7f6f4f9f8c5c4f7db38f4f75dcb2f6c1"
}
```

## 成功响应规范

当前版本统一采用以下成功响应外壳：

```json
{
  "state": "success",
  "message": "操作成功",
  "data": {},
  "meta": {},
  "request_id": "7f6f4f9f8c5c4f7db38f4f75dcb2f6c1"
}
```

字段约定：

- `message` 只放人类可读提示，不承载结构化业务数据
- `data` 只放业务数据主体，前端如果要取结果，优先从这里取
- `meta` 预留给分页、游标、统计等附加信息；当前接口暂未广泛使用
- `request_id` 用于日志检索、审计追踪和问题定位

## 错误处理规范

当前版本已统一接入全局异常处理，设计目标等价于 Spring Boot 中的 `@ControllerAdvice`：

- 路由层不再手写重复的 `try/except` 做 HTTP 映射
- 业务层、服务层、基础设施层优先抛出统一 `AppError` 子类
- FastAPI 全局异常处理器统一把异常翻译为稳定的 HTTP 状态码和错误体
- 每个请求都会返回 `X-Request-ID` 响应头，错误体中也会带上同一个 `request_id`
- 未声明异常统一返回 `500` 和通用消息，避免把内部栈信息直接暴露给调用方

统一错误响应结构：

```json
{
  "state": "error",
  "code": "request_validation_error",
  "message": "请求参数校验失败",
  "request_id": "7f6f4f9f8c5c4f7db38f4f75dcb2f6c1",
  "details": [
    {
      "field": "query.q",
      "message": "字段不能为空"
    }
  ]
}
```

推荐约定：

- `4xx` 表示调用方输入、协议或资源状态问题
- `5xx` 表示系统内部故障或外部依赖故障
- `code` 使用稳定的机器可读标识，前端和调用方基于 `code` 做分支判断
- `message` 使用面向用户或调用方的可读提示
- `request_id` 用于日志检索、审计追踪和问题定位

## 本机检索基础设施

仓库提供了本机开发用的 `Milvus + Elasticsearch` 容器编排文件：`docker-compose.search.yml`。
其中 Elasticsearch 会在构建时自动安装 IK 分词插件。

- `Elasticsearch`：`http://127.0.0.1:9200`
- `Milvus gRPC`：`127.0.0.1:19530`
- `Milvus Health`：`http://127.0.0.1:9091/healthz`
- `MinIO`：`http://127.0.0.1:9001`

启动：

```powershell
docker compose -f docker-compose.search.yml up -d
```

首次启动或改动了 Elasticsearch Dockerfile 后，建议先执行：

```powershell
docker compose -f docker-compose.search.yml build elasticsearch
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

当前应用要求同时准备 Elasticsearch 与 Milvus，复制 `.env.example` 后请确保本机检索基础设施可用：

- `ES_URL=http://127.0.0.1:9200`
- `ES_INDEX_NAME=document_chunks`
- `ES_VERIFY_CERTS=false`
- `MILVUS_URI=http://127.0.0.1:19530`
- `MILVUS_DB_NAME=default`
- `MILVUS_COLLECTION_NAME=document_chunk_vectors`

要启用当前上传与检索链路，请确保已配置阿里云百炼向量化参数：

- `DASHSCOPE_API_KEY=<你的百炼密钥>`
- `CHUNK_EMBEDDING_MODEL=text-embedding-v4`
- `CHUNK_EMBEDDING_DIMENSIONS=1024`

说明：

- 当前工程通过百炼 OpenAI 兼容接口统一封装 embedding 能力，后续接入聊天模型时可直接复用同一客户端。
- ES 只承载文本和结构化检索字段，不再存储 `content_embedding`；其中 `content` 使用 IK 中文分词。
- Milvus 承载 `content_embedding` 向量集合，若向量维度与当前配置不一致，需要重建对应集合。
- 如果在接入 IK 前已经创建过 `document_chunks` 索引，需要删除旧索引并重建，才能让新的 analyzer 生效。

说明：

- 该编排仅面向本机开发与联调，不适用于生产环境。
- Elasticsearch 为单节点并关闭安全认证，便于本地调试。
- Milvus 采用官方 standalone 模式，并带上 `etcd` 与 `MinIO` 依赖容器。

## 提交规范

项目推荐采用 Conventional Commits。推荐格式如下：

```text
feat(api): 初始化健康检查接口
fix(config): 修复环境变量读取异常
docs(readme): 补充启动说明
```

详细协作约定见 `docs/development.md`。
项目级代理与协作约定见 `AGENTS.md`。
