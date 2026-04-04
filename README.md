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
- 发送注册验证码：`POST http://127.0.0.1:8000/auth/register/code`
- 用户注册：`POST http://127.0.0.1:8000/auth/register`
- 用户登录：`POST http://127.0.0.1:8000/auth/login`
- 文件上传：`POST http://127.0.0.1:8000/files/upload`
- 全局文件列表：`GET http://127.0.0.1:8000/files/global`
- 我的文件列表：`GET http://127.0.0.1:8000/files/mine`
- 删除文件：`DELETE http://127.0.0.1:8000/files/{file_id}`
- 上传任务列表：`GET http://127.0.0.1:8000/files/upload-tasks`
- 上传任务详情：`GET http://127.0.0.1:8000/files/upload-tasks/{task_id}`
- 重试上传任务：`POST http://127.0.0.1:8000/files/upload-tasks/{task_id}/retry`
- RAG 聊天：`POST http://127.0.0.1:8000/chat/completions`

## 跨域配置

后端现在支持通过环境变量控制 CORS。只有在配置了 `CORS_ALLOW_ORIGINS` 或 `CORS_ALLOW_ORIGIN_REGEX` 时，应用才会启用跨域中间件。

- `.env.example` 默认放开常见本机前端地址：`5173`、`3000`
- 多个来源、方法或请求头使用英文逗号分隔
- 默认会暴露 `X-Request-ID` 响应头，便于前端联调、日志检索和问题追踪
- 如果前后端通过 Cookie 维持会话，需要把 `CORS_ALLOW_CREDENTIALS` 调整为 `true`

示例：

```powershell
CORS_ALLOW_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
CORS_ALLOW_METHODS=GET,POST,PUT,PATCH,DELETE,OPTIONS
CORS_ALLOW_HEADERS=Content-Type,Authorization,X-Request-ID
CORS_EXPOSE_HEADERS=X-Request-ID
```

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

## 用户认证与用户管理

当前版本已接入基于 `MySQL + JWT` 的用户体系。

- 公开接口：`/auth/register/code`、`/auth/register`、`/auth/login`、`/health/live`、`/docs`、`/openapi.json`、`/redoc`
- 受保护接口：根路径 `/`、`/files/upload`、`/search/chunks`、`/chat/completions`、`/auth/me`、`/auth/password`
- 管理员接口：`/admin/users`
- 登录标识固定为邮箱
- 注册流程默认采用“先发邮箱验证码，再提交注册”的两步校验模式
- JWT 默认有效期为 7 天
- 当前版本不提供 `logout`，前端删除本地 token 即视为退出

认证与注册邮箱验证码相关配置已写入 [.env.example](/e:/PracticalProject/BaozhiRAG/.env.example)。部署前至少需要补齐以下变量：

```powershell
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=baozhi_rag
MYSQL_USERNAME=baozhi
MYSQL_PASSWORD=baozhi123456
JWT_SECRET_KEY=replace-with-a-long-random-secret
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_DAYS=7
OSS_REGION=cn-hangzhou
OSS_ENDPOINT=https://oss-cn-hangzhou.aliyuncs.com
OSS_BUCKET_NAME=your-bucket-name
OSS_ACCESS_KEY_ID=your-access-key-id
OSS_ACCESS_KEY_SECRET=your-access-key-secret
OSS_OBJECT_PREFIX=knowledge-files
REGISTRATION_CODE_SECRET=replace-with-a-long-random-registration-code-secret
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=notice@example.com
SMTP_PASSWORD=replace-with-smtp-password
SMTP_USE_TLS=true
SMTP_USE_SSL=false
SMTP_FROM_EMAIL=notice@example.com
SMTP_FROM_NAME=Baozhi RAG Service
```

其中注册验证码邮件的发件地址由 `SMTP_FROM_EMAIL` 决定，发件人展示名称由 `SMTP_FROM_NAME` 决定。

登录成功后，请在业务请求头中携带：

```text
Authorization: Bearer <access_token>
```

### 认证接口

- `POST /auth/register/code`
  - 字段：`email`
  - 返回：`expires_in`、`expires_at`、`resend_after`
  - 作用：向注册邮箱发送验证码邮件
- `POST /auth/register`
  - 字段：`email`、`password`、`username`、`verification_code`
  - 默认创建 `user` 角色
- `POST /auth/login`
  - 字段：`email`、`password`
  - 返回：`access_token`、`token_type`、`expires_in`、`expires_at`、`user`
- `GET /auth/me`
  - 返回当前登录用户信息
- `PATCH /auth/me`
  - 仅允许修改当前用户的 `username`
- `PATCH /auth/password`
  - 字段：`current_password`、`new_password`

### 管理员接口

- `GET /admin/users`
- `GET /admin/users/{user_id}`
- `POST /admin/users`
- `PATCH /admin/users/{user_id}`
- `DELETE /admin/users/{user_id}`

管理员账号不会由应用自动创建，需要部署阶段手工插入数据库。推荐先生成 Argon2 密码哈希：

```powershell
uv run python -c "from pwdlib import PasswordHash; print(PasswordHash.recommended().hash('Admin12345'))"
```

然后执行 SQL：

```sql
INSERT INTO users (
  id,
  email,
  username,
  password_hash,
  role,
  created_at,
  updated_at
) VALUES (
  'replace_with_uuid_hex',
  'admin@example.com',
  'admin_user',
  'replace_with_argon2_hash',
  'admin',
  UTC_TIMESTAMP(),
  UTC_TIMESTAMP()
);
```

## 文件上传

当前版本把文件上传改为“提交任务 + 后台处理”模式，提供 Word 文件上传、阿里云 OSS 原文持久化、用户级文件元数据管理、原始文件与内容双层去重、后台切块向量化、ES 文档入库、Milvus 向量入库与 chunk 混合检索闭环。

- 接口：`POST /files/upload`
- 请求类型：`multipart/form-data`
- 字段名：`files`
- 当前支持：`.docx`、`.doc`
- 成功时返回 `202 Accepted`
- 业务输入错误返回 `4xx`
- 下游依赖或系统故障返回 `5xx`
- 成功响应示例：`{"state": "success", "message": "上传任务已创建", "data": {"file_count": 1, "tasks": []}, "request_id": "7f6f4f9f..."}` 
- 失败响应示例：`{"state": "error", "code": "unsupported_document_type", "message": "暂不支持的文件格式: .txt", "request_id": "7f6f4f9f..."}` 
- 上传后的原始文件会持久化到阿里云 OSS，本地 `UPLOAD_ROOT_DIR` 仅作为临时接收和 worker 下载目录
- 最终知识文件对象会落到 `knowledge-files/<用户id>/<file_id>/<文件名>`，原始去重 blob 仍保留在 `knowledge-files/raw/...`
- `POST /files/upload` 只负责接收文件、计算原始哈希、登记任务和复用重复任务
- 解析、去重、向量化、ES/Milvus 写入全部由后台 worker 异步完成
- 可通过 `GET /files/upload-tasks` 和 `GET /files/upload-tasks/{task_id}` 轮询任务状态
- 失败任务可通过 `POST /files/upload-tasks/{task_id}/retry` 直接重试，无需重新上传大文件
- 可通过 `GET /files/global` 分页查询管理员上传的全局文件
- 可通过 `GET /files/mine` 分页查询当前用户自己上传的文件
- 可通过 `DELETE /files/{file_id}` 删除当前用户自己上传的知识文件
- 两个列表接口都会返回分页信息，以及可直接用于前端渲染的临时文件地址 `file_url`
- 去重分为两层：
  - `raw_sha256`：解决同一大文件重复提交、重复重传、双击上传
  - `content_sha256`：解决“二进制不同但正文相同”的重复入库
- 管理员上传文件默认全员可检索；普通用户上传文件仅上传者本人可检索
- 同一用户、同一文件名、内容相同：任务完成后标记为重复入库
- 同一用户、同一文件名、内容不同：任务完成后按最新版本覆盖旧文件
- 同一用户、内容相同但文件名不同：任务完成后只更新标题，不重复入库
- 上传后会自动写入 ES 文档索引与 Milvus 向量集合
- 删除知识文件时会同步移除数据库文件记录、检索 chunk 与最终知识文件对象；共享的原始去重 blob 继续保留，用于审计与历史任务重试

示例：

```powershell
curl -X POST "http://127.0.0.1:8000/files/upload" `
  -H "accept: application/json" `
  -F "files=@example.docx"
```

`UPLOAD_ROOT_DIR` 通过 `UPLOAD_ROOT_DIR` 配置，默认值为 `data/uploads`，当前只作为本地临时工作目录。
阿里云 OSS 配置通过 `OSS_REGION`、`OSS_ENDPOINT`、`OSS_BUCKET_NAME`、`OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET`、`OSS_OBJECT_PREFIX` 提供。
切块窗口和旧版 Word 转换临时目录分别通过 `DOC_CHUNK_SIZE`、`DOC_CHUNK_OVERLAP`、`DOC_CONVERT_TEMP_DIR`、`DOC_CONVERT_TIMEOUT_SECONDS` 配置。
异步上传任务相关配置通过 `UPLOAD_INGEST_VERSION`、`UPLOAD_WORKER_CONCURRENCY`、`UPLOAD_WORKER_POLL_INTERVAL_SECONDS`、`UPLOAD_TASK_LEASE_SECONDS`、`UPLOAD_TASK_HEARTBEAT_INTERVAL_SECONDS` 提供。
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
- ES 检索字段：`content`、`merged_terms`、`uploader_user_id`、`visibility_scope`
  - `content` 使用 `ik_max_word` 建索引，`ik_smart` 做查询分析
- Milvus 检索字段：`content_embedding`，并结合 `uploader_user_id`、`visibility_scope` 执行权限过滤
- 结果融合策略：基于 ES 和 Milvus 的 Reciprocal Rank Fusion
- 成功响应中的业务结果放在 `data` 字段
- 检索权限规则：`visibility_scope = global` 的文件所有用户可见，`visibility_scope = owner_only` 的文件仅上传者本人可见

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
        "storage_key": "knowledge-files/user-1/file-1/保险条款.docx",
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

## RAG 聊天

当前版本新增 `POST /chat/completions` 接口，用于基于知识库检索结果执行聊天补全，并支持 SSE 流式输出。

- 请求体字段：`messages`
- 可选字段：`retrieval_size`、`temperature`、`stream`
- 检索策略：默认使用最后一条 `user` 消息作为检索查询
- 风控约束：命中证据时优先基于证据回答；未命中证据时仍会调用模型，但高风险问题必须明确说明当前无知识库依据，不能输出条款或理赔类确定性结论
- 非流式响应优先消费 `data.assistant_message` 与 `data.trace`
- `data.answer`、`data.citations`、`data.finish_reason` 等旧字段仍保留一段兼容期
- 流式响应类型：`text/event-stream`
- 流式事件类型：`message.start`、`citation.add`、`message.delta`、`message.end`、`message.error`

非流式示例：

```powershell
curl -X POST "http://127.0.0.1:8000/chat/completions" `
  -H "Content-Type: application/json" `
  -d '{
    "messages": [
      {"role": "user", "content": "什么是免赔额"}
    ],
    "retrieval_size": 4
  }'
```

流式示例：

```powershell
curl -N -X POST "http://127.0.0.1:8000/chat/completions" `
  -H "Content-Type: application/json" `
  -d '{
    "messages": [
      {"role": "user", "content": "什么是免赔额"}
    ],
    "retrieval_size": 4,
    "stream": true
  }'
```

非流式响应示例：

```json
{
  "state": "success",
  "message": "聊天完成",
  "data": {
    "assistant_message": {
      "message_id": "9d3e0cbbe9f24c4a9fbeb26d61a11c8d",
      "role": "assistant",
      "plain_text": "免赔额通常指理赔时需要由被保险人自行承担的部分。",
      "content_blocks": [
        {
          "block_id": "blk-1",
          "block_type": "markdown",
          "text": "免赔额通常指理赔时需要由被保险人自行承担的部分。",
          "citation_ids": ["cit-1"],
          "sequence": 1
        }
      ],
      "citations": [
        {
          "id": "cit-1",
          "chunk_id": "chunk-1",
          "file_id": "file-1",
          "source_filename": "保险条款.docx",
          "storage_key": "knowledge-files/user-1/file-1/保险条款.docx",
          "chunk_index": 0,
          "char_count": 20,
          "content": "免赔额是指理赔时由被保险人自行承担的金额。",
          "snippet": "免赔额是指理赔时由被保险人自行承担的金额。",
          "merged_terms": ["免赔额"],
          "score": 0.98,
          "heading_path": [],
          "section_title": null,
          "content_type": "paragraph",
          "source_anchor": "chunk:0"
        }
      ],
      "finish_reason": "stop"
    },
    "trace": {
      "request_id": "7f6f4f9f8c5c4f7db38f4f75dcb2f6c1",
      "original_query": "什么是免赔额",
      "retrieval_query": "什么是免赔额",
      "rewrite_applied": false,
      "model": "qwen-plus",
      "usage": null,
      "latency_ms": 128
    },
    "answer": "免赔额通常指理赔时需要由被保险人自行承担的部分。[1]",
    "retrieval_query": "什么是免赔额",
    "citation_count": 1,
    "citations": [
      {
        "id": "cit-1",
        "chunk_id": "chunk-1",
        "file_id": "file-1",
        "source_filename": "保险条款.docx",
        "storage_key": "knowledge-files/user-1/file-1/保险条款.docx",
        "chunk_index": 0,
        "char_count": 20,
        "content": "免赔额是指理赔时由被保险人自行承担的金额。",
        "snippet": "免赔额是指理赔时由被保险人自行承担的金额。",
        "merged_terms": ["免赔额"],
        "score": 0.98,
        "heading_path": [],
        "section_title": null,
        "content_type": "paragraph",
        "source_anchor": "chunk:0"
      }
    ],
    "finish_reason": "stop"
  },
  "request_id": "7f6f4f9f8c5c4f7db38f4f75dcb2f6c1"
}
```

流式事件说明：

- `message.start`：声明消息开始，返回 `message_id`、`request_id`、查询链路与模型信息
- `citation.add`：增量下发单条可展示的引用卡片
- `message.delta`：返回正文增量，包含 `seq`、`offset` 与本次 `text`
- `message.end`：返回最终 `assistant_message` 与 `trace`
- `message.error`：当 SSE 已开始后发生异常时返回错误事件

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

## 服务器部署

服务器部署请优先使用 `docker-compose.server.yml`，该编排与本机联调版的主要区别是：

- 仅对外暴露 `Nginx` 的 `6888` 端口
- `app`、`MySQL`、`Elasticsearch`、`Milvus`、`MinIO` 只在容器内网通信
- Nginx 已补充 SSE 代理参数，适配 `/chat/completions` 的流式输出

部署步骤：

```powershell
cp .env.example .env
vi .env
docker compose -f docker-compose.server.yml build --progress=plain
docker compose -f docker-compose.server.yml up -d
```

也可以使用仓库顶层脚本一键部署（脚本内命令为分开执行）：

```bash
chmod +x deploy_server.sh
./deploy_server.sh
```

建议至少确认以下环境变量已经填写：

- `MYSQL_DATABASE`
- `MYSQL_USERNAME`
- `MYSQL_PASSWORD`
- `MYSQL_ROOT_PASSWORD`
- `JWT_SECRET_KEY`
- `OSS_BUCKET_NAME`
- `OSS_ACCESS_KEY_ID`
- `OSS_ACCESS_KEY_SECRET`
- `DASHSCOPE_API_KEY`
- `BAILIAN_CHAT_MODEL`
- `CHUNK_EMBEDDING_MODEL`
- `CHUNK_EMBEDDING_DIMENSIONS`

服务端镜像构建默认会通过 `.env` 中的以下变量使用腾讯云镜像源，适合腾讯云服务器环境：

- `APP_BUILD_APT_MIRROR=https://mirrors.cloud.tencent.com`
- `APP_BUILD_PYPI_MIRROR=https://mirrors.cloud.tencent.com/pypi/simple`

如果构建阶段在 `uv sync --frozen --no-dev --no-install-project` 卡住，可优先保留 `--progress=plain` 观察具体停留的包名；若怀疑镜像源同步不及时，可临时切换到官方源：

```powershell
APP_BUILD_APT_MIRROR=https://deb.debian.org
APP_BUILD_PYPI_MIRROR=https://pypi.org/simple
docker compose -f docker-compose.server.yml build --progress=plain
docker compose -f docker-compose.server.yml up -d
```

常用排查命令：

```powershell
docker compose -f docker-compose.server.yml ps
docker compose -f docker-compose.server.yml logs -f nginx
docker compose -f docker-compose.server.yml logs -f app
docker compose -f docker-compose.server.yml logs -f mysql
docker compose -f docker-compose.server.yml logs -f elasticsearch
docker compose -f docker-compose.server.yml logs -f milvus
```

停止与清理：

```powershell
docker compose -f docker-compose.server.yml down
docker compose -f docker-compose.server.yml down -v
```

部署完成后可通过以下地址访问：

- `http://服务器IP:6888/`
- `http://服务器IP:6888/docs`
- `http://服务器IP:6888/health/live`

如果服务器前面还会挂云负载均衡、CDN 或 HTTPS 网关，建议把 TLS 终止放在最外层；当前仓库内置的 Nginx 主要负责反向代理和 SSE 透传。

### tests 目录说明

- `tests/` 保留在仓库中，用于本地开发、CI 和回归验证
- 镜像构建时不会复制 `tests/`，因为 [.dockerignore](/e:/PracticalProject/BaozhiRAG/.dockerignore#L1) 已排除该目录
- 运行容器不依赖 `tests/`，服务器上无需单独删除它

## 提交规范

项目推荐采用 Conventional Commits。推荐格式如下：

```text
feat(api): 初始化健康检查接口
fix(config): 修复环境变量读取异常
docs(readme): 补充启动说明
```

详细协作约定见 `docs/development.md`。
项目级代理与协作约定见 `AGENTS.md`。





