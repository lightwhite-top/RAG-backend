# BaozhiRAG

面向金融保险场景的 RAG 客服系统后端基础工程，当前版本完成了 `FastAPI + uv` 的基础环境、质量门禁和团队协作规范初始化。

## 技术栈

- Python 3.13
- FastAPI
- uv
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

当前版本提供 Word 文件上传与切块预览闭环，用于后续 RAG chunk 入库前验证切块效果。

- 接口：`POST /files/upload`
- 请求类型：`multipart/form-data`
- 字段名：`files`
- 当前支持：`.docx`、`.doc`
- 返回：`file_id`、原始文件名、大小、内容类型、相对存储路径、上传时间、切块状态、切块数量、切块预览

示例：

```powershell
curl -X POST "http://127.0.0.1:8000/files/upload" `
  -H "accept: application/json" `
  -F "files=@example.docx"
```

上传目录通过 `BAOZHI_RAG_UPLOAD_ROOT_DIR` 配置，默认值为 `data/uploads`。
切块窗口和旧版 Word 转换临时目录分别通过 `BAOZHI_RAG_DOC_CHUNK_SIZE`、`BAOZHI_RAG_DOC_CHUNK_OVERLAP`、`BAOZHI_RAG_DOC_CONVERT_TEMP_DIR` 配置。

## 提交规范

项目采用 Conventional Commits，并通过 `commit-msg` 钩子校验。推荐格式如下：

```text
feat(api): 初始化健康检查接口
fix(config): 修复环境变量读取异常
docs(readme): 补充启动说明
```

详细协作约定见 `docs/development.md`。
项目级代理与协作约定见 `AGENTS.md`。
