# LightWhite 当前 Word Chunk 逻辑说明

本文档描述当前仓库里已经实现的 Word 切块、领域词增强、可选 Elasticsearch 入库与检索逻辑。内容以现有代码真实行为为准，不是目标设计稿。

涉及的核心实现文件如下：

- `src/baozhi_rag/api/files.py`
- `src/baozhi_rag/api/search.py`
- `src/baozhi_rag/api/dependencies.py`
- `src/baozhi_rag/services/document_preview.py`
- `src/baozhi_rag/services/document_chunking.py`
- `src/baozhi_rag/services/term_matching.py`
- `src/baozhi_rag/services/chunk_search.py`
- `src/baozhi_rag/domain/term_dictionary.py`
- `src/baozhi_rag/infra/retrieval/elasticsearch_chunk_store.py`
- `src/baozhi_rag/core/config.py`

## 1. 当前整体链路

当前 Word 相关逻辑已经不是单纯“上传后打印一个 ES 预览载荷”，而是分成两条链路：

1. 上传切块链路：`POST /files/upload`
2. 检索链路：`GET /search/chunks`

### 1.1 上传切块链路

完整调用链如下：

1. API 层收到 `multipart/form-data` 的 `files` 列表，`files` 表示待上传文件数组。
2. `files.py` 通过依赖函数 `get_document_preview_service()` 构造：
   - `LocalFileStore`
   - `FileUploadService`
   - `DocumentChunkService`
   - 可选的 `ElasticsearchChunkStore`
   - `DocumentPreviewService`
3. `DocumentPreviewService.upload_and_preview_files()` 先调用 `FileUploadService.upload_files()` 完成本地落盘。
4. 每个文件落盘后，调用 `DocumentChunkService.chunk_document()`。
5. `chunk_document()` 根据扩展名分发到：
   - `.docx` -> `chunk_docx()`
   - `.doc` -> `chunk_doc()`
6. `_build_chunks()` 在生成每个 chunk 时，会立刻执行领域词匹配：
   - `fmm_terms`
   - `bmm_terms`
   - `merged_terms`
7. 切块完成后：
   - 日志打印 chunk 总数、每个 chunk 的完整正文、对应的 ES 文档预览
   - API 响应返回前 3 个 chunk 的摘要和 `merged_terms`
8. 如果 `ES_ENABLED=true`，则 `DocumentPreviewService` 还会继续：
   - 确保 ES 索引存在
   - 将当前文件全部 chunk 批量写入 ES
9. 如果上传、切块或 ES 入库过程中任何一步失败：
   - 已成功入库的 `file_id` 会尝试从 ES 回滚删除
   - 本次请求已落盘文件会逆序删除
   - 异常继续抛出到 API 层

### 1.2 检索链路

`GET /search/chunks` 的调用链如下：

1. API 层接收查询参数 `q` 与可选 `size`。
2. `search.py` 通过依赖函数 `get_chunk_search_service()` 构造：
   - 默认领域词词典
   - `ChunkSearchService`
   - `ElasticsearchChunkStore`
3. `ChunkSearchService.search()` 先对查询文本执行：
   - FMM 正向最大匹配
   - BMM 逆向最大匹配
   - 合并去重得到 `merged_terms`
4. `ElasticsearchChunkStore.search()` 组合：
   - `content` 的全文 `match`
   - `merged_terms` 的 `terms`
   - `fmm_terms` 的 `terms`
   - `bmm_terms` 的 `terms`
5. 返回 chunk 命中结果，包括正文、词项字段和 `_score`

## 2. 配置项

当前相关配置集中在 `core/config.py`。

### 2.1 切块与文档转换

- `DOC_CHUNK_SIZE`：单个 chunk 的目标最大字符数
  - 默认值：`800`
- `DOC_CHUNK_OVERLAP`：相邻 chunk 之间的重叠字符数
  - 默认值：`150`
- `DOC_CONVERT_TEMP_DIR`：`.doc` 转 `.docx` 的临时目录
  - 默认值：`data/tmp/converted`

### 2.2 领域词增强

- `DOMAIN_DICTIONARY_PATH`：可选的外部领域词典文件路径
  - 默认值：空
  - 行为：如果配置，会在内置金融保险词典基础上继续叠加文件中的词项

### 2.3 Elasticsearch 相关

- `ES_ENABLED`：是否启用上传后自动入 ES 与检索接口
  - `Settings` 默认值：`false`
  - `.env.example` 当前示例值：`true`
- `ES_URL`：Elasticsearch 服务地址
  - 默认值：`http://127.0.0.1:9200`
- `ES_INDEX_NAME`：chunk 写入的索引名称
  - 默认值：`document_chunks`
- `ES_USERNAME`：Elasticsearch 用户名
  - 默认值：空
- `ES_PASSWORD`：Elasticsearch 密码
  - 默认值：空
- `ES_API_KEY`：Elasticsearch 的 API Key
  - 默认值：空
- `ES_VERIFY_CERTS`：是否校验证书
  - `Settings` 默认值：`true`
  - `.env.example` 当前示例值：`false`
- `SEARCH_DEFAULT_SIZE`：`GET /search/chunks` 默认返回条数
  - 默认值：`10`

## 3. 文件类型分发逻辑

`chunk_document()` 是 Word 切块总入口，实际行为如下：

1. 读取 `file_path.suffix.lower()` 作为扩展名。
2. 执行 `match suffix`：
   - 如果是 `.docx`，直接走 `chunk_docx()`
   - 如果是 `.doc`，走 `chunk_doc()`
   - 否则抛出 `UnsupportedDocumentTypeError`
3. 切块成功后调用 `_log_chunk_preview()` 打日志
4. 返回标准化 `DocumentChunk` 列表

不支持格式时的错误文案形如：

```text
暂不支持的文件格式: .txt
```

## 4. `.docx` 的切块逻辑

### 4.1 读取文档

`chunk_docx()` 使用 `python-docx` 的 `Document(str(file_path))` 打开文件。

行为如下：

1. 如果 `python-docx` 打开失败，抛出 `DocumentParseError`
2. 错误文案形如：

```text
解析 docx 文件失败: 原始文件名
```

### 4.2 抽取段落

打开文档后，会进入 `_extract_docx_segments()`。这一步不是直接按段落输出 chunk，而是先把段落整理成一个片段列表 `segments`。

处理规则：

1. 遍历 `document.paragraphs`
2. 取 `paragraph.text.strip()`
3. 如果结果为空字符串，直接跳过
4. 读取段落样式名 `paragraph.style.name`
5. 调用 `_parse_heading_level(style_name)` 判断是不是标题

`_parse_heading_level()` 使用正则：

```python
r"(Heading|标题)\s*(\d+)"
```

可识别：

- `Heading 1`
- `标题 1`

返回规则：

- 如果是标题样式，返回标题层级整数
- 如果不是标题样式，返回 `None`
- 如果 `style_name` 是 `None`，返回 `None`

### 4.3 标题上下文维护方式

实现里维护了一个 `headings: list[str]`，用于保存当前生效的标题路径。

#### 当前段落是标题时

例如识别到 `Heading 2`，处理规则是：

1. 只保留 `headings[: heading_level - 1]`
2. 把当前标题文本追加进去
3. 将当前完整标题路径 `" / ".join(headings)"` 作为独立片段写入 `segments`

这意味着标题本身会进入后续 chunk 文本，不只是作为元数据。

#### 当前段落不是标题时

处理规则是：

1. 取当前标题路径 `context = " / ".join(headings)`
2. 如果 `context` 不为空，则构造：

```text
标题路径
正文段落
```

也就是 `f"{context}\n{text}"`

3. 如果 `context` 为空，就直接使用正文段落文本
4. 将结果写入 `segments`

#### 示例

如果文档结构是：

```text
Heading 1: 保险责任
正文A
正文B
Heading 2: 免责条款
正文C
```

则 `segments` 大致会变成：

```text
[
  "保险责任",
  "保险责任\n正文A",
  "保险责任\n正文B",
  "保险责任 / 免责条款",
  "保险责任 / 免责条款\n正文C",
]
```

### 4.4 把片段拼成总文本

`chunk_docx()` 不会直接对 `segments` 逐段切块，而是先执行：

```python
"\n\n".join(segments)
```

也就是说：

- 每个片段之间额外插入两个换行
- 最终得到一个长字符串 `text`

如果 `segments` 为空，会抛出：

```text
Word 文档内容为空，无法切块: 原始文件名
```

## 5. 真正的 chunk 切分算法

真正的切分发生在 `_build_chunks()`，它仍然是“基于整段长文本的固定字符窗口滑动切分”，不是“按自然段落输出一个 chunk”。

### 5.1 输入

输入参数有：

- `text`
  - 已经完成标题补全并通过 `"\n\n".join(segments)` 拼好的长文本
- `source_filename`
- `storage_key`
- `file_id`

### 5.2 预处理

先执行：

```python
normalized_text = text.strip()
```

如果结果为空，抛 `DocumentParseError`。

### 5.3 滑动窗口切分

核心变量：

- `start = 0`
- `end = min(start + chunk_size, len(normalized_text))`

每轮逻辑如下：

1. 取子串：

```python
chunk_content = normalized_text[start:end].strip()
```

2. 如果 `chunk_content` 非空，则生成一个 `DocumentChunk`
3. 生成 chunk 前，对 `chunk_content` 调用领域词匹配器：
   - `extract_terms(chunk_content)`
   - 结果挂入 `fmm_terms`
   - 结果挂入 `bmm_terms`
   - 合并去重结果挂入 `merged_terms`
4. 如果 `end >= len(normalized_text)`，结束循环
5. 否则计算下一轮起点：

```python
next_start = end - chunk_overlap
start = next_start if next_start > start else end
```

这个判断的作用是避免极端边界条件下的死循环。

### 5.4 当前算法的真实特征

当前实现的关键特征如下：

- 是按字符数切，不按 token 数切
- 是对整篇拼接后的长文本切，不是按段落切完再合并
- overlap 是纯字符回退，不是按句子边界回退
- 超长段落不会单独特殊处理，只会自然被滑动窗口切开
- chunk 边界不保证落在句号、段落或标题边界上
- 领域词匹配是在 chunk 生成后、按 chunk 文本执行的，不是在整篇文档级别一次性执行

### 5.5 chunk 对象字段

每个 chunk 当前包含：

- `file_id`：原始文件所属文件标识
- `chunk_id`：chunk 唯一标识，格式为 `{file_id}-chunk-{chunk_index}`
- `chunk_index`：chunk 序号，从 `0` 开始递增
- `content`：当前窗口切出的正文内容
- `char_count`：当前 chunk 的字符数，等于 `len(chunk_content)`
- `source_filename`：上传时原始文件名
- `storage_key`：本地存储相对路径
- `fmm_terms`：FMM 正向最大匹配结果
- `bmm_terms`：BMM 逆向最大匹配结果
- `merged_terms`：由 `fmm_terms + bmm_terms` 按出现顺序去重得到的合并词项

## 6. FMM（正向最大匹配）和 BMM（反向最大匹配）

FMM 的逻辑是从句子的**最左端（开头）**开始，向右进行扫描和匹配。

BMM 的逻辑与 FMM 相反，它从句子的**最右端（末尾）**开始，向左进行扫描和匹配。

### 6.1 词典来源

默认词典来自 `domain/term_dictionary.py` 中的内置金融保险词集合，例如：

- `保单`
- `保险责任`
- `责任免除`
- `免赔额`
- `理赔`
- `重大疾病保险`

如果配置了 `DOMAIN_DICTIONARY_PATH`，则会继续加载外部文本文件：

- 按行读取
- 去除空行
- `#` 开头视为注释
- 和内置词典合并

### 6.2 FMM 规则

正向最大匹配从左向右扫描文本：

1. 从当前位置开始
2. 尝试匹配词典允许的最长词长
3. 命中后写入结果，并将游标前进到词末
4. 如果未命中，则游标只前进 1 个字符

### 6.3 BMM 规则

逆向最大匹配从右向左扫描文本：

1. 从当前位置向左尝试匹配最长词项
2. 命中后写入结果，并将游标回退到词首
3. 如果未命中，则游标只回退 1 个字符
4. 最后再把结果反转回原始阅读顺序

### 6.4 合并规则

`merged_terms` 的生成规则是：

```python
list(dict.fromkeys([*fmm_terms, *bmm_terms]))
```

也就是：

- 先保留全部 `fmm_terms`
- 再拼接全部 `bmm_terms`
- 再按出现顺序去重

## 7. `.doc` 的处理逻辑

`.doc` 不直接解析，而是通过 `chunk_doc()` 先转成 `.docx`。

### 7.1 转换逻辑

`_convert_doc_to_docx()` 的流程：

1. 构造输出目录：

```text
{doc_convert_temp_dir}/{file_stem}
```

例如：

```text
data/tmp/converted/contract
```

2. 调用外部命令：

```text
soffice --headless --convert-to docx --outdir <output_dir> <file_path>
```

3. 期望输出文件路径：

```text
{output_dir}/{file_stem}.docx
```

4. 如果未安装 `soffice`，抛出：

```text
未找到 soffice，无法解析 .doc 文件
```

5. 如果命令执行失败或输出文件不存在，抛出：

```text
.doc 转换失败: <stderr 或 stdout 文案>
```

### 7.2 转换后的切块

转换成功后：

1. `chunk_doc()` 直接调用 `chunk_docx(converted_path, ...)`
2. 后续切块、词匹配、日志、ES 文档组装逻辑与 `.docx` 完全一致

### 7.3 临时文件清理

无论成功还是失败，`chunk_doc()` 都会在 `finally` 里调用 `_cleanup_converted_file()`：

1. 删除临时生成的 `.docx`
2. 尝试删除其父目录
3. 如果删除失败，静默忽略

## 8. ES 文档模型与入库逻辑

### 8.1 当前 ES 文档结构

`DocumentChunk.to_search_document()` 会构造真实 ES 文档，而不是“只打印一个未来预览”。

当前字段包括：

- `chunk_id`：chunk 唯一标识
- `file_id`：所属文件唯一标识
- `source_filename`：原始文件名
- `storage_key`：本地存储相对路径
- `chunk_index`：chunk 序号
- `char_count`：chunk 字符数
- `content`：chunk 正文
- `fmm_terms`：FMM 正向最大匹配词项数组
- `bmm_terms`：BMM 逆向最大匹配词项数组
- `merged_terms`：FMM 和 BMM 合并去重后的词项数组

因此，日志中的 `document_chunk_es_preview` 现在打印的是“真实 ES 入库文档形状”，不是占位用的基础字段字典。

### 8.2 索引创建逻辑

`ElasticsearchChunkStore.ensure_index()` 会先检查索引是否存在；不存在时调用 `indices.create()` 创建。

当前 mapping 采用：

- `chunk_id`、`file_id`、`source_filename`、`storage_key`：`keyword`，用于精确过滤和标识
- `chunk_index`、`char_count`：`integer`，用于数字字段存储
- `content`：`text`，用于全文检索
- `fmm_terms`、`bmm_terms`、`merged_terms`：`keyword`，用于精确词项匹配

并使用：

```text
dynamic = strict
```

### 8.3 批量入库逻辑

`index_chunks()` 的行为：

1. 先确保索引存在
2. 将 chunk 列表转换成 ES bulk `operations`
3. 使用 `bulk(..., refresh="wait_for")` 写入
4. 如果 ES 返回 `errors=true`，会提取第一条错误原因并抛出异常

### 8.4 删除与回滚逻辑

按 `file_id` 删除时，使用：

```python
delete_by_query(
    index=index_name,
    query={"term": {"file_id": file_id}},
    refresh=True,
    conflicts="proceed",
)
```

这也是上传链路在 ES 入库部分的回滚手段。

## 9. 控制台日志逻辑

切块完成后，`_log_chunk_preview()` 会打印日志。

### 9.1 总览日志

先打印一条总览：

```text
document_chunk_preview filename=<原始文件名> storage_key=<相对路径> chunk_count=<数量>
```

### 9.2 每个 chunk 的完整日志

当前会打印所有 chunk，每个 chunk 输出：

- `filename`：原始文件名
- `chunk_index`：chunk 序号
- `chunk_id`：chunk 唯一标识
- `char_count`：chunk 字符数
- `content`：chunk 完整正文

日志事件名是：

```text
document_chunk_item_full
```

其中 `content` 是 chunk 的完整正文，不做字符截断。

### 9.3 ES 文档预览日志

每个 chunk 的完整日志之后，还会额外打印一条 ES 文档预览日志。

日志事件名是：

```text
document_chunk_es_preview
```

当前打印字段就是 `to_search_document()` 的完整结果，包括：

- `chunk_id`：chunk 唯一标识
- `file_id`：所属文件唯一标识
- `source_filename`：原始文件名
- `storage_key`：本地存储相对路径
- `chunk_index`：chunk 序号
- `char_count`：chunk 字符数
- `content`：chunk 正文
- `fmm_terms`：FMM 正向最大匹配词项数组
- `bmm_terms`：BMM 逆向最大匹配词项数组
- `merged_terms`：合并去重后的领域词数组

## 10. API 返回给调用方的预览逻辑

### 10.1 上传接口返回

单文件响应包含：

- `chunk_status`：切块状态，当前固定为 `success`
- `chunk_count`：chunk 总数
- `chunk_preview`：最多前 3 个 chunk 的预览数组

每个 `chunk_preview` 元素包含：

- `chunk_index`：chunk 序号
- `char_count`：chunk 字符数
- `preview_text`：用于前端展示的摘要文本
- `merged_terms`：当前 chunk 识别出的合并领域词数组

其中 `preview_text` 的生成规则是：

1. 替换换行为空格
2. 截取前 `160` 个字符

注意：

- 接口不会返回完整 chunk 正文
- 不会返回所有 chunk
- 当前只返回 `merged_terms`，不会把 `fmm_terms` 和 `bmm_terms` 暴露在上传接口预览里

### 10.2 检索接口返回

`GET /search/chunks` 的响应会返回：

- `query`：原始查询文本
- `size`：命中结果数量
- `hits`：命中结果数组

每个 hit 包含：

- `chunk_id`：chunk 唯一标识
- `file_id`：所属文件唯一标识
- `source_filename`：原始文件名
- `storage_key`：本地存储相对路径
- `chunk_index`：chunk 序号
- `char_count`：chunk 字符数
- `content`：chunk 正文
- `fmm_terms`：FMM 正向最大匹配词项数组
- `bmm_terms`：BMM 逆向最大匹配词项数组
- `merged_terms`：合并去重后的领域词数组
- `score`：Elasticsearch 返回的检索得分

## 11. 检索查询逻辑

当前检索不是单纯的 `content match`，而是“全文 + 领域词字段”的混合查询。

`ElasticsearchChunkStore.build_search_query()` 生成的结构为：

1. `content` 的 `match`
   - `boost = 3.0`
2. `merged_terms` 的 `constant_score + terms`
   - `boost = 6.0`
3. `fmm_terms` 的 `constant_score + terms`
   - `boost = 4.0`
4. `bmm_terms` 的 `constant_score + terms`
   - `boost = 4.0`

最外层是：

```python
{
  "bool": {
    "should": [...],
    "minimum_should_match": 1,
  }
}
```

这意味着：

- 命中正文全文也可以返回
- 命中领域词字段也可以返回
- 领域词字段的分值权重高于普通 `content` 匹配

## 12. 失败与回滚逻辑

### 12.1 上传成功但切块失败

`DocumentPreviewService.upload_and_preview_files()` 的行为：

1. 先批量调用 `FileUploadService.upload_files()`
2. 然后逐个对已上传文件调用 `chunk_document()`
3. 如果任意文件切块失败：
   - 逆序删除本次请求已落盘文件
   - 将异常继续向上抛出

### 12.2 ES 入库失败

如果切块成功，但 ES 入库过程中失败：

1. 记录已成功入库的 `file_id`
2. 对这些 `file_id` 逆序执行 `delete_chunks_by_file_id()`
3. 删除本次请求落盘文件
4. 将异常继续向上抛出

实现上，ES 回滚阶段使用 `suppress(Exception)`，也就是：

- 会尽力回滚
- 但不会让回滚过程中的次级异常覆盖主异常

### 12.3 文件删除时的额外清理

`LocalFileStore.delete()` 在删除具体文件后，会继续调用 `_cleanup_empty_parent_dirs()`：

1. 从文件所在目录开始向上
2. 只要目录为空，就继续删除
3. 一直删到上传根目录为止
4. 遇到非空目录就停止

## 13. 当前实现的真实结论

如果要用一句话概括当前实现：

> 现在的粒度本质上是“带标题上下文的全文字符滑窗切块 + 每个 chunk 做 FMM/BMM 词项增强 + 可选 ES 入库与混合检索”。

更细一点讲：

- 段落抽取阶段保留了标题上下文
- 真正切块时仍然是整篇长文本字符滑窗
- FMM/BMM 是按 chunk 文本执行，不是按原始段落执行
- ES 文档已经是真实入库模型，不再只是占位预览
- 检索已经具备接口和基础混合召回逻辑

## 14. 当前实现的限制

当前逻辑仍然存在这些明确限制：

- 只抽取 `document.paragraphs`
- 不处理表格文本
- 不处理页眉、页脚、脚注、尾注、批注
- 不按句号、分号、标题边界对齐 chunk
- 不做 token 级长度控制
- `content` 仍使用 ES 默认文本分析，不依赖中文分词插件
- 领域词词典目前是内置集合加可选外部文本文件，不支持在线热更新
- 当前只实现了 ES 检索，没有接向量检索和混合重排
- 没有 Embedding、Milvus、rerank 接入

## 15. 一个简化示意

假设配置：

- `chunk_size = 20`
- `chunk_overlap = 5`

拼接后的文本是：

```text
保险责任

保险责任
第一段正文abcdefg

保险责任
第二段正文hijklmn
```

那么切块仍更接近下面这种滑窗效果：

```text
chunk0: [0:20]
chunk1: [15:35]
chunk2: [30:50]
...
```

而不是：

```text
chunk0 = 标题 + 第一段
chunk1 = 标题 + 第二段
```

区别在于，当前每个 `chunkN` 生成后还会立刻得到：

```text
fmm_terms
bmm_terms
merged_terms
```

并在启用 ES 时写入 `document_chunks` 索引，供 `/search/chunks` 使用。
