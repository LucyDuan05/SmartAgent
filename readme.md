# 泰迪杯数据挖掘挑战赛 - C题：竞赛智能客服机器人

## 项目概述

本项目旨在构建一个智能客服机器人，用于回答关于各类竞赛的咨询问题。系统能够处理三种类型的问题：
1.  **基本信息查询**：如竞赛时间、地点、主办单位等。
2.  **统计分析查询**：如特定类型竞赛的数量（需结合代码逻辑）。
3.  **开放性问题**：如竞赛准备建议、技术趋势等。

该系统基于检索增强生成（RAG）架构，结合了混合搜索向量数据库和大型语言模型（LLM）的能力，旨在提供准确、流畅且富有洞察力的回答。

## 技术方案

### 1. 总体架构

```
┌───────────────────┐      ┌────────────────────────┐      ┌────────────────┐
│ PDF 文件 (data/raw) │─┐    │  混合搜索向量数据库模块  │─(上下文)→│  问答处理模块  │
└───────────────────┘ │    │ (vector_store.py)      │      │ (qa_system.py) │
                      └─(处理)→│ - PDF 解析与分块       │      └───────┬────────┘
                        │ - Gemini Embedding       │              │
                        │ - FAISS (语义)           │              │ (用户问题)
                        │ - BM25 (关键词)          │              ↓
                        │ - 加权分数融合           │      ┌───────┴────────┐
                        └────────────────────────┘      │ 用户交互界面   │
                                                        │ (命令行/Web等) │
                                                        └────────────────┘
```
*(注：数据更新管理模块和独立的统计分析模块在此方案中未详细实现，重点放在核心问答流程)*

### 2. 核心技术栈

- **编程语言**: Python 3.10+
- **大语言模型 (LLM)**: Google Gemini 2.5 Pro (通过 `google-generativeai` 库访问)
- **文本嵌入 (Embeddings)**: Google Gemini Embedding API (`models/embedding-001`)
- **向量数据库索引**: FAISS (`faiss-cpu`) - 用于语义相似度搜索
- **关键词搜索引擎**: BM25 (`rank_bm25`) - 用于关键词匹配
- **PDF 处理**: PyMuPDF (`fitz`)
- **HTTP 请求**: `requests` (用于通过自定义服务器访问LLM，*如果需要*)
- **数据处理与工具**: NumPy, tqdm, glob

### 3. 模块设计

#### 3.1 向量数据库模块 (`src/vector_store.py` - 集成实现)

该模块负责从原始 PDF 文件构建和查询一个混合搜索引擎，集成了数据处理、嵌入和索引构建：

- **PDF 处理与分块**:
    - 使用 `PyMuPDF(fitz)` 自动遍历 `data/raw/` 目录下的所有 PDF 文件，提取文本。
    - 实现基于字符数和重叠的文本分块策略 (`chunk_text`)，将长文本分割成适合处理的大小（例如 1500 字符，300 字符重叠）。
    - 为每个文本块附加元数据（来源文件名、块索引）。
- **文本嵌入**:
    - 使用 `google-generativeai` 库调用 Google Gemini Embedding API (`models/embedding-001`) 将文本块转换为高质量的向量嵌入。嵌入时指定任务类型为 `RETRIEVAL_DOCUMENT`。
- **混合索引构建**:
    - **FAISS 索引**: 使用 `faiss-cpu` 构建 `IndexFlatL2` 索引，存储文本块的向量嵌入，用于快速语义相似度检索。
    - **BM25 索引**:
        - 使用 `simple_tokenizer` 对文本块进行分词（推荐替换为 `jieba` 以获得更好的中文效果）。
        - 使用 `rank_bm25` 库构建 BM25 索引，用于高效的关键词匹配。
    - 将 FAISS 索引、BM25 索引对象以及带元数据的文本块列表分别保存到 `data/embeddings/` 目录下的 `.index` 和 `.pkl` 文件中。
- **混合搜索与融合**:
    - 实现 `hybrid_search` 方法。
    - 对用户查询，分别执行：
        - **FAISS 搜索**: 使用查询的 Gemini 嵌入 (`RETRIEVAL_QUERY` 类型) 查找语义最相似的 `k` 个文本块，得到距离。
        - **BM25 搜索**: 使用查询的分词结果查找关键词最匹配的 `k` 个文本块，得到 BM25 分数。
    - **加权分数融合**:
        - 将 FAISS 距离转换为相似度分数。
        - 对 FAISS 相似度分数和 BM25 分数进行归一化处理。
        - 根据预设权重 `alpha` (可调，例如 0.5) 计算两种分数的加权和作为最终相关性分数。
    - 返回按最终融合分数排序的 Top-N 结果（包含文本块、元数据和融合分数）。

```python
# 核心类与方法示例
class HybridVectorStore:
    def build_indices(self, chunks_with_metadata, tokenized_corpus, embeddings):
        # 构建并保存 FAISS 和 BM25 索引
        pass
    def load_indices(self):
        # 加载所有索引和元数据
        pass
    def hybrid_search(self, query, top_k, initial_k, alpha):
        # 执行 FAISS 和 BM25 搜索，进行加权分数融合并返回结果
        pass

def process_pdfs_in_directory(pdf_dir, chunk_size, chunk_overlap):
    # 遍历 PDF, 提取文本, 分块, 返回块元数据和分词语料
    pass
```

#### 3.2 问答处理模块 (`src/qa_system.py`)

该模块基于 RAG 架构，协调向量数据库检索和 LLM 生成，以回答用户问题：

- **初始化**:
    - 加载 `HybridVectorStore` 实例，并加载所有索引和元数据。
    - 配置 `google-generativeai` 库，初始化 Google Gemini 2.5 Pro 模型客户端。
- **上下文检索**:
    - 接收用户问题 `user_question`。
    - 调用 `HybridVectorStore` 的 `hybrid_search` 方法，获取与问题最相关的 Top-K (例如 K=10) 个文本块及其元数据，作为上下文。
- **Prompt 工程**:
    - **动态构建 Prompt**: 结合以下元素构建发送给 Gemini 模型的最终 Prompt：
        - **背景知识**: 硬编码的关键全局信息（如专项赛归属、泰迪杯全称、比赛总列表）。
        - **角色定义**: 指示模型扮演"专业、严谨且富有启发性的竞赛信息智能客服"。
        - **行为准则与风格指南**:
            - 强调回答必须基于背景知识和提供的上下文，保证严谨性。
            - 要求回答自然流畅。
            - 根据问题类型（基础信息、统计计算、开放性问题）提供不同的回答侧重点和风格指导。
            - 对开放性问题，要求先总结事实，再**补充**通用启发性建议（与事实部分区分）。
            - 指示无需注明来源文件名。
            - 规定无法回答时的标准回复。
        - **检索到的上下文**: 使用 `_format_context` 方法格式化检索结果。
        - **用户问题**: 清晰地呈现原始用户问题。
- **答案生成**:
    - 使用 `google-generativeai` 库将构建好的 Prompt 发送给 Gemini 2.5 Pro 模型 (`model.generate_content`)。
    - 配置适当的安全设置。
    - 处理 API 响应，提取生成的文本答案。
    - 处理潜在错误（如 API 调用失败、内容被阻止、空响应）。
- **交互逻辑**: 提供简单的命令行循环，接收用户输入，调用 `answer_question` 处理并打印结果。

```python
# 核心类与方法示例
class QASystem:
    def __init__(self):
        # 初始化 HybridVectorStore 和 Gemini 模型
        pass
    def _format_context(self, context):
        # 格式化检索到的上下文
        pass
    def answer_question(self, user_question, search_top_k=10):
        # 检索上下文 -> 构建 Prompt (含背景知识和风格指南) -> 调用 Gemini -> 返回答案
        pass
```

#### 3.3 统计分析与数据更新模块

*(这些模块在当前实现中未作为独立功能开发，统计分析能力依赖于 LLM 基于检索上下文的推理，数据更新需要重新运行 `vector_store.py` 的构建流程)*

### 4. 实现流程 (当前方案)

1.  **环境准备**: 安装 `requirements.txt` 中的所有依赖。设置 `GEMINI_API_KEY` 环境变量。
2.  **数据准备**: 将所有竞赛 PDF 文件放入 `data/raw/` 目录。
3.  **索引构建**:
    - 运行 `python src/vector_store.py` （确保脚本内 `MODE = 'build'`）。
    - 该脚本会自动处理 `data/raw/` 下的 PDF，进行分块、生成 Gemini 嵌入、构建并保存 FAISS 和 BM25 索引及元数据到 `data/embeddings/`。
4.  **问答系统运行**:
    - 运行 `python src/qa_system.py`。
    - 脚本会加载构建好的索引。
    - 在命令行中输入问题进行交互。
5.  **系统测试与优化**:
    - 使用不同类型的问题测试回答的准确性、相关性和流畅性。
    - **可调参数**:
        - `vector_store.py`: `DEFAULT_CHUNK_SIZE`, `DEFAULT_CHUNK_OVERLAP` (影响检索粒度), `HYBRID_ALPHA` (控制语义与关键词权重)。
        - `qa_system.py`: `search_top_k` (提供给 LLM 的上下文数量), Prompt 内容 (角色、指南、背景知识)。
    - (可选) 替换 `simple_tokenizer` 为 `jieba` 等专业分词库。

### 5. 项目文件结构

```
project/
├── data/
│   ├── raw/               # 原始PDF文件 (输入)
│   └── embeddings/        # 存储 FAISS 索引, BM25 索引, 块元数据 (.index, .pkl)
├── src/
│   ├── vector_store.py    # 混合向量数据库构建与查询 (集成PDF处理)
│   ├── qa_system.py       # 问答处理与 LLM 调用
│   └── ...                # (其他辅助或未来模块)
├── requirements.txt       # 依赖包
└── README.md              # 项目说明 (本文档)
```

### 6. 关键依赖包 (示例)

```
google-generativeai>=0.5.0 # 或更高版本
faiss-cpu>=1.7.4
rank_bm25>=0.2.2
pymupdf>=1.23.0 # Fitz
numpy>=1.24.0
tqdm>=4.66.0
# pandas (如果用于数据分析或处理表格)
# python-dotenv (如果用于加载环境变量)
# requests (如果需要手动调用API)
# jieba (推荐用于中文分词)
```
*(请根据实际 `requirements.txt` 调整)*

### 7. 项目亮点与总结

- **端到端 RAG 实现**: 构建了从 PDF 处理到混合检索再到 LLM 生成的完整问答流程。
- **混合搜索**: 结合 FAISS 的语义理解能力和 BM25 的关键词匹配精度，通过加权分数融合提升检索效果，尤其对于包含特定术语的查询。
- **集成数据处理**: `vector_store.py` 脚本自动化了 PDF 读取、分块、嵌入和索引构建过程。
- **先进 LLM 应用**: 使用了 Google Gemini 2.5 Pro 模型进行答案生成。
- **精细化 Prompt 工程**: 设计了包含背景知识、角色扮演、行为准则和风格指南的复杂 Prompt，以引导 LLM 生成高质量、符合场景需求的回答。
- **模块化设计**: 代码结构清晰，`vector_store.py` 和 `qa_system.py` 分别负责不同的核心功能。

## 后续提升方向

- **集成专业中文分词**: 使用 `jieba` 或其他库替换 `simple_tokenizer`，提升 BM25 效果。
- **优化分块策略**: 尝试基于句子、段落或语义的分块方法。
- **表格信息提取**: 专门处理 PDF 中的表格数据，可能需要结构化存储或特殊处理。
- **对话历史管理**: 实现多轮对话能力，理解对话上下文。
- **答案来源引用**: 在回答中指明关键信息来源于哪些原始文档或段落（需要更复杂的元数据跟踪）。
- **用户界面**: 构建 Web 或 GUI 界面，提升用户体验。
- **评估体系**: 建立量化评估指标（如 RAGAS、BLEU、ROUGE 或人工评估）来衡量系统性能。
- **自动化更新**: 实现检测 `data/raw/` 目录变化并自动触发索引更新的机制。
