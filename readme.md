# 泰迪杯数据挖掘挑战赛 - C题：竞赛智能客服机器人

## 项目概述

本项目旨在构建一个智能客服机器人，用于回答关于各类竞赛的咨询问题。系统能够处理三种类型的问题：
1. 基本信息查询（如竞赛时间、地点等）
2. 统计分析查询（如特定类型竞赛的数量）
3. 开放性问题（如竞赛准备建议）

此外，系统还具备知识库更新与管理功能，能够处理新增和变更的竞赛信息。

## 技术方案

### 1. 总体架构

```
┌───────────────┐    ┌────────────────┐    ┌──────────────┐
│ PDF文档处理模块 │ => │  向量数据库模块  │ => │  问答处理模块  │
└───────────────┘    └────────────────┘    └──────────────┘
                                                   ↓
┌───────────────┐    ┌────────────────┐    ┌──────────────┐
│ 数据更新管理模块 │ <= │   统计分析模块   │ <= │ 用户交互界面  │
└───────────────┘    └────────────────┘    └──────────────┘
```

### 2. 核心技术栈

- **编程语言**: Python 3.10+
- **大语言模型**: Google Gemini 2 Flash
- **向量数据库**: FAISS/Chroma
- **PDF处理**: PyPDF2, PyMuPDF
- **数据处理**: Pandas, NumPy
- **文本嵌入**: SentenceTransformers

### 3. 模块设计

#### 3.1 PDF文档处理模块

数据整理阶段任务；依据附件 1 中竞赛规程文档，按表 1 格式获取各个竞赛的基本信息，并保存为文件
“result_1.xlsx”。
表 1 竞赛详情表
赛项名称 赛道 发布时
间 报名时间 组织单位 官网

该模块负责从PDF文档中提取文本内容，并进行预处理：

- 使用PyMuPDF(fitz)提取PDF文本
- 文本清洗与规范化
- 文本分块，适合向量化存储
- 表格信息提取与结构化
- 使用正则表达式提取关键信息（如日期、地点等）

```python
# 核心函数示例
def extract_text_from_pdf(pdf_path):
    """从PDF文件提取文本"""
    # 实现代码
    
def extract_structured_info(text):
    """从文本中提取结构化信息"""
    # 实现代码
```

#### 3.2 向量数据库模块

该模块负责文本向量化和存储，为快速检索提供支持：

- 使用SentenceTransformers将文本转换为向量
- 使用FAISS/Chroma建立高效向量索引
- 支持语义相似度搜索
- 支持增量更新

```python
# 核心函数示例
def embed_text(text_chunks):
    """将文本块转换为向量"""
    # 实现代码
    
def build_vector_store(embeddings, texts):
    """构建向量存储"""
    # 实现代码
    
def semantic_search(query, top_k=5):
    """语义搜索功能"""
    # 实现代码
```

#### 3.3 问答处理模块

该模块负责理解用户问题，检索相关信息，并生成回答：

- 问题分类（基本查询/统计分析/开放性）
- 问题向量化
- 相关信息检索
- 使用Gemini 2 Flash生成回答
- 回答验证与优化

```python
# 核心函数示例
def classify_question(question):
    """对问题进行分类"""
    # 实现代码
    
def generate_answer(question, relevant_info):
    """生成回答"""
    # 实现代码
```

#### 3.4 统计分析模块

该模块负责处理需要计算或统计的查询：

- 竞赛分类与计数
- 时间分析（如月度/年度分布）
- 组织单位分析
- 交叉统计功能

```python
# 核心函数示例
def count_competitions_by_type(type):
    """按类型统计竞赛数量"""
    # 实现代码
    
def analyze_time_distribution():
    """分析时间分布"""
    # 实现代码
```

#### 3.5 数据更新管理模块

该模块负责处理知识库的更新：

- 新增PDF文档处理
- 变更信息检测
- 向量数据库增量更新
- 结构化数据更新

```python
# 核心函数示例
def update_knowledge_base(new_pdfs, updated_pdfs):
    """更新知识库"""
    # 实现代码
    
def detect_changes(old_pdf, new_pdf):
    """检测PDF变更"""
    # 实现代码
```

### 4. Gemini 2 Flash模型集成

使用Google的Gemini 2 Flash模型作为核心问答引擎：

- 通过Google AI API集成Gemini模型
- 构建有效的Prompt模板
- 使用RAG(检索增强生成)技术提高回答质量
- 结合向量检索结果生成精确答案

```python
# 核心函数示例
def setup_gemini_client():
    """设置Gemini客户端"""
    # 实现代码
    
def create_prompt(question, context):
    """创建prompt模板"""
    # 实现代码
    
def rag_answer_generation(question, retrieved_docs):
    """RAG回答生成"""
    # 实现代码
```

### 5. 实现流程

1. **数据准备阶段**：
   - 处理18个PDF文档，提取文本
   - 构建结构化数据表
   - 文本分块与向量化
   - 建立向量索引

2. **问答系统构建**：
   - 实现问题分类逻辑
   - 实现检索逻辑
   - 集成Gemini模型
   - 实现回答生成逻辑

3. **更新机制实现**：
   - 设计增量更新流程
   - 实现变更检测
   - 实现数据库更新逻辑

4. **系统测试与优化**：
   - 使用附件2中的问题进行测试
   - 优化检索与生成策略
   - 微调回答质量

### 6. 项目文件结构

```
project/
├── data/
│   ├── raw/               # 原始PDF文件
│   ├── processed/         # 处理后的文本
│   └── embeddings/        # 向量嵌入存储
├── src/
│   ├── pdf_processor.py   # PDF处理模块
│   ├── vector_store.py    # 向量数据库模块
│   ├── qa_system.py       # 问答处理模块
│   ├── statistics.py      # 统计分析模块
│   ├── updater.py         # 更新管理模块
│   └── main.py            # 主程序入口
├── utils/
│   ├── text_utils.py      # 文本处理工具
│   └── evaluation.py      # 评估工具
├── notebooks/             # 开发笔记本
├── results/               # 结果输出
├── tests/                 # 测试代码
├── requirements.txt       # 依赖包
└── README.md              # 项目说明
```

### 7. 关键依赖包

```
python-dotenv>=1.0.0
pandas>=2.0.0
numpy>=1.24.0
pymupdf>=1.22.0
pypdf2>=3.0.0
langchain>=0.1.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
google-generativeai>=0.3.0
```

### 8. 实现步骤与时间规划

1. **第1-2天**: 环境搭建与数据预处理
2. **第3-5天**: 向量数据库构建与基础查询实现
3. **第6-8天**: Gemini模型集成与问答系统开发
4. **第9-10天**: 统计分析功能实现
5. **第11-12天**: 更新管理机制实现
6. **第13-14天**: 系统集成与测试
7. **第15天**: 结果整理与文档完善

## 后续提升方向

- 引入更复杂的文本分块策略，提高检索精度
- 使用混合检索策略(BM25+向量检索)
- 增加知识图谱表示，处理更复杂的关系查询
- 添加对话历史管理，支持多轮交互
- 增强回答解释性，提供信息来源
