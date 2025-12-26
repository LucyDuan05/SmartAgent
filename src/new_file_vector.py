from typing import List, Tuple, Dict, Any
import faiss
import numpy as np
# from sentence_transformers import SentenceTransformer # 移除 SentenceTransformer
import google.generativeai as genai # 导入 Gemini
import pickle
import os
# import time # 添加 time 用于处理可能的 API 速率限制
import glob # 用于查找文件
import fitz # PyMuPDF
from tqdm import tqdm # 用于显示进度条
from typing import List, Tuple, Dict, Any
import re # 用于关键词提取
from rank_bm25 import BM25Okapi 
from vector_store import process_pdfs_in_directory, HybridVectorStore # 导入自定义函数


# Gemini 模型名称
GEMINI_EMBEDDING_MODEL = "models/embedding-001" # 使用推荐的标准模型 "models/embedding-001"
# 存储索引和文本数据的文件路径
DEFAULT_INDEX_PATH = "D:/SmartAgent/SmartAgent/data1/embeddings/vector_store.index"
DEFAULT_METADATA_PATH = "D:/SmartAgent/SmartAgent/data1/embeddings/chunks_with_metadata.pkl" # 修改路径以反映内容
DEFAULT_BM25_PATH = "D:/SmartAgent/SmartAgent/data1/embeddings/bm25_index.pkl" # 新增 BM25 索引路径
# PDF 文件目录
DEFAULT_PDF_DIR = "D:/SmartAgent/SmartAgent/data/raw/"
# 文本分块参数
DEFAULT_CHUNK_SIZE = 1500   # 减小块大小
DEFAULT_CHUNK_OVERLAP = 300 # 适当增加重叠比例
# **优化：语义搜索参数**
INITIAL_SEARCH_K = 20 # 初始从 FAISS 检索的候选数量
FINAL_SEARCH_K = 5    # 经过重排后最终返回给用户的数量
HYBRID_ALPHA = 0.2    # 加权融合中 FAISS 相似度的权重 (0 到 1)

# !! 安全警告: 不建议在代码中硬编码 API 密钥 !!
# !! 请考虑使用环境变量或配置文件 !!
GEMINI_API_KEY = "AIzaSyCE5MzNYr2EK65NlwAHorD2saD-lb1JXmc" # 使用你提供的 API Key


if __name__ == '__main__':
    # 初始化向量库
    vector_store = HybridVectorStore()
    
    # 定义新旧数据路径
    old_data_dir = "D:\\SmartAgent\\SmartAgent\\data\\raw"       # 初始数据目录
    new_data_dir = "D:\\SmartAgent\\SmartAgent\\data\\new"       # 新增数据目录
    
    # 首次构建全量索引
    if not os.path.exists(DEFAULT_INDEX_PATH):
        chunks, tokenized = process_pdfs_in_directory(
            old_data_dir, 
            DEFAULT_CHUNK_SIZE, 
            DEFAULT_CHUNK_OVERLAP
        )
        embeddings = vector_store.embed_chunks(chunks)
        vector_store.build_indices(chunks, tokenized, embeddings)
    
    # 执行增量更新
    vector_store.incremental_update(new_data_dir)