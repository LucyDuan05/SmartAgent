
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
# AIzaSyDbTNm6B9JM4FxtCC8bkQybzCoJ1Oq-c94
# --- PDF 处理和文本分块 ---

def extract_text_from_pdf(pdf_path: str) -> str:
    """从单个PDF文件中提取所有页面的文本。"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"错误：处理PDF文件 '{pdf_path}' 时出错: {e}")
        return ""

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """将文本分割成指定大小并带有重叠的块。"""
    if not text:
        return []
    
    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = min(start_index + chunk_size, len(text))
        chunks.append(text[start_index:end_index])
        
        # 移动到下一个块的起始位置，考虑重叠
        # 如果没有重叠，下一个起始点是 end_index
        # 如果有重叠，下一个起始点是 end_index - chunk_overlap
        # 确保 start_index 至少移动一个字符，防止无限循环
        next_start = start_index + chunk_size - chunk_overlap
        if next_start <= start_index: 
            next_start = start_index + 1 # 强制前进
            
        start_index = next_start
        
        # 如果最后一块太小，可以考虑合并或丢弃，这里我们保留它
        
    return chunks

def simple_tokenizer(text: str) -> List[str]:
    """
    简单的分词器：按字符分割中文，或按空格分割英文/数字。
    注意：对于中文，使用 jieba 等专业分词库效果会好得多。
    """
    # 简单的中文处理：按字符分割
    # 简单的英文/数字处理：小写化，按非字母数字分割
    text = text.lower()
    tokens = re.findall(r'[一-\u9fff]|[a-zA-Z0-9]+', text) # 匹配中文字符或连续的字母数字
    return tokens if tokens else list(text) # 如果没有匹配到，则按单个字符分割

def process_pdfs_in_directory(pdf_dir: str, chunk_size: int, chunk_overlap: int) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[List[str]]]:
    """处理PDF，返回带元数据的块和用于BM25的分词后文本块列表。"""
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    all_chunks_with_metadata = []
    tokenized_corpus = [] # 用于 BM25 的分词后语料库
    
    if not pdf_files:
        print(f"警告：目录 '{pdf_dir}' 中未找到 PDF 文件。")
        return [], []

    print(f"在 '{pdf_dir}' 中找到 {len(pdf_files)} 个 PDF 文件。开始处理...")
    
    for pdf_path in tqdm(pdf_files, desc="处理 PDF 文件"):
        filename = os.path.basename(pdf_path)
        print(f"\n正在处理: {filename}")
        
        # 1. 提取文本
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text:
            print(f"警告：未能从 '{filename}' 提取文本，跳过。")
            continue
            
        # 2. 文本分块
        text_chunks = chunk_text(pdf_text, chunk_size, chunk_overlap)
        print(f"  - 提取并分割成 {len(text_chunks)} 个文本块。")
        
        # 3. 添加元数据
        for i, chunk in enumerate(text_chunks):
            metadata = {
                "source": filename,
                "chunk_index": i + 1, # 块的序号（从1开始）
                # 可以添加更多元数据，如页码（如果分块逻辑更复杂）
            }
            all_chunks_with_metadata.append((chunk, metadata))
            # 为 BM25 分词
            tokenized_corpus.append(simple_tokenizer(chunk))
            
    print(f"\n所有 PDF 处理完成。总共生成 {len(all_chunks_with_metadata)} 个文本块。")
    return all_chunks_with_metadata, tokenized_corpus

# --- 向量存储与混合搜索类 ---

class HybridVectorStore:
    """
    管理PDF处理、文本向量化、FAISS索引、BM25索引和使用加权分数融合的混合搜索。
    """
    def __init__(self):
        """
        初始化HybridVectorStore并配置Gemini客户端。
        """
        print(f"正在配置 Google Gemini 客户端...")
        try:
            genai.configure(api_key=GEMINI_API_KEY, transport='rest')
            # 创建一个临时的 Gemini 模型实例以检查 API 密钥是否有效
            # 注意：这里仅用于配置检查，实际嵌入在 embed_chunks 中进行
            _ = genai.GenerativeModel(GEMINI_EMBEDDING_MODEL) # 尝试访问模型以验证配置
            print("Gemini 客户端配置成功。")
        except Exception as e:
            print(f"错误：配置 Gemini 客户端失败。请检查 API 密钥是否有效。错误信息: {e}")
            # 可以在此处引发异常或设置一个标志指示初始化失败
            raise ConnectionError("无法初始化 Gemini 客户端。") from e

        self.faiss_index = None
        self.bm25_index = None
        self.chunks_with_metadata: List[Tuple[str, Dict[str, Any]]] = [] # 存储块和元数据
        # self.model 不再是 SentenceTransformer 实例，嵌入直接通过 genai API 调用
        self.tokenized_corpus = []  # 新增属性存储语料库
        self.chunks_with_metadata = []  # 新增元数据存储

    def embed_chunks(self, chunks_with_metadata: List[Tuple[str, Dict[str, Any]]]) -> np.ndarray:
        """
        将文本块列表通过 Gemini API 转换为向量嵌入。

        Args:
            chunks_with_metadata (List[Tuple[str, Dict[str, Any]]]): 包含文本块和元数据的列表。

        Returns:
            np.ndarray: 文本块的向量嵌入矩阵。
        """
        text_chunks = [chunk for chunk, metadata in chunks_with_metadata] # 仅提取文本部分进行嵌入
        
        if not text_chunks:
            print("错误：没有文本块需要嵌入。")
            return np.array([])

        print(f"正在为 {len(text_chunks)} 个文本块生成 Gemini 嵌入...")
        all_embeddings = []
        try:
            # Gemini embed_content API 通常可以处理批量输入，但也有速率限制
            # 这里我们分批处理，以避免潜在的请求大小或速率限制问题
            # 可以根据需要调整 batch_size
            batch_size = 100 # Gemini embed_content 一次最多处理 100 个内容
            for i in tqdm(range(0, len(text_chunks), batch_size), desc="生成嵌入"):
                batch = text_chunks[i:i+batch_size]
                print(f"  处理批次 {i//batch_size + 1}/{(len(text_chunks) + batch_size - 1)//batch_size} ({len(batch)} 个文本块)...")
                # 重要：使用 task_type="RETRIEVAL_DOCUMENT" 进行文档嵌入以用于后续搜索
                result = genai.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    content=batch,
                    task_type="RETRIEVAL_DOCUMENT" # 用于文档嵌入
                )
                all_embeddings.extend(result['embedding'])
                # 可以考虑在批次之间添加短暂休眠以避免速率限制
                # time.sleep(1) 

            print("向量嵌入生成完成。")
            return np.array(all_embeddings).astype('float32') # 确保是 float32

        except Exception as e:
            print(f"错误：使用 Gemini API 生成嵌入时出错: {e}")
            # 返回一个空的 numpy 数组或根据需要处理错误
            return np.array([])

    def embed_query(self, query: str) -> np.ndarray:
        """
        将单个查询字符串通过 Gemini API 转换为向量嵌入。
        """
        try:
            # 重要：查询使用 task_type="RETRIEVAL_QUERY"
            result = genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=query,
                task_type="RETRIEVAL_QUERY" # 用于查询嵌入
            )
            return np.array([result['embedding']]).astype('float32') # 返回二维数组
        except Exception as e:
            print(f"错误：使用 Gemini API 生成查询嵌入时出错: {e}")
            return np.array([])

    def build_indices(self,
                      chunks_with_metadata: List[Tuple[str, Dict[str, Any]]],
                      tokenized_corpus: List[List[str]], # 新增：分词后的语料
                      embeddings: np.ndarray,
                      index_path: str = DEFAULT_INDEX_PATH,
                      metadata_path: str = DEFAULT_METADATA_PATH,
                      bm25_path: str = DEFAULT_BM25_PATH): # 新增：BM25 路径
        """构建 FAISS 和 BM25 索引并保存。"""
        # --- 构建 FAISS ---
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
             print(f"错误：嵌入向量格式不正确 ({embeddings.shape}) 或为空，无法构建索引。")
             return

        if len(chunks_with_metadata) != embeddings.shape[0]:
            print(f"错误：文本块数量 ({len(chunks_with_metadata)}) 与嵌入向量数量 ({embeddings.shape[0]}) 不匹配。")
            return

        dimension = embeddings.shape[1]
        print(f"正在构建维度为 {dimension} 的FAISS索引...")
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings) # 确认 embeddings 是 float32
        self.chunks_with_metadata = chunks_with_metadata # 保存块和元数据
        print(f"FAISS索引构建完成，包含 {self.faiss_index.ntotal} 个向量。")

        # --- 构建 BM25 ---
        if len(chunks_with_metadata) != len(tokenized_corpus):
            print(f"错误：文本块数量与分词后语料数量不匹配。")
            return
        print(f"正在构建 BM25 索引...")
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print(f"BM25索引构建完成。")

        # --- 保存 ---
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
        
        print(f"正在将 FAISS 索引保存到 {index_path}...")
        faiss.write_index(self.faiss_index, index_path)
        print(f"正在将文本块及元数据保存到 {metadata_path}...")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunks_with_metadata, f)
        print(f"正在将 BM25 索引保存到 {bm25_path}...")
        with open(bm25_path, 'wb') as f:
            pickle.dump(self.bm25_index, f) # 保存 BM25 对象
        print("所有索引和数据已成功保存。")

        with open(bm25_path, 'wb') as f:
            pickle.dump({
                "model": self.bm25_index,
                "corpus": tokenized_corpus  # 显式保存语料库
            }, f)

    def load_indices(self,
                     index_path: str = DEFAULT_INDEX_PATH,
                     metadata_path: str = DEFAULT_METADATA_PATH,
                     bm25_path: str = DEFAULT_BM25_PATH) -> bool:
        """加载 FAISS 索引、元数据和 BM25 索引。"""
        if not all(os.path.exists(p) for p in [index_path, metadata_path, bm25_path]):
            print(f"错误：找不到必需的索引或数据文件 ({index_path}, {metadata_path}, {bm25_path})。")
            return False
        try:
            print(f"正在从 {index_path} 加载 FAISS 索引...")
            self.faiss_index = faiss.read_index(index_path)
            print(f"FAISS索引加载完成 ({self.faiss_index.ntotal} 向量)。")

            print(f"正在从 {metadata_path} 加载文本块及元数据...")
            with open(metadata_path, 'rb') as f:
                self.chunks_with_metadata = pickle.load(f)
            print(f"文本块及元数据加载完成 ({len(self.chunks_with_metadata)} 个)。")

            print(f"正在从 {bm25_path} 加载 BM25 索引...")
            with open(bm25_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
            print(f"BM25索引加载完成。")

            if not (self.faiss_index.ntotal == len(self.chunks_with_metadata)):
                 print("警告：FAISS索引大小与元数据数量不匹配！")
            # BM25 索引本身不直接存储文档数量，但依赖的语料库应与元数据匹配
            if os.path.exists(bm25_path):
                with open(bm25_path, 'rb') as f:
                    bm25_data = pickle.load(f)
                    self.bm25_index = bm25_data["model"]
                    self.tokenized_corpus = bm25_data["corpus"]  # 新增属性
            return True
        except Exception as e:
            print(f"加载索引或数据时出错: {e}")
            self.faiss_index = None; self.bm25_index = None; self.chunks_with_metadata = []
            return False
        
        

    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """对字典中的分数进行 Min-Max 归一化到 [0, 1] 区间。"""
        if not scores:
            return {}
        
        min_score = min(scores.values())
        max_score = max(scores.values())
        
        # 防止除零错误
        if max_score == min_score:
            # 如果所有分数都相同，可以将它们归一化为 0.5 或 1
            return {idx: 0.5 for idx in scores}
            
        normalized_scores = {
            idx: (score - min_score) / (max_score - min_score)
            for idx, score in scores.items()
        }
        return normalized_scores

    def hybrid_search(self, 
                      query: str, 
                      top_k: int = FINAL_SEARCH_K, 
                      initial_k: int = INITIAL_SEARCH_K, 
                      alpha: float = HYBRID_ALPHA
                     ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        执行混合搜索（FAISS + BM25），使用加权分数融合。

        Args:
            query (str): 用户查询字符串。
            top_k (int): 最终返回的结果数量。
            initial_k (int): FAISS 和 BM25 各自检索的初始候选数量。
            alpha (float): FAISS 相似度得分的权重 (0 到 1)。

        Returns:
            List[Tuple[str, Dict[str, Any], float]]: 包含(文本块, 元数据, 融合分数)的列表，按融合分数降序排列。
        """
        if not all([self.faiss_index, self.bm25_index, self.chunks_with_metadata]):
            print("错误：FAISS索引、BM25索引或文本块未加载。")
            return []

        # --- 1. 语义搜索 (FAISS) ---
        print(f"正在为查询生成 Gemini 嵌入: '{query}'")
        query_embedding = self.embed_query(query)
        if query_embedding.size == 0:
            print("错误：无法为查询生成向量嵌入。")
            return []
        
        print(f"正在执行 FAISS 搜索 (k={initial_k})...")
        distances, faiss_indices = self.faiss_index.search(query_embedding, initial_k)
        
        # 收集 FAISS 结果 (doc_index -> distance)
        faiss_results_map = {}
        if faiss_indices.size > 0:
             for i in range(len(faiss_indices[0])):
                 idx = faiss_indices[0][i]
                 if idx != -1:
                      faiss_results_map[idx] = distances[0][i]
        print(f"FAISS 找到 {len(faiss_results_map)} 个候选。")

        # --- 2. 关键词搜索 (BM25) ---
        print(f"正在执行 BM25 搜索 (k={initial_k})...")
        tokenized_query = simple_tokenizer(query)
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(doc_scores)[::-1][:initial_k] 
        
        # 收集 BM25 结果 (doc_index -> bm25_score)
        bm25_results_map = {idx: doc_scores[idx] for idx in bm25_top_indices if doc_scores[idx] > 0}
        print(f"BM25 找到 {len(bm25_results_map)} 个候选。")

        # --- 3. 融合分数计算 ---
        print("正在计算融合分数...")
        
        # 获取所有候选文档的索引
        all_indices = set(faiss_results_map.keys()) | set(bm25_results_map.keys())
        
        # 准备用于归一化的分数列表
        faiss_similarities = {}
        bm25_scores_for_norm = {}

        # 默认值处理：如果某个系统未找到文档，给予一个低分/高距离
        default_bm25_score = 0.0 
        # 找到 FAISS 结果中的最大距离，用于给未找到的文档设定一个"更差"的距离
        max_faiss_dist = max(faiss_results_map.values()) if faiss_results_map else 1.0 
        default_faiss_dist = max_faiss_dist * 1.1 # 比最差的还差一点

        for idx in all_indices:
            # 计算 FAISS 相似度
            faiss_dist = faiss_results_map.get(idx, default_faiss_dist)
            # 将距离转换为相似度，确保非负且距离0时相似度最高
            faiss_sim = 1.0 / (1.0 + faiss_dist) 
            faiss_similarities[idx] = faiss_sim
            
            # 获取 BM25 分数
            bm25_score = bm25_results_map.get(idx, default_bm25_score)
            bm25_scores_for_norm[idx] = bm25_score

        # 归一化 FAISS 相似度和 BM25 分数
        norm_faiss_sim = self._normalize_scores(faiss_similarities)
        norm_bm25_score = self._normalize_scores(bm25_scores_for_norm)

        # 计算最终加权分数
        final_scores = {}
        for idx in all_indices:
            final_scores[idx] = alpha * norm_faiss_sim.get(idx, 0) + \
                                (1 - alpha) * norm_bm25_score.get(idx, 0)

        # --- 4. 排序并返回 Top K ---
        sorted_doc_indices = sorted(final_scores.keys(), key=lambda idx: final_scores[idx], reverse=True)
        
        final_results = []
        print(f"融合后排序，提取 Top {top_k} 结果...")
        for idx in sorted_doc_indices[:top_k]:
             if 0 <= idx < len(self.chunks_with_metadata):
                 chunk, metadata = self.chunks_with_metadata[idx]
                 final_results.append((chunk, metadata, final_scores[idx])) # 返回块、元数据和融合分数
        
        print(f"混合搜索完成，返回 {len(final_results)} 个结果。")
        return final_results
    

    # --- 增量更新索引 ---

    def incremental_update(self,
                          new_pdf_dir: str,
                          chunk_size: int = DEFAULT_CHUNK_SIZE,
                          chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                          index_path: str = DEFAULT_INDEX_PATH,
                          metadata_path: str = DEFAULT_METADATA_PATH,
                          bm25_path: str = DEFAULT_BM25_PATH):
        """
        增量更新索引：添加新 PDF 文件到现有向量库
        
        参数:
            new_pdf_dir: 新增 PDF 文件目录
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
        """
        # --- 1. 加载现有数据 ---
        # 加载现有数据
        existing_chunks = self.chunks_with_metadata.copy()
        existing_tokenized = self.tokenized_corpus.copy()
        existing_embeddings = None
        
        # 处理新 PDF 文件
        new_chunks, new_tokenized = process_pdfs_in_directory(
            new_pdf_dir, chunk_size, chunk_overlap
        )
        if not new_chunks:
            print("警告：未找到有效的新增文本块")
            return
        
        # --- 合并元数据 ---
        all_chunks = existing_chunks + new_chunks
        all_tokenized = existing_tokenized + new_tokenized
        
        # --- 更新 FAISS 索引 ---
        # 生成新嵌入
        new_embeddings = self.embed_chunks(new_chunks)
        if new_embeddings.size == 0:
            print("错误：新嵌入生成失败")
            return
        
        # 合并旧嵌入（需重新加载）
        if self.faiss_index and self.faiss_index.ntotal > 0:
            existing_embeddings = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)
            all_embeddings = np.vstack([existing_embeddings, new_embeddings])
        else:
            all_embeddings = new_embeddings

        # 重建 FAISS 索引
        dimension = all_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(all_embeddings)  # 全量添加确保一致性

        # --- 重建 BM25 索引 ---
        self.bm25_index = BM25Okapi(all_tokenized)
        
        # 更新内存中的元数据
        self.chunks_with_metadata = all_chunks
        self.tokenized_corpus = all_tokenized

        # 保存完整索引
        self._save_indices(all_chunks, all_embeddings, bm25_path)

    def _save_indices(self, chunks):
        """保存合并后的全量索引"""
        # 保存 FAISS
        faiss.write_index(self.faiss_index, DEFAULT_INDEX_PATH)
        
        # 保存元数据
        with open(DEFAULT_METADATA_PATH, 'wb') as f:
            pickle.dump(chunks, f)
        
        # 保存 BM25（含语料库）
        with open(DEFAULT_BM25_PATH, 'wb') as f:
            pickle.dump({
                "model": self.bm25_index,
                "corpus": self.tokenized_corpus
            }, f)

    def _save_indices(self, chunks, metadata_path, bm25_path):
        # 保存元数据
        with open(metadata_path, 'wb') as f:
            pickle.dump(chunks, f)
        
        # 保存 BM25（含语料库）
        with open(bm25_path, 'wb') as f:
            pickle.dump({
                "model": self.bm25_index,
                "corpus": self.tokenized_corpus  # 包含完整语料库
            }, f)

# === 主执行逻辑 ===
if __name__ == '__main__':
    
    # --- 模式选择： 'build' 或 'search' ---
    # 你可以修改这里来决定是构建索引还是仅执行搜索
    MODE = 'build' # 或者 'search'

    try:
        if MODE == 'build':
            # --- 1. 处理PDF并分块 ---
            print("[模式: 构建索引]")
            chunks_meta, tokenized_corpus = process_pdfs_in_directory(
                DEFAULT_PDF_DIR, 
                DEFAULT_CHUNK_SIZE, 
                DEFAULT_CHUNK_OVERLAP
            )
            
            if not chunks_meta:
                print("未能生成文本块，无法继续构建索引。")
            else:
                # --- 2. 初始化 VectorStore (配置 Gemini) ---
                vector_store_builder = HybridVectorStore()

                # --- 3. 生成嵌入 ---
                embeddings = vector_store_builder.embed_chunks(chunks_meta)

                # --- 4. 构建并保存索引 ---
                if embeddings.size > 0:
                    vector_store_builder.build_indices(chunks_meta, tokenized_corpus, embeddings)
                    print("\n[构建模式] 所有索引构建完成并已保存。\n")
                else:
                    print("\n[构建模式] 嵌入向量生成失败，无法构建索引。")

        elif MODE == 'search':
             # --- 仅加载并搜索 ---
            print("[模式: 搜索]")
            vector_store_searcher = HybridVectorStore()
            
            # a) 加载索引和数据
            load_successful = vector_store_searcher.load_indices()
            
            if load_successful:
                # b) 执行示例搜索
                query1 = "第七届全国青少年人工智能创新挑战赛的报名时间"
                results1 = vector_store_searcher.hybrid_search(query1, top_k=FINAL_SEARCH_K, alpha=HYBRID_ALPHA)
                print(f"\n查询: '{query1}'")
                print(f"结果 (Top {FINAL_SEARCH_K} after Weighted Hybrid Search, alpha={HYBRID_ALPHA}):")
                if results1:
                    for i, (chunk, meta, score) in enumerate(results1):
                        # 打印块内容时检查关键词是否存在
                        kw_present = "报名" in chunk.lower() or "时间" in chunk.lower()
                        print(f"{i+1}. [Weighted Score: {score:.4f}] [来源: {meta.get('source', '未知')}] [含关键词: {kw_present}] {chunk[:200]}...") 
                else:
                    print("未找到相关结果。")

                query2 = "数据分析比赛有哪些？"
                results2 = vector_store_searcher.hybrid_search(query2, top_k=FINAL_SEARCH_K, alpha=HYBRID_ALPHA)
                print(f"\n查询: '{query2}'")
                print(f"结果 (Top {FINAL_SEARCH_K} after Weighted Hybrid Search, alpha={HYBRID_ALPHA}):")
                if results2:
                    for i, (chunk, meta, score) in enumerate(results2):
                         print(f"{i+1}. [Weighted Score: {score:.4f}] [来源: {meta.get('source', '未知')}] {chunk[:200]}...")
                else:
                    print("未找到相关结果。")
            else:
                print("[搜索模式] 加载索引失败，无法执行搜索。")
        else:
             print(f"错误：未知的模式 '{MODE}'。请选择 'build' 或 'search'。")

    except ConnectionError as e:
         print(f"初始化或连接失败: {e}")
    except Exception as e:
        print(f"运行时发生意外错误: {e}")
        import traceback
        traceback.print_exc() # 打印详细的错误堆栈

    print(f"\n[{MODE} 模式] 脚本执行完毕。") 