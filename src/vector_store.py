import faiss
import numpy as np
import pickle
import os
import glob
import fitz
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import re
from rank_bm25 import BM25Okapi
import jieba
import pandas as pd
from sentence_transformers import SentenceTransformer
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 使用本地 SentenceTransformer 模型（第一次运行会自动下载）
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"

DEFAULT_INDEX_PATH = os.path.join(BASE_DIR, "data1/embeddings/vector_store.index")
DEFAULT_METADATA_PATH = os.path.join(BASE_DIR, "data1/embeddings/chunks_with_metadata.pkl")
DEFAULT_BM25_PATH = os.path.join(BASE_DIR, "data1/embeddings/bm25_index.pkl")
DEFAULT_PDF_DIR = os.path.join(BASE_DIR, "data/raw/")
DEFAULT_EXCEL_PATH = os.path.join(BASE_DIR, "result1.xlsx")

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
INITIAL_SEARCH_K = 20
FINAL_SEARCH_K = 5

def format_excel_metadata(filename: str, df: pd.DataFrame) -> str:
    """根据 Excel 匹配文件名，生成元数据前缀"""
    if df is None or df.empty:
        return ""
    
    clean_filename = re.sub(r'[^\w\u4e00-\u9fff]+', '', filename)
    best_match = None
    best_score = 0
    
    for idx, row in df.iterrows():
        track_name = str(row.get('赛道', ''))
        clean_track = re.sub(r'[^\w\u4e00-\u9fff]+', '', track_name)
        
        # 精确包含匹配
        if clean_track and (clean_track in clean_filename or clean_filename in clean_track):
            best_match = row
            break
            
        # 字符重合度匹配 (作为后备方案)
        score = len(set(clean_filename) & set(clean_track))
        if score > best_score:
            best_score = score
            best_match = row
            
    if best_match is not None and best_score > 5:  # 设置一个最小匹配阈值
        components = []
        if pd.notna(best_match.get('赛项名称')): components.append(f"赛项：{str(best_match['赛项名称']).strip()}")
        if pd.notna(best_match.get('赛道')): components.append(f"赛道：{str(best_match['赛道']).strip()}")
        if pd.notna(best_match.get('报名时间')): components.append(f"报名时间：{str(best_match['报名时间']).strip()}")
        
        if components:
            return f"[{' | '.join(components)}] "
    return ""

def extract_text_from_pdf(pdf_path: str) -> str:
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
    if not text:
        return []
    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = min(start_index + chunk_size, len(text))
        chunks.append(text[start_index:end_index])
        
        next_start = start_index + chunk_size - chunk_overlap
        if next_start <= start_index: 
            next_start = start_index + 1
            
        start_index = next_start
    return chunks

def jieba_tokenizer(text: str) -> List[str]:
    text = text.lower()
    tokens = jieba.lcut(text, cut_all=False)
    tokens = [t.strip() for t in tokens if t.strip() and not re.match(r'^[^\w\u4e00-\u9fff]+$', t.strip())]
    return tokens

def process_pdfs_in_directory(pdf_dir: str, excel_path: str, chunk_size: int, chunk_overlap: int) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[List[str]]]:
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    all_chunks_with_metadata = []
    tokenized_corpus = []
    
    if not pdf_files:
        print(f"警告：目录 '{pdf_dir}' 中未找到 PDF 文件。")
        return [], []

    # 1. 尝试加载 Excel 元数据
    try:
        df = pd.read_excel(excel_path)
        print(f"成功加载 Excel 元数据，共 {len(df)} 条记录")
    except Exception as e:
        print(f"无法加载 Excel 文件 {excel_path}, 继续处理无元数据附加。错误: {e}")
        df = pd.DataFrame()

    print(f"在 '{pdf_dir}' 中找到 {len(pdf_files)} 个 PDF 文件。开始处理...")
    
    for pdf_path in tqdm(pdf_files, desc="处理 PDF 文件"):
        filename = os.path.basename(pdf_path)
        print(f"\n正在处理: {filename}")
        
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text:
            continue
            
        text_chunks = chunk_text(pdf_text, chunk_size, chunk_overlap)
        
        # 2. 为该文件生成 Excel 前缀
        prefix_meta = format_excel_metadata(filename, df)
        if prefix_meta:
             print(f"   √ 匹配到项目信息: {prefix_meta}")
        
        for i, chunk in enumerate(text_chunks):
            # 将元数据拼接在内容前，实现元数据增强检索 Metadata Filtering
            enhanced_chunk = f"{prefix_meta}{chunk}"
            
            metadata = {
                "source": filename,
                "chunk_index": i + 1,
            }
            all_chunks_with_metadata.append((enhanced_chunk, metadata))
            tokenized_corpus.append(jieba_tokenizer(enhanced_chunk))
            
    return all_chunks_with_metadata, tokenized_corpus

class HybridVectorStore:
    def __init__(self):
        try:
            print(f"正在加载 SentenceTransformer 本地模型 ({EMBEDDING_MODEL_NAME})...")
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as e:
            raise RuntimeError(f"无法初始化 SentenceTransformer 模型: {e}")

        self.faiss_index = None
        self.bm25_index = None
        self.chunks_with_metadata: List[Tuple[str, Dict[str, Any]]] = []
        self.tokenized_corpus = []

    def embed_chunks(self, chunks_with_metadata: List[Tuple[str, Dict[str, Any]]]) -> np.ndarray:
        text_chunks = [chunk for chunk, metadata in chunks_with_metadata] 
        if not text_chunks:
            return np.array([])

        print(f"正在为 {len(text_chunks)} 个文本块生成嵌入...")
        # 批量处理，显示进度条
        embeddings = self.model.encode(text_chunks, show_progress_bar=True, normalize_embeddings=True)
        return np.array(embeddings).astype('float32')

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode([query], normalize_embeddings=True)
        return np.array(embedding).astype('float32')

    def build_indices(self, chunks_with_metadata, tokenized_corpus, embeddings, index_path=DEFAULT_INDEX_PATH, metadata_path=DEFAULT_METADATA_PATH, bm25_path=DEFAULT_BM25_PATH):
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
             return
        dimension = embeddings.shape[1]
        
        # 使用 Inner Product 余弦相似度
        self.faiss_index = faiss.IndexFlatIP(dimension) 
        self.faiss_index.add(embeddings)
        self.chunks_with_metadata = chunks_with_metadata
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.tokenized_corpus = tokenized_corpus

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
        
        faiss.write_index(self.faiss_index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunks_with_metadata, f)
        with open(bm25_path, 'wb') as f:
            pickle.dump({
                "model": self.bm25_index,
                "corpus": tokenized_corpus
            }, f)

    def load_indices(self, index_path=DEFAULT_INDEX_PATH, metadata_path=DEFAULT_METADATA_PATH, bm25_path=DEFAULT_BM25_PATH) -> bool:
        if not all(os.path.exists(p) for p in [index_path, metadata_path, bm25_path]):
            return False
        try:
            self.faiss_index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                self.chunks_with_metadata = pickle.load(f)
            with open(bm25_path, 'rb') as f:
                bm25_data = pickle.load(f)
            if isinstance(bm25_data, dict) and "model" in bm25_data:
                self.bm25_index = bm25_data["model"]
                self.tokenized_corpus = bm25_data.get("corpus", [])
            else:
                self.bm25_index = bm25_data
                self.tokenized_corpus = []
            return True
        except Exception as e:
            return False

    def hybrid_search(self, query: str, top_k: int = FINAL_SEARCH_K) -> List[Tuple[str, Dict[str, Any], float]]:
        if not all([self.faiss_index, self.bm25_index, self.chunks_with_metadata]):
            return []

        query_embedding = self.embed_query(query)
        if query_embedding.size == 0:
            return []
        
        distances, faiss_indices = self.faiss_index.search(query_embedding, 60)
        
        faiss_rank = [idx for idx in faiss_indices[0] if idx != -1]
        tokenized_query = jieba_tokenizer(query)
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(doc_scores)[::-1][:60] 
        bm25_rank = [idx for idx in bm25_top_indices if doc_scores[idx] > 0]

        rrf_scores = {}
        for rank, doc_idx in enumerate(faiss_rank):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (60 + rank + 1)
        for rank, doc_idx in enumerate(bm25_rank):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (60 + rank + 1)

        sorted_doc_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)
        
        final_results = []
        for idx in sorted_doc_indices[:top_k]:
             if 0 <= idx < len(self.chunks_with_metadata):
                 chunk, metadata = self.chunks_with_metadata[idx]
                 final_results.append((chunk, metadata, rrf_scores[idx])) 
        
        return final_results

if __name__ == '__main__':
    MODE = 'build'
    try:
        if MODE == 'build':
            print("[模式: 构建本地强化索引]")
            chunks_meta, tokenized_corpus = process_pdfs_in_directory(DEFAULT_PDF_DIR, DEFAULT_EXCEL_PATH, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
            if chunks_meta:
                vector_store_builder = HybridVectorStore()
                embeddings = vector_store_builder.embed_chunks(chunks_meta)
                if embeddings.size > 0:
                    vector_store_builder.build_indices(chunks_meta, tokenized_corpus, embeddings)
                    print("\n[构建模式] 所有索引构建完成并已保存。\n")
        elif MODE == 'search':
            vector_store_searcher = HybridVectorStore()
            if vector_store_searcher.load_indices():
                results1 = vector_store_searcher.hybrid_search("第七届全国青少年人工智能创新挑战赛的报名时间")
                print(f"找到 {len(results1)} 个结果")
                for i, (chunk, meta, score) in enumerate(results1):
                    # 获取前缀（通过查找 ']' 字符）
                    prefix_end = chunk.find(']')
                    if prefix_end != -1:
                         display_text = chunk[:prefix_end+1] + " ... " + chunk[prefix_end+1:prefix_end+50]
                    else:
                         display_text = chunk[:50]
                    print(f"{i+1}. [得分: {score:.4f}] {display_text}...") 
    except Exception as e:
        import traceback
        traceback.print_exc()
