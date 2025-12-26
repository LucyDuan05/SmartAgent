
import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import json

# 从 vector_store 模块导入 HybridVectorStore 类
# 确保 vector_store.py 在 Python 的路径中，或者在同一个目录下
try:
    from vector_store import HybridVectorStore
except ImportError:
    print("错误：无法导入 HybridVectorStore。请确保 vector_store.py 在 Python 路径或同级目录中。")
    # 可以选择退出或提供默认实现/占位符
    exit(1) 

# --- 配置 ---
# !! 安全警告: 不建议在代码中硬编码 API 密钥 !!
# !! 请考虑使用环境变量或配置文件 !!
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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") # 优先使用环境变量

# DeepSeek 模型配置
MODEL_NAME = "deepseek/deepseek-v3-base:free" # DeepSeek 模型

# HybridVectorStore 使用的索引路径（应与 vector_store.py 中的配置一致）
# 如果 vector_store.py 中的路径被修改，这里也需要同步修改
DEFAULT_INDEX_PATH = "data/embeddings/vector_store.index"
DEFAULT_METADATA_PATH = "data/embeddings/chunks_with_metadata.pkl"
DEFAULT_BM25_PATH = "data/embeddings/bm25_index.pkl"

class QASystem:
    """
    问答系统类，负责检索上下文并使用 DeepSeek 生成答案。
    """
    def __init__(self, 
                 index_path: str = DEFAULT_INDEX_PATH,
                 metadata_path: str = DEFAULT_METADATA_PATH,
                 bm25_path: str = DEFAULT_BM25_PATH,
                 model_name: str = MODEL_NAME):
        """
        初始化问答系统。

        Args:
            index_path (str): FAISS 索引文件路径。
            metadata_path (str): 元数据文件路径。
            bm25_path (str): BM25 索引文件路径。
            model_name (str): 用于生成答案的 DeepSeek 模型名称。
        """
        print("初始化问答系统...")
        
        # 1. 配置 OpenRouter API
        print(f"配置 OpenRouter API (模型: {model_name})...")
        if not OPENROUTER_API_KEY.startswith("sk-or-v1-"):
             print("错误：无效的 OPENROUTER_API_KEY。请设置正确的 API 密钥环境变量。")
             raise ValueError("无效的 OpenRouter API Key")
        try:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
            self.model_name = model_name
            print("OpenRouter 客户端初始化成功。")
        except Exception as e:
            print(f"错误：初始化 OpenRouter 客户端失败: {e}")
            raise ConnectionError(f"无法初始化 OpenRouter 客户端") from e
            
        # 2. 初始化并加载向量存储
        print("初始化并加载混合向量存储...")
        self.vector_store = HybridVectorStore() # 使用 vector_store.py 中的类
        load_success = self.vector_store.load_indices(
            index_path=index_path,
            metadata_path=metadata_path,
            bm25_path=bm25_path
        )
        if not load_success:
            print("错误：加载向量存储索引失败。问答系统无法运行。")
            # 可以选择抛出异常或设置一个错误状态
            raise RuntimeError("无法加载向量存储索引")
        print("向量存储加载成功。")
        
        print("问答系统初始化完成。")

    def _format_context(self, context: List[Tuple[str, Dict[str, Any], float]]) -> str:
        """将检索到的上下文格式化为字符串，用于 Prompt。"""
        if not context:
            return "没有找到相关的上下文信息。"
            
        formatted_context = "\n".join([
            f"---\n[来源: {metadata.get('source', '未知')}]\n{chunk}\n---"
            for chunk, metadata, score in context  # score 在这里不直接使用，但可用于调试
        ])
        return formatted_context

    def answer_question(self, user_question: str, search_top_k: int = 20) -> str:
        """
        回答用户的问题。

        Args:
            user_question (str): 用户提出的问题。
            search_top_k (int): 从向量存储中检索多少个相关块用于上下文。

        Returns:
            str: 由 DeepSeek 模型生成的答案。
        """
        print(f"\n收到问题: {user_question}")
        
        # 1. 检索相关上下文
        print(f"正在检索相关上下文 (Top {search_top_k})...")
        # 注意：这里的 top_k 是最终返回给 LLM 的数量，
        # hybrid_search 内部的 initial_k 用于初步检索 (默认为 20)
        try:
            # 调用 hybrid_search 时传入更新后的 top_k
            retrieved_context = self.vector_store.hybrid_search(user_question, top_k=search_top_k)
        except Exception as e:
             print(f"错误：混合搜索失败: {e}")
             return "抱歉，在检索相关信息时遇到错误。"
             
        if not retrieved_context:
            print("未找到相关上下文。")
            return "抱歉，我没有找到与您问题直接相关的信息。"
            
        formatted_context = self._format_context(retrieved_context)
        # print(f"\n--- 检索到的上下文 ---\n{formatted_context}\n----------------------") # 用于调试

        # 2. 构建优化后的 Prompt
        # 定义背景信息
        competition_list = [
            "智能芯片与计算思维专项赛", "智能数据采集装置设计专项赛", "智慧城市主题设计专项赛",
            "虚拟仿真平台创新设计专项赛", "无人驾驶智能车专项赛", "太空探索智能机器人专项赛",
            "太空电梯工程设计专项赛", "生成式人工智能应用专项赛", "三维程序创意设计专项赛",
            "人工智能综合创新专项赛", "开源鸿蒙机器人专项赛", "竞技机器人专项赛",
            "极地资源勘探专项赛", "机器人工程设计专项赛", "编程创作与信息学专项赛",
            "2024年（第12届）\"泰迪杯\"数据挖掘挑战赛", "3D编程模型创新设计专项赛",
            "\"未来校园\"智能应用专项赛"
        ]
        competition_string = ", ".join(competition_list)
        background_info = f"""背景知识（你已经知道的知识内容，回答的时候不需要重复说出来）：
1. 所有的"专项赛"都属于"第七届全国青少年人工智能创新挑战赛"的一部分。
2. "泰迪杯数据挖掘挑战赛组织委员会"组织的比赛是指"2024 年（第 12 届）"泰迪杯"数据挖掘挑战赛"。
3. 我们总共有这些比赛：{competition_string}
"""

        # 构建包含背景知识、角色定义和风格指南的 Prompt
        prompt = f"""{background_info}

**你的角色**：一个专业、严谨且富有启发性的竞赛信息智能客服。

**核心任务**：根据以上背景知识和下方提供的上下文信息，准确、清晰地回答用户的问题。

**行为准则与回答风格**：
1.  **严谨性优先**：所有直接回答问题的内容，特别是关于事实、数据、规则、时间、地点等信息，必须**严格依据**所提供的"上下文信息"和"背景知识"。禁止猜测、编造或引入任何外部信息。
2.  **自然流畅**：用自然、专业且易于理解的语言进行回答，避免生硬的机器人语气。
3.  **针对性回答**：
    *   **基础信息查询（如"..."是什么时候？"、"..."在哪里举办？"）**：直接、准确地从上下文中提取信息作答。如果信息不存在或不明确，请明确指出"根据现有资料，无法确定..."或类似表述。
    *   **统计/计算类问题（如"有多少种..."？"、"..."的比例是多少？"）**：基于上下文进行逻辑推断或计算。如果上下文信息不足以完成精确计算，请说明情况，并给出基于现有信息的最可能推论（同时说明其局限性）。
    *   **开放性问题（如"如何准备..."？"、"..."有什么建议？"）**：
        a.  首先，总结上下文信息中与问题相关的**事实性内容或建议**，并清晰说明这是基于所提供的资料。
        b.  然后，可以**补充**一些具有启发性的、更**通用**的思考角度或建议。这部分补充内容应与问题主题相关（例如，涉及创新、学习方法、策略规划、挑战应对等），但要**明确与前面基于资料的回答区分开**（例如，使用"除此之外，从更宏观的角度来看..."或"一些通用的建议可能包括..."等过渡语）。补充建议应着重于启发思考，而非提供未经证实的具体方案。
4.  **信息来源**：回答时**无需**注明信息来源于哪个具体文件（如 `[来源: xxx.pdf]`）。
5.  **无法回答时**：如果综合背景知识和上下文信息确实无法回答用户问题，请**直接、诚恳地告知用户**"抱歉，根据我所掌握的信息，暂时无法回答您关于'...'的问题。"，不要尝试模糊作答。

---
上下文信息：
{formatted_context}
---

问题：
{user_question}

回答：
"""
        # print(f"\n--- 生成的 Prompt ---\n{prompt}\n---------------------\n") # 用于调试

        # 3. 调用 DeepSeek 模型生成答案
        print("正在调用 DeepSeek 模型生成答案...")
        try:
            # 使用英文作为中转，确保API请求正常进行
            english_instruction = "Answer the question based on the provided context in Chinese."
            
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://competition-qa-system.com", # 示例网站URL
                    "X-Title": "Competition QA System", # 使用英文网站名称
                },
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": english_instruction
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
            )
            
            if completion.choices and len(completion.choices) > 0:
                final_answer = completion.choices[0].message.content
                print("生成答案成功。")
                return final_answer
            else:
                print("警告：DeepSeek 模型返回了空响应。")
                try: 
                    print(f"原始响应: {completion}")
                except Exception: 
                    pass
                return "抱歉，无法生成回答，模型返回了空响应。"

        except Exception as e:
            print(f"错误：调用 OpenRouter API 生成答案时出错: {e}")
            # 更详细的错误调试信息
            import traceback
            traceback.print_exc()
            
            # 尝试备用方法：将提示转为JSON字符串，确保编码正确
            try:
                print("尝试备用方法...")
                prompt_dict = {
                    "query": user_question,
                    "context": formatted_context,
                    "instruction": "请根据上下文回答问题，回答必须是中文。"
                }
                
                completion = self.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://competition-qa-system.com",
                        "X-Title": "Competition QA System",
                    },
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": json.dumps(prompt_dict, ensure_ascii=False)
                        }
                    ]
                )
                
                if completion.choices and len(completion.choices) > 0:
                    final_answer = completion.choices[0].message.content
                    print("备用方法生成答案成功。")
                    return final_answer
                
            except Exception as e2:
                print(f"备用方法也失败: {e2}")
                
            return "抱歉，在生成答案时遇到错误。请检查API连接或编码设置。"

# === 主执行逻辑 (示例) ===
if __name__ == '__main__':
    try:
        # 初始化问答系统
        qa_system = QASystem()
        
        # 示例问题循环
        print("\n欢迎使用竞赛智能客服机器人！输入 '退出' 来结束。")
        while True:
            user_input = input("请输入您的问题: ")
            if user_input.lower() == '退出':
                break
                
            # 获取答案
            answer = qa_system.answer_question(user_input)
            
            # 打印答案
            print(f"\n机器人回答: {answer}")
            
        print("\n感谢使用，再见！")
        
    except (ValueError, ConnectionError, RuntimeError) as e:
         print(f"问答系统初始化失败: {e}")
    except Exception as e:
        print(f"运行时发生意外错误: {e}")
        import traceback
        traceback.print_exc() 