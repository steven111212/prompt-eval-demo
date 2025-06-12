import logging
import os
from typing import Any, Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# 常數設定 - 與 rag_provider.py 保持一致
CHROMA_PATH: str = "longcare_db"
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_AI_EMBEDDING_MODEL: str = "text-embedding-3-large"

# 初始化嵌入和載入 Chroma 資料庫
embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
    model=OPENAI_AI_EMBEDDING_MODEL, 
    openai_api_key=OPENAI_API_KEY
)

db_chroma: Chroma = Chroma(
    collection_name="longcare_service_collection",
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
)


def retrieve_documents(query: str, top_k: int = 5) -> List[str]:
    """
    根據問題檢索相關文檔
    
    Args:
        query: 用戶的問題
        top_k: 返回的文檔數量
        
    Returns:
        List[str]: 相關文檔內容列表
    """
    try:
        # 在向量資料庫中搜尋相關文件
        docs_chroma: List[Tuple[Document, float]] = (
            db_chroma.similarity_search_with_score(
                query,
                k=top_k,
            )
        )
        
        # 過濾相似度結果 (與 rag_provider.py 保持一致的邏輯)
        filtered_docs = [(doc, score) for doc, score in docs_chroma if score < 0.6]
        
        if not filtered_docs:
            filtered_docs = docs_chroma[:max(3, top_k//2)]
        
        # 提取文檔內容
        contexts = []
        for doc, score in filtered_docs:
            contexts.append(doc.page_content)
            
        logging.info(f"✅ 成功檢索到 {len(contexts)} 個相關文檔")
        return contexts
        
    except Exception as e:
        logging.error(f"❌ retrieve_documents 發生錯誤: {str(e)}")
        return []


def get_var(var_name: str, prompt: str, other_vars: Dict[str, Any]) -> Dict[str, Any]:
    """
    promptfoo 動態變量獲取函數
    
    Args:
        var_name: 變量名稱 (通常是 'context')
        prompt: 當前的 prompt 模板
        other_vars: 其他變量字典，包含 'query' 等
        
    Returns:
        Dict: 包含 'output' 鍵的字典
    """
    try:
        # 從 other_vars 中獲取問題
        query = other_vars.get('query', '')
        
        if not query:
            logging.warning("⚠️  未找到 query 變量")
            return {'output': ''}
        
        # 獲取 top_k 參數 (如果有的話)
        top_k = other_vars.get('top_k', 5)
        
        # 檢索相關文檔
        contexts = retrieve_documents(query, top_k)
        
        if not contexts:
            logging.warning(f"⚠️  未找到與問題相關的文檔: {query}")
            return {'output': '未找到相關文檔'}
        
        # 格式化上下文 (與 rag_provider.py 保持一致的格式)
        formatted_contexts = []
        for i, context in enumerate(contexts, 1):
            formatted_context = f"【文檔 {i}】\n{context}\n"
            formatted_contexts.append(formatted_context)
        
        # 合併所有上下文
        final_context = "\n\n".join(formatted_contexts)
        
        logging.info(f"✅ 成功為問題 '{query[:50]}...' 生成上下文")
        
        return {
            'output': final_context
        }
        
    except Exception as e:
        logging.error(f"❌ get_var 發生錯誤: {str(e)}")
        return {
            'output': f'錯誤：無法獲取上下文 - {str(e)}'
        }


