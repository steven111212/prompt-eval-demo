import logging
import os
from typing import Any, Dict, List, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import traceback
# 常數設定
CHROMA_PATH: str = "longcare_db"
OPENAI_AI_MODEL: str = "gpt-4o-mini"
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

# 長照服務專用提示模板
PROMPT_TEMPLATE: str = """
請根據以下台灣長照服務給付規定資料回答問題：

相關法規資料：
{context}

請回答這個問題：{question}

回答要求：
- 請提供準確且詳細的回答
- 只根據提供的法規資料來回答，不要添加資料中沒有的資訊
- 如果涉及服務編號（如BA01、BB01等），請明確標示
- 如果涉及費用，請提供具體金額
- 使用繁體中文回答
- 如果資料中沒有相關資訊，請明確說明
- 對於服務內容、適用對象、限制條件等要詳細說明
- 可以適當引用具體的法規條文和數據
"""


def call_api(
    prompt: str, options: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    使用 RAG 處理長照服務相關查詢並返回回應
    """
    try:
        # 從 context 中獲取動態加載的上下文
        dynamic_context = context.get('vars', {}).get('context', '')
        
        # 如果沒有提供動態上下文，則自己檢索
        if not dynamic_context:
            # 獲取相似度搜尋的數量
            k: int = options.get("config", {}).get("topK", 5)
            
            # 在向量資料庫中搜尋相關文件
            docs_chroma: List[Tuple[Document, float]] = (
                db_chroma.similarity_search_with_score(
                    prompt,
                    k=k,
                )
            )
            
            # 過濾相似度結果
            filtered_docs = [(doc, score) for doc, score in docs_chroma if score < 0.6]
            
            if not filtered_docs:
                filtered_docs = docs_chroma[:max(3, k//2)]
            
            # 準備上下文
            contexts = []
            context_parts = []
            
            for doc, score in filtered_docs:
                contexts.append(doc.page_content)
                
                metadata = doc.metadata
                service_code = metadata.get('service_code', '')
                service_type = metadata.get('service_type', '未分類')
                page_num = metadata.get('page_number', '未知')
                
                formatted_content = f"【{service_type}】"
                if service_code:
                    formatted_content += f" {service_code}"
                formatted_content += f" (第{page_num}頁)\n內容：{doc.page_content}\n"
                
                context_parts.append(formatted_content)
            
            context_text = "\n" + "="*50 + "\n".join(context_parts)
        else:
            # 使用動態提供的上下文
            context_text = dynamic_context
            # 從動態上下文中提取原始文檔列表（用於評估）
            contexts = dynamic_context.split('\n') if isinstance(dynamic_context, str) else []
        
        # 使用模板生成最終提示
        prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_template(
            PROMPT_TEMPLATE
        )
        final_prompt: str = prompt_template.format(
            context=context_text, 
            question=prompt
        )
        
        # 呼叫 OpenAI API
        chat: ChatOpenAI = ChatOpenAI(
            model_name=OPENAI_AI_MODEL,
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        
        message: HumanMessage = HumanMessage(content=final_prompt)
        response: AIMessage = chat.invoke([message])
        
        # 返回評估所需的完整格式
        result = {
            "output": response.content,
            "context": contexts,  # RAG 評估必需
        }
        
        logging.info(f"✅ 成功處理查詢，返回 {len(contexts)} 個上下文片段")
        return result
        
    except Exception as e:
        # 在 CI log 中會看到這個錯誤內容
        print("❌ ERROR in call_api():", e)
        traceback.print_exc()
        return {
            "output": f"[ERROR] call_api failed: {e}",
            "context": [],
        }