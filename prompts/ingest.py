"""
台灣長照服務給付規定 RAG 系統的文件攝取腳本
處理長照服務給付相關 PDF 文件，分割文字並儲存至向量資料庫
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

# 配置日誌
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 常數設定
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("請設定 OPENAI_API_KEY 環境變數")

CHROMA_PATH: str = "longcare_db"
# 使用本地文件路徑，將你的 PDF 放在 documents 資料夾中
BASE_PATH: str = "./documents/"
CHUNK_SIZE: int = 500  # 針對長照規定文件，使用較大的塊大小以保持內容完整性
CHUNK_OVERLAP: int = 150
MAX_WORKERS: int = 2  # 本地處理，減少並行數
OPENAI_AI_EMBEDDING_MODEL: str = "text-embedding-3-large"

# 長照相關文件列表
PDF_FILES: List[str] = [
    "長期照顧服務.pdf",  # 你上傳的文件
    # 可以添加更多相關文件
    # "長照服務法.pdf",
    # "長照服務申請及給付辦法.pdf",
    # "長照專業服務手冊.pdf",
]

# 中文文字分割器的設定
CHINESE_SEPARATORS = [
    "\n\n",
    "\n", 
    "。",
    "；",
    "，",
    "、",
    " ",
    "",
]


def create_chinese_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    建立適合中文長照文件的文字分割器
    
    Returns:
        配置好的中文文字分割器
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHINESE_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


def extract_service_info(content: str) -> Dict[str, str]:
    """
    從文件內容中提取服務資訊
    
    Args:
        content: 文件內容
        
    Returns:
        包含服務類型、編號等資訊的字典
    """
    service_info = {}
    
    # 提取服務編號 (如 BA01, BB01, CA07 等)
    import re
    service_code_match = re.search(r'([A-Z]{2}\d{2}[a-z]?)', content)
    if service_code_match:
        service_info['service_code'] = service_code_match.group(1)
        
        # 根據編號前綴判斷服務類型
        prefix = service_code_match.group(1)[:2]
        service_types = {
            'AA': '長照醫事服務',
            'BA': '照顧及專業服務',
            'BB': '日間照顧服務',
            'BC': '家庭托顧服務',
            'BD': '社區式服務',
            'CA': '復能照護服務',
            'CB': '專業護理服務',
            'CC': '居家環境改善',
            'CD': '居家護理',
            'DA': '交通接送服務',
        }
        service_info['service_type'] = service_types.get(prefix, '其他服務')
    
    # 提取價格資訊
    price_match = re.search(r'(\d{1,5})\s+(\d{1,5})', content)
    if price_match:
        service_info['price_basic'] = price_match.group(1)
        service_info['price_subsidized'] = price_match.group(2)
    
    return service_info


def process_local_pdf(pdf_file: str) -> Tuple[str, List[Document]]:
    """
    處理本地 PDF 文件並返回文件塊
    
    Args:
        pdf_file: 要處理的 PDF 文件名
        
    Returns:
        包含文件名和文件塊列表的元組
    """
    file_path = os.path.join(BASE_PATH, pdf_file)
    
    try:
        # 檢查文件是否存在
        if not os.path.exists(file_path):
            logging.warning(f"文件不存在: {file_path}")
            return pdf_file, []
        
        loader = PyPDFLoader(file_path)
        pages: List[Document] = loader.load()
        
        # 為每頁添加詳細的元數據
        for i, page in enumerate(pages):
            page.metadata.update({
                "source": pdf_file,
                "page_number": i + 1,
                "document_type": "長照服務給付規定",
                "language": "zh-TW"
            })
            
            # 提取服務資訊並添加到元數據
            service_info = extract_service_info(page.page_content)
            page.metadata.update(service_info)
        
        text_splitter = create_chinese_text_splitter()
        chunks: List[Document] = text_splitter.split_documents(pages)
        
        # 為文件塊添加額外的元數據
        for chunk in chunks:
            # 保留頁面的所有元數據
            # 添加塊的特定資訊
            if len(chunk.page_content) > 100:  # 只處理有意義的塊
                chunk.metadata['chunk_length'] = len(chunk.page_content)
        
        logging.info(f"成功處理 {pdf_file}，生成 {len(chunks)} 個文件塊")
        return pdf_file, chunks
        
    except Exception as e:
        logging.error(f"處理 {pdf_file} 時發生錯誤: {str(e)}")
        return pdf_file, []


def process_pdfs() -> List[Document]:
    """
    處理所有 PDF 文件並分割成文件塊
    
    Returns:
        所有文件塊的列表
    """
    all_chunks: List[Document] = []
    
    # 檢查 documents 資料夾是否存在
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
        logging.info(f"已建立 {BASE_PATH} 資料夾，請將 PDF 文件放入此資料夾")
        return all_chunks
    
    # 檢查是否有文件要處理
    available_files = [f for f in PDF_FILES if os.path.exists(os.path.join(BASE_PATH, f))]
    
    if not available_files:
        logging.warning(f"在 {BASE_PATH} 中找不到任何指定的 PDF 文件")
        logging.info(f"請確保以下文件存在於 {BASE_PATH} 中:")
        for file in PDF_FILES:
            logging.info(f"  - {file}")
        return all_chunks
    
    logging.info(f"找到 {len(available_files)} 個文件要處理")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有 PDF 處理任務
        future_to_pdf: Dict = {
            executor.submit(process_local_pdf, pdf_file): pdf_file
            for pdf_file in available_files
        }
        
        # 使用進度條處理完成的任務
        with tqdm(total=len(available_files), desc="處理長照服務文件") as pbar:
            for future in as_completed(future_to_pdf):
                pdf_file: str = future_to_pdf[future]
                try:
                    _, chunks = future.result()
                    all_chunks.extend(chunks)
                    pbar.update(1)
                except Exception as e:
                    logging.error(f"處理 {pdf_file} 失敗: {str(e)}")
                    pbar.update(1)
    
    logging.info(f"總共處理了 {len(all_chunks)} 個文件塊，來源於 {len(available_files)} 個文件")
    return all_chunks


def create_vector_store(chunks: List[Document], batch_size: int = 50) -> None:
    """
    從文件塊建立並持久化向量資料庫
    
    Args:
        chunks: 要嵌入的文件塊列表
        batch_size: 每批次處理的文件數量
    """
    if not chunks:
        logging.warning("沒有文件塊可處理")
        return
    
    embeddings = OpenAIEmbeddings(
        model=OPENAI_AI_EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    
    logging.info("建立長照服務給付規定向量資料庫...")
    
    # 處理第一批
    current_batch: List[Document] = chunks[:batch_size]
    db = Chroma.from_documents(
        current_batch,
        embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="longcare_service_collection",
    )
    
    # 處理剩餘批次
    with tqdm(
        total=len(chunks), 
        initial=batch_size, 
        desc="嵌入長照服務文件"
    ) as pbar:
        for i in range(batch_size, len(chunks), batch_size):
            current_batch = chunks[i : i + batch_size]
            db.add_documents(current_batch)
            pbar.update(len(current_batch))
    
    logging.info(f"長照服務給付規定向量資料庫已建立並保存至 {CHROMA_PATH}")


def setup_documents_folder():
    """
    設置文件資料夾並提供使用說明
    """
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
        
    readme_content = """# 長照服務給付規定文件資料夾

請將以下 PDF 文件放入此資料夾：

1. 長照服務給付及支付基準.pdf
2. 其他相關長照法規文件

## 使用說明

1. 將 PDF 文件放入此資料夾
2. 執行 `python ingest_longcare.py`
3. 系統會自動處理文件並建立向量資料庫

## 支援的文件類型

- 長照服務給付及支付基準
- 長照服務法規
- 長照專業服務手冊
- 其他相關法規文件
"""
    
    readme_path = os.path.join(BASE_PATH, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)


def main() -> None:
    """主執行函數"""
    logging.info("開始處理台灣長照服務給付規定文件...")
    
    # 設置文件資料夾
    setup_documents_folder()
    
    # 處理文件
    chunks: List[Document] = process_pdfs()
    
    if chunks:
        create_vector_store(chunks)
        logging.info("台灣長照服務給付規定 RAG 系統建置完成！")
        
        # 顯示統計資訊
        service_types = {}
        for chunk in chunks:
            service_type = chunk.metadata.get('service_type', '未分類')
            service_types[service_type] = service_types.get(service_type, 0) + 1
        
        logging.info("服務類型統計:")
        for service_type, count in service_types.items():
            logging.info(f"  {service_type}: {count} 個文件塊")
            
    else:
        logging.warning("沒有成功處理任何文件")
        logging.info(f"請確保將 PDF 文件放入 {BASE_PATH} 資料夾中")


if __name__ == "__main__":
    main()