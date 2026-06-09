import os
import glob
import warnings
warnings.filterwarnings("ignore")

# 从 .env 文件加载敏感配置（API Key 等），不硬编码在源码中
from dotenv import load_dotenv
load_dotenv()

# PDF 解析 & OCR（图片型 PDF 需要 OCR 提取文字）
import fitz  # pymupdf
import easyocr

# 文本分割
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# 向量化 & 向量库
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# 在线大模型对接
from langchain_openai import ChatOpenAI
# RAG问答链
from langchain_classic.chains import RetrievalQA
# 网页对话界面
import gradio as gr

# ====================== 配置区（敏感信息在 .env 文件中）======================
PDF_FOLDER = "./pdfs"               # PDF存放文件夹
VECTOR_STORE_PATH = "./vector_db"   # 向量库存放路径
# DeepSeek在线大模型配置 —— 从环境变量读取，避免源码泄露密钥
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL_TEMP = float(os.getenv("MODEL_TEMP", "0.0"))  # 越低越严谨，不会编造内容
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "4"))  # 每次检索4段文档片段

# 启动前校验：必须配置 API Key
if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "未检测到 DEEPSEEK_API_KEY，请在 .env 文件中配置。\n"
        "参考 .env.example 文件，填入你的 DeepSeek API Key。"
    )
# ========================================================================

# 全局 EasyOCR Reader（只需初始化一次，支持中文）
ocr_reader = None

def get_ocr_reader():
    """延迟初始化 OCR（首次调用时加载模型，之后复用）"""
    global ocr_reader
    if ocr_reader is None:
        print("正在加载 EasyOCR 中文识别模型（首次启动需下载，约 100MB）...")
        ocr_reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
        print("OCR 模型加载完成")
    return ocr_reader


def load_pdfs(PDF_FOLDER):
    """
    加载文件夹中全部 PDF，自动判断是否需要 OCR：
    - 如果 pymupdf 能提取到文字，直接使用
    - 如果是图片型 PDF（无文字层），则用 EasyOCR 识别
    """
    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"在 {PDF_FOLDER} 中未找到任何 PDF 文件")

    all_docs = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"正在处理: {filename}")
        doc = fitz.open(pdf_path)

        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            text = page.get_text().strip()

            if len(text) > 20:
                # 有足够的文字层，直接使用
                print(f"  第 {page_idx + 1} 页: 直接提取文字 ({len(text)} 字符)")
            else:
                # 图片型 PDF，需要 OCR
                print(f"  第 {page_idx + 1} 页: 无文字层，启动 OCR 识别...")
                reader = get_ocr_reader()
                # 渲染页面为图片（300 DPI 保证识别精度）
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                # EasyOCR 识别
                results = reader.readtext(img_bytes, detail=0)
                text = "\n".join(results)
                print(f"    OCR 识别到 {len(text)} 字符")

            if text.strip():
                all_docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "page": page_idx,
                        "total_pages": doc.page_count,
                        "filename": filename,
                    }
                ))

        doc.close()

    print(f"\nPDF 加载完成，共 {len(all_docs)} 页有效内容")
    return all_docs


def build_knowledge_base():
    """加载pdf（含OCR）、切片、构建/加载向量知识库"""
    # 1. 加载 PDF（自动处理文字层 + 图片 OCR）
    raw_docs = load_pdfs(PDF_FOLDER)

    # 2. 文本分片（适配中文 Lesson Learn 文档）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "；", "，", "：", "？", "！", " "]
    )
    split_docs = splitter.split_documents(raw_docs)
    print(f"文档分片完成，片段总数：{len(split_docs)}")

    # 3. 向量化模型（本地轻量向量模型，无需联网）
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. 持久化向量库，重复启动不用重新解析PDF
    if os.path.exists(VECTOR_STORE_PATH):
        vector_db = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("加载已有知识库成功")
    else:
        vector_db = FAISS.from_documents(split_docs, embeddings)
        vector_db.save_local(VECTOR_STORE_PATH)
        print("全新构建知识库并保存完成")
    return vector_db

def init_qa_chain(vector_db):
    """初始化RAG问答链，对接DeepSeek在线大模型"""
    llm = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=MODEL_TEMP,
        model_name="deepseek-chat"
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVE_TOP_K})
    # RAG核心链：只基于检索到的文档回答，禁止瞎编
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True # 返回参考文档片段，方便溯源
    )
    return qa_chain

def chat_func(question, history):
    """对话接口函数，给Gradio网页调用"""
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    # 溯源引用（可选，展示来源PDF页码）
    source_info = []
    for doc in result["source_documents"]:
        src = os.path.basename(doc.metadata["source"])
        page = doc.metadata["page"] + 1
        source_info.append(f"【文档】{src} 第{page}页")
    if source_info:
        answer += "\n\n参考来源：\n" + "\n".join(source_info)
    return answer

if __name__ == "__main__":
    # 1. 构建知识库
    db = build_knowledge_base()
    # 2. 初始化问答模型
    qa_chain = init_qa_chain(db)
    # 3. 启动网页对话窗口（浏览器自动弹出）
    demo = gr.ChatInterface(
        fn=chat_func,
        title="Lesson Learn 经验知识库问答助手",
        description="仅基于上传PDF文档回答，无相关内容会如实告知，不编造信息",
        chatbot=gr.Chatbot(height=600)
    )
    # Windows本地访问地址：http://127.0.0.1:7860
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)