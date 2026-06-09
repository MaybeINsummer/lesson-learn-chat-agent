"""快速验证所有模块是否能正常导入和 API Key 是否配置"""
import os
from dotenv import load_dotenv
load_dotenv()

# 测试所有导入
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_classic.chains import RetrievalQA
import gradio as gr

print("All imports OK")

# 检查 API Key
key = os.getenv("DEEPSEEK_API_KEY")
if key:
    masked = key[:8] + "****" + key[-4:]
    print(f"API Key configured: {masked}")
else:
    print("ERROR: API Key not found in .env")
    exit(1)

# 检查 PDF 目录
if os.path.isdir("./pdfs"):
    pdfs = [f for f in os.listdir("./pdfs") if f.endswith(".pdf")]
    print(f"PDF folder OK: {len(pdfs)} files found")
    for p in pdfs:
        print(f"  - {p}")
else:
    print("WARNING: ./pdfs directory not found")

print("\nAll checks passed — program is ready to run.")
