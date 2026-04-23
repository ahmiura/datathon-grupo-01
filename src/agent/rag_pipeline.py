import os
import logging
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VECTOR_STORE_DIR = "data/processed/faiss_index"
DEFAULT_PDF_PATH = "data/raw"

def build_or_load_vector_store(pdf_path: str = DEFAULT_PDF_PATH):
    embed_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    embeddings = GoogleGenerativeAIEmbeddings(model=embed_model)
    
    # Se o banco vetorial já existe, apenas carrega (evita reprocessamento a cada chamada - GAP 03)
    if os.path.exists(VECTOR_STORE_DIR):
        return FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
        
    logger.info(f"Vector Store não encontrado. Construindo a partir de {pdf_path}...")
    if not os.path.exists(pdf_path):
        logger.warning(f"Documento {pdf_path} não encontrado. Retornando banco vetorial vazio.")
        return None
        
    # 1. Carrega o Documento
    loader = PyPDFDirectoryLoader(pdf_path)
    docs = loader.load()
    
    # 2. Divide em pedaços (Chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # 3. Cria o Vector Store em lotes (Proteção contra Rate Limit 429 - Free Tier)
    vectorstore = None
    batch_size = 10  # Envia 10 pedaços por vez
    
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i+batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        
        if i + batch_size < len(splits):
            time.sleep(6)  # Pausa de 6 segundos entre os lotes
            
    vectorstore.save_local(VECTOR_STORE_DIR)
    return vectorstore

def update_vector_store(pdf_path: str = DEFAULT_PDF_PATH) -> bool:
    """
    Força a reconstrução do Vector Store. 
    Usado pelo Airflow (Continuous Training) para injetar novos documentos no cérebro do Agente.
    """
    embed_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    embeddings = GoogleGenerativeAIEmbeddings(model=embed_model)
    
    logger.info(f"Atualizando Vector Store a partir de {pdf_path}...")
    if not os.path.exists(pdf_path):
        logger.error(f"Diretório {pdf_path} não encontrado.")
        return False
        
    loader = PyPDFDirectoryLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Criação em lotes para evitar Rate Limit na atualização via Airflow
    vectorstore = None
    batch_size = 10
    
    logger.info(f"Iniciando vetorização de {len(splits)} chunks em lotes...")
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i+batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
            
        logger.info(f"Vetorizados {min(i+batch_size, len(splits))}/{len(splits)} chunks...")
        if i + batch_size < len(splits):
            time.sleep(6)
            
    vectorstore.save_local(VECTOR_STORE_DIR)
    logger.info(f"✅ Vector Store atualizado com sucesso com {len(docs)} páginas!")
    return True

def query_documents(query: str, k: int = 3) -> str:
    vectorstore = build_or_load_vector_store()
    if not vectorstore:
        return "Nenhum documento interno disponível para consulta no momento."
        
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([f"Trecho:\n{doc.page_content}" for doc in docs])
    return context
