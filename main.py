import os
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_core.documents import Document
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from typing import List, Tuple, Any, Dict
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory


# Load environment variables
load_dotenv()

# Configuration
OLLAMA_EMBEDDINGS_URL = os.getenv("OLLAMA_EMBEDDINGS_URL", "http://localhost:11434")
OLLAMA_EMBEDDINGS_MODEL = os.getenv("OLLAMA_EMBEDDINGS_MODEL", "bge-m3:latest")
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma3:latest")
QDRANT_URL = os.getenv("QDRANT_URL", ":memory:")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "qa_collection")
DIRPATH = os.getenv("DIRPATH", "./data")


def get_embeddings():
    
    return OllamaEmbeddings(
        base_url=OLLAMA_EMBEDDINGS_URL,
        model=OLLAMA_EMBEDDINGS_MODEL
    )

def get_chat_model():
    return ChatOllama(
        base_url=OLLAMA_CHAT_URL,
        model=OLLAMA_CHAT_MODEL,
        temperature=0.1
    )

def load_and_split_documents(directory_path: str) -> List[Document]:
    """Load and split PDF documents from a directory.
    
    Args:
        directory_path: Path to the directory containing PDF files
        
    Returns:
        List of Document objects
    """
    from glob import glob
    
    all_docs = []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    # Find all PDF files in the directory and subdirectories
    pdf_files = glob(f"{directory_path}/**/*.pdf", recursive=True)
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {directory_path}")
        
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            split_docs = text_splitter.split_documents(documents)
            all_docs.extend(split_docs)
            print(f"Processed: {pdf_file} - {len(split_docs)} chunks")
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            continue
            
    print(f"\nTotal documents processed: {len(pdf_files)}")
    print(f"Total chunks created: {len(all_docs)}")
    return all_docs

def get_document_store(docs, embeddings):
    return Qdrant.from_documents(
        docs,
        embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        prefer_grpc=False,
        force_recreate=True
    )

def memory_chat():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer")

def main():
    embeddings = get_embeddings()
    docs = load_and_split_documents(DIRPATH)
    doc_store = get_document_store(docs, embeddings)
    llm = get_chat_model()
    memory = memory_chat()

    # 設置系統提示詞
    system_prompt = SystemMessagePromptTemplate.from_template(
        """
        你是一位專門根據下方文件內容提供資訊的小幫手。你只能根據「以下資訊」回答問題，不能憑空猜測，也不能提供文件以外的知識。
        如果用戶的問題跟下方資訊無關，請直接回覆：「很抱歉，我只能回答與提供文件相關的問題。」
        請使用繁體中文回答問題。
        以下是你查到的相關資訊：
        {context}
        """
    )

    human_prompt = HumanMessagePromptTemplate.from_template("{question}")
        
    CHAT_PROMPT = ChatPromptTemplate.from_messages([
        system_prompt,
        human_prompt
    ])
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=doc_store.as_retriever(),
        combine_docs_chain_kwargs={"prompt": CHAT_PROMPT},
        return_source_documents=True,
        memory=memory,
        verbose=False
    )

    while True:
        query = input('you: ')
        if query == 'q':
            break
        result = qa.invoke({"question": query})
        print("answer:", result["answer"])

if __name__ == "__main__":
    main()