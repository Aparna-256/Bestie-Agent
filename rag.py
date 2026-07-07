import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ==========================================
# 📂 Folder to store uploaded PDFs
# ==========================================

UPLOAD_FOLDER = "uploads"
VECTOR_DB_FOLDER = "vectorstore"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)


# ==========================================
# 🎀 Embedding Model
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Cache the vector store in memory
VECTORSTORE = None


# ==========================================
# 📚 Build Vector Database
# ==========================================

def create_vector_database(pdf_path: str):

    global VECTORSTORE

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    VECTORSTORE = FAISS.from_documents(
        chunks,
        embeddings
    )

    VECTORSTORE.save_local(VECTOR_DB_FOLDER)

    return VECTORSTORE


# ==========================================
# 📖 Load Existing Vector Database
# ==========================================

def load_vector_database():

    if not os.path.exists(VECTOR_DB_FOLDER):
        return None

    try:

        vectorstore = FAISS.load_local(
            VECTOR_DB_FOLDER,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vectorstore

    except Exception:
        return None


# ==========================================
# 🔍 Retrieve Similar Chunks
# ==========================================

def retrieve_documents(query: str, k: int = 4):

    global VECTORSTORE

    if VECTORSTORE is None:
        VECTORSTORE = load_vector_database()

    if VECTORSTORE is None:
        return []

    return VECTORSTORE.similarity_search(query, k=k)


# ==========================================
# 📝 Build Context
# ==========================================

def get_context(query: str):

    docs = retrieve_documents(query)

    if len(docs) == 0:
        print("\n❌ No relevant documents found.\n")
        return ""

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

   

    return context