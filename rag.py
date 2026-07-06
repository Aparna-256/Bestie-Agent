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


# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)


# ==========================================
# 🎀 Embedding Model
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# 📚 Build Vector Database
# ==========================================

def create_vector_database(pdf_path: str):

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Save locally
    vectorstore.save_local(VECTOR_DB_FOLDER)

    return vectorstore


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
# 🔍 Search Similar Chunks
# ==========================================

def retrieve_documents(query: str, k: int = 4):

    vectorstore = load_vector_database()

    if vectorstore is None:
        return []

    docs = vectorstore.similarity_search(
        query,
        k=k
    )

    return docs


# ==========================================
# 📝 Build Context for LLM
# ==========================================

def get_context(query: str):

    docs = retrieve_documents(query)

    if len(docs) == 0:
        return ""

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return context