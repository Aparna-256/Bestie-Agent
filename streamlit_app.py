import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from agent import create_bestie_agent
from rag import create_vector_database

# ==========================================================
# 🎀 Load Environment Variables
# ==========================================================

load_dotenv()

# ==========================================================
# 🎀 Create LLM
# ==========================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# ==========================================================
# 🎀 Create Agent
# ==========================================================

agent = create_bestie_agent(llm)

# ==========================================================
# 🎀 Streamlit Config
# ==========================================================

st.set_page_config(
    page_title="🎀 Bestie Bot",
    page_icon="🎀",
    layout="wide"
)

st.title("🎀 Bestie Bot")
st.caption("Your Personal AI Assistant 💖")

# ==========================================================
# 📂 Upload Folder
# ==========================================================

os.makedirs("uploads", exist_ok=True)

# ==========================================================
# 📄 Sidebar
# ==========================================================

st.sidebar.title("📂 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    pdf_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.sidebar.spinner("Creating Vector Database..."):

        create_vector_database(pdf_path)

    st.sidebar.success("✅ PDF Ready!")

# ==========================================================
# 🎀 Chat History
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================================
# Display Previous Messages
# ==========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================================
# Chat Input
# ==========================================================

user_input = st.chat_input(
    "Ask Bestie anything... 💌"
)

if user_input:

    # --------------------------
    # Show User Message
    # --------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # --------------------------
    # AI Response
    # --------------------------

    with st.chat_message("assistant"):

        with st.spinner("🎀 Bestie is thinking..."):

            try:

                response = agent.invoke(
                    {
                        "input": user_input
                    }
                )

                answer = response["output"]

            except Exception as e:

                answer = f"❌ Error:\n\n{str(e)}"

            st.markdown(answer)

    # --------------------------
    # Save AI Response
    # --------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )