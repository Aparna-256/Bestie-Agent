import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from agent import create_bestie_agent
from rag import create_vector_database

# ==========================================================
# 🎀 Load Environment Variables
# ==========================================================

load_dotenv()

# ==========================================================
# 🎀 Streamlit Config
# ==========================================================

st.set_page_config(
    page_title="🎀 Bestie Bot",
    page_icon="🎀",
    layout="wide"
)

# ==========================================================
# 🎀 Create LLM
# ==========================================================

MODEL_NAME = ""

if os.getenv("GOOGLE_API_KEY"):

    MODEL_NAME = "Gemini 2.5 Flash"

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
    )

elif os.getenv("GROQ_API_KEY"):

    MODEL_NAME = "Llama 3.3 70B (Groq)"

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

else:

    st.error(
        "❌ No API key found.\n\nPlease add either GOOGLE_API_KEY or GROQ_API_KEY to your .env file."
    )
    st.stop()

# ==========================================================
# 🎀 Create Agent
# ==========================================================

agent = create_bestie_agent(llm)

# ==========================================================
# 🎀 Main UI
# ==========================================================

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

st.sidebar.success(f"🤖 Using: {MODEL_NAME}")

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

    with st.sidebar.spinner("📚 Creating Vector Database..."):

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

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

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

                error = str(e)

                if "429" in error:

                    answer = (
                        "⚠️ **Rate limit reached.**\n\n"
                        "The AI provider has temporarily limited requests on the free tier.\n\n"
                        "Please wait a few seconds and try again."
                    )

                else:

                    answer = (
                        "❌ **Something went wrong.**\n\n"
                        f"```\n{error}\n```"
                    )

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )