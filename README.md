# ✨ Bestie Bot 🎀

An AI-powered personal assistant built using **LangChain**, **Groq LLM**, **FAISS**, and **Streamlit**.

Bestie Bot is designed to have natural conversations, remember previous chats, answer questions using PDFs through RAG, and use tools to make interactions smarter.

---

# ✨ Features 🎀

- AI-powered conversational assistant
- Chat memory for context-aware conversations
- Chat with PDFs using Retrieval-Augmented Generation (RAG)
- Semantic search with FAISS
- Groq LLM integration
- Custom tool support
- Streamlit web interface
- Modular and beginner-friendly project structure

---

# ✨ Tech Stack 🎀

- Python
- LangChain
- Groq API
- FAISS
- HuggingFace Embeddings
- Streamlit
- Python Dotenv

---

# ✨ Project Structure 🎀

```text
Bestie-Bot/
│
├── app.py
├── streamlit_app.py
├── agent.py
├── rag.py
├── memory.py
├── tools.py
├── requirements.txt
├── .env.example
└── LICENSE
```

---

# ✨ Installation 🎀

Clone the repository

```bash
git clone https://github.com/<your-username>/Bestie-Bot.git
```

Move into the project

```bash
cd Bestie-Bot
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

# ✨ Environment Variables 🎀

Create a `.env` file from `.env.example`

```env
GROQ_API_KEY=your_groq_api_key
```

---

# ✨ Run the Project 🎀

Run the terminal version

```bash
python app.py
```

Run the Streamlit version

```bash
streamlit run streamlit_app.py
```

---

# ✨ Future Improvements 🎀

- Voice Assistant
- Image Understanding
- Web Search Integration
- Multi-PDF Support
- Persistent Memory
- LangGraph Workflows

---

# ✨ License 🎀

This project is licensed under the MIT License.

---

# ✨ Author 🎀

**Aparna Jha**
