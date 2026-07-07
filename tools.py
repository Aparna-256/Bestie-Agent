from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
import random

# Import RAG function
from rag import get_context


# ==========================================================
# 🎀 Calculator Tool
# ==========================================================

@tool
def calculator(expression: str) -> str:
    """
    Solves mathematical expressions.

    Example:
    45 * 89
    """

    try:
        return str(eval(expression))

    except Exception:
        return "❌ Invalid mathematical expression."


# ==========================================================
# 🎀 Current Time Tool
# ==========================================================

@tool
def current_time(query: str = "") -> str:
    """
    Returns the current local time.
    """

    return datetime.now().strftime("%d %B %Y | %I:%M:%S %p")


# ==========================================================
# 🎀 Joke Tool
# ==========================================================

@tool
def tell_joke(query: str = "") -> str:
    """
    Returns a random joke.
    """

    jokes = [

        "😂 Why don't scientists trust atoms? Because they make up everything!",

        "🤣 Why did the bicycle fall over? Because it was two-tired!",

        "😆 Why did the tomato blush? Because it saw the salad dressing!",

        "🤭 Why did the math book look sad? Because it had too many problems!",

        "😂 Why did the scarecrow win an award? Because he was outstanding in his field!"

    ]

    return random.choice(jokes)


# ==========================================================
# 🌍 Internet Search Tool
# ==========================================================

search_tool = DuckDuckGoSearchRun()


# ==========================================================
# 📄 PDF Search Tool (RAG)
# ==========================================================

@tool
def search_pdf(question: str) -> str:
    """
Search ONLY the uploaded PDF.

Use this tool whenever the user asks about:

- uploaded document
- names
- dates
- confidential information
- project details
- summaries

Never answer these questions from your own knowledge.

Return the most relevant information from the uploaded PDF.
"""

    context = get_context(question)

    if context.strip() == "":
        return "No relevant information found inside the uploaded PDF."

    return context