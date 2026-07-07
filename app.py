import os
import sys
from dotenv import load_dotenv

# Reconfigure stdout to support printing emojis in Windows command prompt
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from agent import create_bestie_agent
from memory import chat_history

load_dotenv()

if os.environ.get("GROQ_API_KEY"):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )
elif os.environ.get("GEMINI_API_KEY"):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0,
    )
else:
    raise ValueError("Neither GROQ_API_KEY nor GEMINI_API_KEY is set in the environment.")

agent = create_bestie_agent(llm)

print("🎀 Welcome to Bestie Agent!")
print("Type 'exit' anytime.\n")

while True:

    user_input = input("YOU 🎀 : ")

    if user_input.lower() == "exit":
        print("Goodbye Bestie 💖")
        break

    history = ""

    for message in chat_history.messages:
        history += f"{message.type}: {message.content}\n"

    response = agent.invoke(
        {
            "input": user_input,
            "chat_history": history,
        }
    )

    print("\n🎀 Bestie 💖")
    print("=" * 50)
    print(response["output"])
    print("=" * 50)

    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(response["output"])