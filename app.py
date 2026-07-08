import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from agent import create_bestie_agent
from memory import chat_history

load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)

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