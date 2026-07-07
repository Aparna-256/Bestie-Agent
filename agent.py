from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from tools import (
    calculator,
    current_time,
    tell_joke,
    search_tool,
    search_pdf
)


def create_bestie_agent(llm):

    tools = [
        calculator,
        current_time,
        tell_joke,
        search_tool,
        search_pdf
    ]

    template = """
You are Bestie Bot, a friendly AI assistant.

You have access to the following tools:

{tools}

Available tools:
{tool_names}

Tool Usage Rules:

- Use search_pdf for questions about the uploaded PDF.
- Use current_time for date and time.
- Use calculator for mathematical calculations.
- Use tell_joke only when the user asks for a joke.
- Use search_tool for current or internet information.
- Otherwise answer directly.

Use the following format:

Question: the user's question

Thought: think about what to do

Action: one of [{tool_names}]

Action Input: the input to the tool

Observation: the result of the tool

...(repeat if needed)...

Thought: I now know the final answer

Final Answer: the answer to the user

Question: {input}

{agent_scratchpad}
"""
    prompt = PromptTemplate.from_template(template)

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate",
    )

    return agent_executor