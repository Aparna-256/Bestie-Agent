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
You are Bestie Bot, a cheerful AI assistant.

You have access to the following tools:

{tools}

Use the following format:

Question: the user's question

Thought: think about what to do

Action: one of [{tool_names}]

Action Input: the input to the tool

Observation: the result of the tool

...(repeat Thought/Action/Observation if needed)...

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
    )

    return agent_executor