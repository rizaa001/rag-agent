import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from tools import search_postings, web_search

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

tools = [search_postings, web_search]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful job market research assistant.

Use search_postings for questions about saved job postings, required skills, or qualifications.
Use web_search for current company news, hiring status, or recent market trends not in the saved postings.

Only answer based on what the tools return. If neither tool gives you enough information, say so honestly instead of guessing."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or 'quit'): ")
        if query.lower() == "quit":
            break
        result = agent_executor.invoke({"input": query})
        print("\n--- ANSWER ---")
        print(result["output"])