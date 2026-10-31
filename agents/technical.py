from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import create_retriever_tool
from .rag_utils import get_llm, get_rag_retriever


def create_technical_agent():
    llm = get_llm()
    retriever = get_rag_retriever("technical")

    # Define RAG Tool
    rag_tool = create_retriever_tool(
        retriever,
        name="search_technical_docs",
        description="Search technical documentation for app errors, PIN resets, passcode issues, and card activation."
    )
    tools = [rag_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Technical Support Agent. Use the `search_technical_docs` tool to answer technical and app issues. Be concise and step-by-step."),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)