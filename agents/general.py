from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import create_retriever_tool
from .rag_utils import get_llm, get_rag_retriever


def create_general_agent():
    llm = get_llm()
    retriever = get_rag_retriever("general")

    rag_tool = create_retriever_tool(
        retriever,
        name="search_general_faq",
        description="Search general knowledge base for supported countries, currencies, card types, and age limits."
    )
    tools = [rag_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a General Support Agent. Answer general inquiry questions using the `search_general_faq` tool. Be friendly and helpful."),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)