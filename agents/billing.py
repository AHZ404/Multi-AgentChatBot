from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from .rag_utils import get_llm

@tool
def check_invoice(user_id: str) -> str:
    """Check invoice and recent transaction charges for a user."""
    # Mock database call
    return f"Invoice #1092 for user {user_id}: Paid $45.00 on 2026-09-15. No extra fees found."

def create_billing_agent():
    llm = get_llm()
    tools = [check_invoice]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Billing Support Specialist. Use the `check_invoice` tool to assist users with fees, charges, and refunds."),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)