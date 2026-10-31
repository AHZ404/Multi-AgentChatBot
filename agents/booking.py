from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from .rag_utils import get_llm

@tool
def check_available(order_id: str) -> str:
    """Check card delivery status or physical card availability."""
    return f"Order #{order_id}: Physical card dispatched on 2026-09-28. Estimated delivery in 3 business days."

def create_booking_agent():
    llm = get_llm()
    tools = [check_available]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Booking and Physical Card Specialist. Use `check_available` to trace card shipments and deliveries."),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)