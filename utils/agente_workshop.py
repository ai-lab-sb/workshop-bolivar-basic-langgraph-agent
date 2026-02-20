from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode

from helpers.prompts import AGENT_SYSTEM_PROMPT
from utils.memory import FirestoreSaver
from helpers.clients_n_models import tools, llm_with_tools


class AgentState(MessagesState):
    thread_id: str
    solicitud: str
    ultima_pregunta: str
    ultima_respuesta: str


def model_call(state: AgentState) -> AgentState:

    system_prompt = SystemMessage(content=AGENT_SYSTEM_PROMPT)

    print(f"System prompt: {system_prompt}")
    print(f"Messages: {state['messages']}")

    response = llm_with_tools.invoke([system_prompt] + state['messages'])

    return {'messages': response}

def should_continue(state: AgentState) -> str:

    messages = state['messages']
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)

graph.add_node('agent', model_call)
tool_node = ToolNode(tools=tools)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'agent')
graph.add_conditional_edges(
    'agent',
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
graph.add_edge('tools', 'agent')

memory = FirestoreSaver(database="workshop-agent-memory", collection_name="workshop-agent-memory", pw_collection_name="workshop-agent-memory-pw")


app = graph.compile(checkpointer=memory)