from langchain_core.messages import HumanMessage

from utils.agente_workshop import app


def call_agente_workshop(thread_id: str, user_message: str):

    input_message = HumanMessage(content=user_message)

    configurable = {
        "metadata": {"thread_id": thread_id},
        "configurable": {"thread_id": thread_id}
    }

    initial_state = {
        "messages": [input_message],
        "thread_id": thread_id
    }

    result = app.invoke(input=initial_state, config=configurable)

    last_message = result.get("messages", [])[-1]
    ultima_pregunta = result.get("ultima_pregunta", '')
    ultima_respuesta = result.get("ultima_respuesta", '')

    return {
        "output_message": last_message.content,
        "thread_id": thread_id,
        "ultima_pregunta": ultima_pregunta,
        "ultima_respuesta": ultima_respuesta
    }