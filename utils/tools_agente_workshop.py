from pydantic import BaseModel, Field
from typing_extensions import Annotated
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from helpers.clients_n_models import FirestoreVectorStore, llm


class InputRAGTool(BaseModel):
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]
    pregunta: str = Field(None, description='La pregunta que tiene el usuario ed seguros o coberturas.')


def rag(user_input: str) -> str:
    """
    Receives a user question, retrieves relevant documents from Firestore,
    and returns the LLM response augmented with that context.
    """

    # --- Retriever ---
    vector_store = FirestoreVectorStore(
        project="sb-iadaia-cap-dev",
        database="vs-rag-workshop",
        collection="t_seguros_fake_gemini",
    )

    results = vector_store.as_retriever(query=user_input, k=3)

    # Build context from retrieved documents
    context = "\n\n".join(
        f"- {doc.get('name', '')}: {doc.get('complete_description', doc.get('short_description', ''))}"
        for doc in results
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful insurance assistant. Use ONLY the following context "
         "to answer the user's question. If the context does not contain enough "
         "information, say so. Always answer in Spanish.\n\n"
         "Context:\n{context}"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"context": context, "question": user_input})


@tool(args_schema=InputRAGTool)
def call_rag_productos_y_coberturas(pregunta: str, state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> dict:
    """
    Esta tool hace una búsqueda semántica de coberturas o seguros para un usuario, dada una pregunta específica.
    """

    try:

        rag_response = rag(user_input=pregunta)

        update_dict = {
            "messages": [
                ToolMessage(
                    content=f"Tool ejecutada correctamente. Respuesta de la tool: {rag_response}",
                    tool_call_id=tool_call_id
                )
            ],
            "ultima_pregunta": pregunta,
            "ultima_respuesta": rag_response
        }

        return Command(update=update_dict)

    except Exception as e:
        print(f"Error inesperado en call_rag_productos_y_coberturas: {e}")
        update_dict = {
            "messages": [
                ToolMessage(
                    content="Error inesperado al procesar la solicitud del usuario.",
                    tool_call_id=tool_call_id
                )
            ],
            "ultima_pregunta": pregunta,
            "ultima_respuesta": "Lo siento gfe, fallé :("
        }

        return Command(update=update_dict)