from langchain_google_vertexai import ChatVertexAI

from helpers.vector_store import FirestoreVectorStore
from utils.tools_agente_workshop import call_rag_productos_y_coberturas


llm = ChatVertexAI(model_name="gemini-2.5-flash")

vector_store = FirestoreVectorStore(
    project="sb-iadaia-cap-dev",
    database=f"vs-rag-workshop",
    collection="t_seguros_fake_gemini"
)


tools = [call_rag_productos_y_coberturas]
llm_with_tools = llm.bind_tools(tools)