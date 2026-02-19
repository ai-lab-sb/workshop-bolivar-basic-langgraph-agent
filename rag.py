import sys
sys.path.append(".secrets")

from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from load_credentials import load_credentials
from fs_vector_store import FirestoreVectorStore


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

    # --- LLM ---
    credentials = load_credentials()

    llm = ChatVertexAI(
        model_name="gemini-2.5-flash",
        credentials=credentials,
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


if __name__ == "__main__":
    question = "Me gustaría un seguro para tener mejores sueños"
    answer = rag(question)
    print(f"Pregunta: {question}\n\nRespuesta:\n{answer}")
