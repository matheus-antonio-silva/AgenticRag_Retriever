"""Nó de retrieval: busca os chunks mais relevantes no índice FAISS."""

from typing import Callable

from src.models.state import AgentState


def build_retrieve_node(vector_store) -> Callable[[AgentState], dict]:
    """Cria o nó que recupera documentos por similaridade semântica.

    Args:
        vector_store: Índice FAISS com os chunks de contrato indexados.

    Returns:
        Função de nó que recebe o estado e retorna os top-5 documentos relevantes.
    """

    def recupera_documentos(state: AgentState) -> dict:
        question = state["question"]
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    return recupera_documentos
