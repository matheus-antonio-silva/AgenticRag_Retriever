"""Nó de formatação: converte chunks recuperados em contexto textual para o LLM."""

from src.models.state import AgentState
from src.utils.carrega_documentos import formata_docs_metadados


def formata_contexto(state: AgentState) -> dict:
    """Formata os documentos recuperados em uma string única de contexto.

    Args:
        state: Estado atual do grafo contendo os chunks em ``documents``.

    Returns:
        Dicionário com a chave ``context`` preenchida para o nó de geração.
    """
    documents = state["documents"]
    context = formata_docs_metadados(documents)
    return {"context": context}
