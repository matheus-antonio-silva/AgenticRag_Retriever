"""Esquema de estado compartilhado entre os nós do grafo LangGraph."""

from typing import Sequence, TypedDict


class AgentState(TypedDict):
    """Estado propagado pelo pipeline retrieve → format_context → generate.

    Attributes:
        question: Pergunta do usuário sobre os contratos.
        documents: Chunks recuperados pelo retriever (objetos Document do LangChain).
        context: Texto formatado com fonte, página e conteúdo dos chunks.
        answer: Resposta final gerada pelo LLM.
    """

    question: str
    documents: Sequence[dict]
    context: str
    answer: str
