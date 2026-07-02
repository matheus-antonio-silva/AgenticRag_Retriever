"""Montagem do grafo LangGraph com o pipeline RAG linear."""

from langgraph.graph import END, StateGraph

from src.models.nodes.format_context import formata_contexto
from src.models.nodes.generate import build_generate_node
from src.models.nodes.retrieve import build_retrieve_node
from src.models.state import AgentState


def build_workflow(vector_store, rag_prompt, llm):
    """Monta e compila o grafo retrieve → format_context → generate.

    Args:
        vector_store: Índice FAISS para busca de chunks relevantes.
        rag_prompt: Template de prompt com contexto e pergunta.
        llm: Modelo de linguagem para geração da resposta.

    Returns:
        Aplicação LangGraph compilada, invocável com ``{"question": "..."}``.
    """
    retrieve_node = build_retrieve_node(vector_store)
    generate_node = build_generate_node(rag_prompt, llm)

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("format_context", formata_contexto)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "format_context")
    workflow.add_edge("format_context", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
