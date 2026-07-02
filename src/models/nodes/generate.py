"""Nó de geração: produz a resposta final usando o LLM e o contexto recuperado."""

from typing import Callable

from langchain_core.output_parsers import StrOutputParser

from src.models.state import AgentState


def build_generate_node(rag_prompt, llm) -> Callable[[AgentState], dict]:
    """Cria o nó que gera a resposta a partir do contexto e da pergunta.

    Args:
        rag_prompt: Template de prompt com variáveis ``context`` e ``question``.
        llm: Modelo de linguagem (ex.: ChatGroq) para geração da resposta.

    Returns:
        Função de nó que invoca a chain LCEL ``prompt | llm | parser``.
    """
    rag_chain = rag_prompt | llm | StrOutputParser()

    def gera_resposta(state: AgentState) -> dict:
        answer = rag_chain.invoke(
            {
                "context": state["context"],
                "question": state["question"],
            }
        )
        return {"answer": answer}

    return gera_resposta
