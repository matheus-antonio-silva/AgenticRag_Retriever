"""Factory do agente: conecta LLM Groq, prompt RAG e grafo LangGraph."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config.settings import load_settings
from src.models.prompt import RAG_PROMPT_TEMPLATE
from src.models.workflow import build_workflow


def create_agent_app(vector_store):
    """Cria a aplicação do agente de contratos pronta para invocação.

    Configura o LLM Groq, o template de prompt RAG e compila o grafo
    de retrieval e geração sobre o vector store fornecido.

    Args:
        vector_store: Índice FAISS já carregado ou recém-criado.

    Returns:
        Grafo LangGraph compilado para processar perguntas do usuário.
    """
    settings = load_settings()
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model_name=settings.model_name,
        temperature=settings.temperature,
    )
    return build_workflow(
        vector_store=vector_store,
        rag_prompt=rag_prompt,
        llm=llm,
    )
