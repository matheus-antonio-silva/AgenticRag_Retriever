"""Orquestrador da aplicação: bootstrap do RAG e loop interativo no terminal."""

import os
import sys

from src.config.settings import load_settings
from src.models.workflow_ia import create_agent_app
from src.services.embeddings import build_embeddings
from src.services.vectorstore import build_vector_store
from src.utils.carrega_documentos import carrega_pdfs


def _validate_pdf_folder(pdf_folder_path: str) -> None:
    """Encerra a aplicação se a pasta de PDFs não existir ou estiver vazia."""
    if not os.path.exists(pdf_folder_path) or not os.listdir(pdf_folder_path):
        print(f"\nErro: a pasta de PDFs '{pdf_folder_path}' está ausente ou vazia.")
        print("Por favor, crie a pasta e adicione seus arquivos PDF de contrato.")
        sys.exit(1)


def _load_documents_for_store(pdf_folder_path: str, vector_store_path: str) -> list:
    """Carrega e divide PDFs apenas quando o índice FAISS ainda não existe."""
    if os.path.exists(vector_store_path):
        print(f"\nStore vetorial encontrado em '{vector_store_path}'. Pulando carregamento/divisão de PDFs.")
        print("Se os contratos foram alterados, exclua a pasta do store vetorial e execute novamente.")
        return []

    docs_for_store = carrega_pdfs(pdf_folder_path)
    if not docs_for_store:
        print("\nSaindo: Nenhum documento foi processado para criar o store vetorial.")
        sys.exit(1)

    return docs_for_store


def _build_vector_store(docs_for_store: list, embeddings_model, vector_store_path: str):
    """Cria ou carrega o índice FAISS, tratando erros de compatibilidade."""
    try:
        return build_vector_store(docs_for_store, embeddings_model, vector_store_path)
    except ValueError as error:
        print(f"\nErro ao inicializar o store vetorial: {error}")
        sys.exit(1)


def _print_sources(final_state: dict) -> None:
    """Exibe os nomes dos arquivos fonte usados na resposta."""
    print("\n--- Fontes dos Documentos Recuperados (para contexto) ---")
    documents = final_state.get("documents")
    if documents:
        sources = {doc.metadata.get("source", "Desconhecida") for doc in documents}
        print(", ".join(sorted(sources)))
    else:
        print("Nenhum documento foi recuperado para esta consulta.")


def run_chat_loop(agent_app) -> None:
    """Executa o loop interativo de perguntas e respostas no terminal."""
    print("\n--- Executando Agente de Contratos ---")

    while True:
        user_query = input(
            "\nDigite sua pergunta sobre os contratos (ou digite 'sair' para encerrar): \n> "
        )

        if user_query.lower() == "sair":
            break

        if not user_query:
            continue

        print("\nProcessando consulta...")
        final_state = agent_app.invoke({"question": user_query})

        print("\n--- Resposta Final ---")
        print(final_state.get("answer", "Nenhuma resposta gerada."))
        _print_sources(final_state)
        print("-" * 50)

    print("\nAgente finalizado.")


def run_agent() -> None:
    """Inicializa o pipeline RAG completo e inicia o chat no terminal.

    Fluxo:
        1. Carrega configurações e valida a pasta de PDFs.
        2. Indexa PDFs (se necessário) e cria/carrega o FAISS.
        3. Monta o agente LangGraph com LLM Groq.
        4. Entra no loop interativo de perguntas.
    """
    print("\n--- Inicializando o Agente de Contratos ---")

    settings = load_settings()
    pdf_folder_path = str(settings.pdf_folder_path)
    vector_store_path = str(settings.vector_store_path)

    _validate_pdf_folder(pdf_folder_path)
    docs_for_store = _load_documents_for_store(pdf_folder_path, vector_store_path)

    embeddings_model = build_embeddings(settings.embedding_model_name)
    vector_store = _build_vector_store(docs_for_store, embeddings_model, vector_store_path)
    agent_app = create_agent_app(vector_store)

    run_chat_loop(agent_app)


if __name__ == "__main__":
    run_agent()
