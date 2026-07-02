"""Serviço de criação e carregamento do banco de dados vetorial FAISS."""

from src.utils.carrega_documentos import cria_carrega_vectordb


def build_vector_store(documents, embeddings, store_path):
    """Cria ou carrega o índice FAISS local a partir dos documentos e embeddings.

    Args:
        documents: Chunks de documentos para indexação (vazio se o índice já existir).
        embeddings: Modelo de embeddings compatível com o índice.
        store_path: Caminho de persistência do vector store.

    Returns:
        Instância FAISS pronta para uso no nó de retrieval.
    """
    return cria_carrega_vectordb(
        documents=documents,
        embeddings=embeddings,
        store_path=store_path,
    )
