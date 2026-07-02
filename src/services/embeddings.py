"""Factory do modelo de embeddings usado na indexação e busca vetorial."""

from langchain_huggingface import HuggingFaceEmbeddings


def build_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Instancia o modelo de embeddings HuggingFace para o pipeline RAG.

    Args:
        model_name: Identificador do modelo no HuggingFace Hub
            (ex.: ``sentence-transformers/all-MiniLM-L6-v2``).

    Returns:
        Modelo de embeddings configurado para gerar vetores de texto.
    """
    return HuggingFaceEmbeddings(model_name=model_name)
