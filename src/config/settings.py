"""Configurações centralizadas do pipeline RAG (paths, modelos e credenciais)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RAGSettings:
    """Parâmetros imutáveis do agente de contratos.

    Attributes:
        groq_api_key: Chave de API da Groq para o LLM.
        model_name: Modelo de linguagem usado na geração de respostas.
        temperature: Temperatura do LLM (0.0 = respostas determinísticas).
        pdf_folder_path: Pasta com os PDFs de contrato para indexação.
        vector_store_path: Caminho de persistência do índice FAISS.
        embedding_model_name: Modelo HuggingFace para gerar embeddings.
    """

    groq_api_key: str
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    pdf_folder_path: Path = PROJECT_ROOT / "documents"
    vector_store_path: Path = PROJECT_ROOT / "database" / "vectordb"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


def load_settings() -> RAGSettings:
    """Carrega variáveis de ambiente e retorna as configurações do RAG.

    Raises:
        ValueError: Se ``GROQ_API_KEY`` não estiver definida no ambiente.
    """
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY não encontrada no ambiente.")
    return RAGSettings(groq_api_key=api_key)
