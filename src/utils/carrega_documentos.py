"""Utilitários de ingestão de PDFs, persistência FAISS e formatação de contexto."""

import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


def carrega_pdfs(folder_path: str) -> List[dict]:
    """Carrega PDFs de uma pasta, divide em chunks e retorna os trechos indexáveis.

    Args:
        folder_path: Caminho da pasta que contém os arquivos PDF.

    Returns:
        Lista de chunks (documentos LangChain) prontos para embedding.
    """
    print(f"Carregando PDFs de: {folder_path}")

    documents = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)

            try:
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()

                for doc in loaded_docs:
                    doc.metadata["source"] = filename

                documents.extend(loaded_docs)
                print(f" - {filename} carregado")

            except Exception as e:
                print(f" - Erro ao carregar {filename}: {e}")

    if not documents:
        print(f"Nenhum documento foi encontrado na pasta: {folder_path}")
        return []

    print(f"Dividindo {len(documents)} páginas de documentos")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    split_docs = text_splitter.split_documents(documents)

    print(f"Criado {len(split_docs)} chunks de documentos")

    return split_docs


def _valida_dimensao_embedding(vector_store: FAISS, embeddings, store_path: str) -> None:
    """Garante que o modelo de embedding atual é compatível com o índice FAISS salvo."""
    index_dim = vector_store.index.d
    query_dim = len(embeddings.embed_query(" "))
    if index_dim != query_dim:
        raise ValueError(
            f"Incompatibilidade de dimensões do embedding: o índice FAISS usa "
            f"{index_dim} dimensões, mas o modelo atual produz {query_dim}. "
            f"Exclua a pasta '{store_path}' e execute novamente para recriar o índice."
        )


def cria_carrega_vectordb(documents: List[dict], embeddings, store_path: str) -> FAISS:
    """Cria um novo índice FAISS ou carrega um existente do disco.

    Args:
        documents: Chunks de documentos usados na criação do índice (ignorados se já existir).
        embeddings: Modelo de embeddings para indexação e busca.
        store_path: Caminho local onde o índice FAISS é salvo ou carregado.

    Returns:
        Instância FAISS pronta para retrieval.
    """
    if os.path.exists(store_path):
        print(f"Carregando Vector Store existente: {store_path}")

        vector_store = FAISS.load_local(
            store_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        _valida_dimensao_embedding(vector_store, embeddings, store_path)

        print("Vector Store carregado")

    else:
        if not documents:
            raise ValueError(
                "Nenhum documento fornecido para criar o banco de dados vetorial"
            )

        print(
            f"Criando um novo banco de dados vetorial com os documentos fornecidos, "
            f"total de documentos: {len(documents)}"
        )

        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(store_path)

        print("Banco de dados vetorial criado e salvo.")

    return vector_store


def formata_docs_metadados(docs: List[dict]) -> str:
    """Formata chunks recuperados em uma única string de contexto para o LLM.

    Cada chunk inclui fonte, página e conteúdo, separados por delimitadores.
    """
    return "\n\n--\n\n".join(
        f"Fonte: {doc.metadata.get('source', 'Desconhecida')} "
        f"(Página: {doc.metadata.get('page', 'N/D')})\n\n{doc.page_content}"
        for doc in docs
    )
