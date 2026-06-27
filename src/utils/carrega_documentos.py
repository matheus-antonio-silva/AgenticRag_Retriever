## Libs para carregar os docuemntos dentro do banco de dados vetorial

## Importacao das Libs
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from typing import List

def carrega_pdfs(folder_path: str) -> List[dict]:

    print(f'Carregando PDFs de: {folder_path}')

    # Cria uma lista para armazenar os documentos carregados

    documents = []

    # Percorre todos os arquivos da pasta

    for filename in os.listdir(folder_path):

        # Verifica se o arquivo tem extensao .pdf

        if filename.lower().endswith('.pdf'):

            # Obtem o caminho completo do arquivo

            file_path = os.path.join(folder_path,filename)

            try:

                # Cria carregador de PDF para o caminho da pasta
                loader = PyPDFLoader(file_path)

                # Carrega as páginas do PDF como documento

                loaded_docs = loader.load()

                # Anexa nome do arquivo aos metadaos de cada documento

                for doc in loaded_docs:
                    doc.metadata['source'] = filename
                
                # Adiciona as paginas carregadas na lista de documentos

                documents.extend(loaded_docs)

                print(f' - {filename} carregado')

            except Exception as e:
                # Em caso de erro
                print(f' - Erro ao carregar {filename}: {e}')
    
    # Verificar se há arquivos carregado

    if not documents:
        print(f'Nenhum documento foi encotrado na pasta : {filename}')
        return []
    
    print(f'Dividindo {len(documents)} páginas de documentos')

    # Cria o divisor de texto com tamanho do texto e overlap

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 200)

    # Divide os documentos em Chunks
    split_docs = text_splitter.split_documents(documents)

    print(f'Criado {len(split_docs)} chunks de documentos')

    return split_docs



# Funcao para criar ou carregar vector store FAISS
def cria_carrega_vectordb(documents : List[dict], embeddings, store_path : str) -> FAISS:

    # Verifica se já existe banco de dados criado

    if os.path.exists(store_path):

        print(f'Carregando Vector Store existente : {store_path}')

        # Carrega o store localmente, com desserializacao

        vector_store = FAISS.load_local(store_path,embeddings,allow_dangerous_deserialization=True)

        print(f'Vector Store carregado')
    
    else :

        if not documents:
            # Se nao houver lista de documentos aparece erro
            raise ValueError('Nenhum documento fornecido para criar o banco de dados vetorial')

        print(f'Criando um novo banco de dados vetorial com os documentos fornecidos, total de documents : {len(documents)}')

        # Cria o banco de dados vetorial com o modelo de embedding e os documentos fornecidos
        vector_store = FAISS.from_documents(documents, embeddings)

        # Salva os documentos no banco de dados vetorial local
        vector_store.save_local(store_path)

        print(f'Banco de dados vetorial criado e salvo.')

    return vector_store


# Funcao para formatar documentos em uma única String
def formata_docs_metadados(docs : List[dict]) -> str:

    # Une Chunks separados

    return '\n\n--\n\n'.join(
    
        # Para cada documento na lista, cria uma string com fonte, página e conteúdo
        f"Fonte: {doc.metadata.get('source', 'Desconhecida')} (Página: {doc.metadata.get('page', 'N/D')})\n\n{doc.page_content}"
        
        # Itera sobre todos os documentos fornecidos
        for doc in docs
    )

