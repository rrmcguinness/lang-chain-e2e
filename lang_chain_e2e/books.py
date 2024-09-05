# Copyright 2024 Google, LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import getpass
import os
from typing import List
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores.base import VectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from lang_chain_e2e.utils import SuppressStdout, get_embeddings_model, get_gemini_flash, print_with_time

DB_DIRECTORY = 'db'
DOCUMENT_SOURCE_DIRECTORY = 'third_party/books'

CHUNK_SIZE=1000
CHUNK_OVERLAP=20
HIDE_SOURCE_DOCUMENTS=False

# Prompt Template
TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=TEMPLATE,
)

def chunk_docs() -> List[Document]:
    """This method loads the PDF files from the source directory
    and uses the explicit PyPDFLoader to ensure each PDF is broken
    down into individual pages for citation."""

    data: List[Document] = []
    print_with_time('Loading PDFs')
    files = os.listdir(DOCUMENT_SOURCE_DIRECTORY)
    for file in files:
        if file.endswith('.pdf'):
            loader = PyPDFLoader(f'{DOCUMENT_SOURCE_DIRECTORY}/{file}')
            original_data = loader.load()
            print(f'{file} - Pages: {len(original_data)}')
            data.extend(original_data)

    print(f'total pages: {len(data)}')

    print_with_time('Splitting PDFs')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_splits = text_splitter.split_documents(data)
    print_with_time(f'Total chunks: {len(all_splits)}')
    return all_splits


def get_persistent_vector_store(chunked_docs, embedding_function) -> VectorStore:
    with SuppressStdout():
        #if not os.path.isdir(DB_DIRECTORY):
            vector_db = Chroma.from_documents(
                documents=chunked_docs,
                embedding=embedding_function, 
                collection_name='books',
                persist_directory=DB_DIRECTORY)
            vector_db.persist()
            return vector_db
        #else:
        #    return Chroma(persist_directory=DB_DIRECTORY, embedding_function=embedding_function)
        

def format_docs(docs):
    """A simple document formatter if the type of document was not chunked"""
    return "\n\n".join(doc.page_content for doc in docs)

        
def main():
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
    llm = get_gemini_flash()
    
    
    embedding_function=get_embeddings_model()

    # 1) Read in all of the PDFs and chunk them into smaller pieces for embeddings
    chunked_docs = chunk_docs()
    # 2) Generate the embeddings and store them in an in-memory vector store
    vector_store = get_persistent_vector_store(chunked_docs, embedding_function)
    # 3) Load a chat large language model to interpret questions using the vector store embeddings
    llm = get_gemini_flash()
    
    # 4) Create the QA Chain using LCEL (Lang Chain Expression Language)
    
    # Start the REPL
    while True:
        query = input("\nQuery: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # search_type="similarity_score_threshold", search_kwargs={"k": 10, "score_threshold": 0.2 }

        qa_chain = (
            {
                "context": vector_store.as_retriever(search_type="mmr") | format_docs,
                "question": RunnablePassthrough()
            }
            | QA_CHAIN_PROMPT
            | llm
            | StrOutputParser()
        ) 
        
        # Call the QA chain to print the response
        resp = qa_chain.invoke(query)
        print(resp)