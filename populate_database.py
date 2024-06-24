import argparse
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from get_embedding import get_embedding
from langchain_community.vectorstores.chroma import Chroma

CHROMA_PATH = "chromaDB"
DATA_PATH = "data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset",action="store_true",help="Reset the database")
    args = parser.parse_args()
    print("HI")
    if args.reset:
        print("Clearing Chroma DB")
        clear_database()
    documents = load_documents()
    chunks = split_documents(documents)
    add_chunks_to_db(chunks)
    
def load_documents():
    documentLoader = PyPDFDirectoryLoader(DATA_PATH)
    return documentLoader.load()

def split_documents(documents:list[Document]):
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex=False,
    )
    return textSplitter.split_documents(documents)

def add_chunks_to_db(documents:list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding()
    )
    chunkWithIds = calculate_chunk_ids(documents)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of Chunks in existing DB : {len(existing_ids)}")

    #Add new chunks
    new_chunks=[]
    for chunk in chunkWithIds:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
            
    
    if len(new_chunks) > 0:
        new_chunk_ids = [new_chunk.metadata["id"] for new_chunk in new_chunks]
        db.add_documents(new_chunks,ids = new_chunk_ids)
        db.persist()
        print(f"Number of new chunks added to DB : {len(new_chunks)}")
    else:
        print("No New Chunks to Add")

def calculate_chunk_ids(chunks:list[Document]):
    # This will create IDs like "data/biography.pdf:6:9"
    # Page Source : Page Number : Chunk ID
    last_page_id = None
    chunk_id = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            chunk_id += 1
        else:
            chunk_id = 0
        last_page_id = current_page_id
        current_chunk_id = f"{current_page_id}:{chunk_id}"
        chunk.metadata["id"] = current_chunk_id
    return chunks



def clear_database():
    if os.path.exists(CHROMA_PATH):
        print("exists chroma directory")
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()