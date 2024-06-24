from langchain_community.embeddings.ollama import OllamaEmbeddings;
def get_embedding():
    #(jina/jina-embeddings-v2-base-en),(nomic-embed-text)
    embedding = OllamaEmbeddings(model="jina/jina-embeddings-v2-base-en")
    return embedding
