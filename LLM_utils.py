from langchain_ollama import OllamaLLM

from langchain_community.embeddings import SentenceTransformerEmbeddings

# Define a function to initialize the model and get a response
def LLM(model="phi4:latest" , temp = 0):

    # llm = Ollama(model="phi4:latest")
    llm = OllamaLLM(model = model, base_url="http://192.168.0.10:11434" , temperature= temp)
    
    # Call the model with the prompt

    
    return llm



def embeddings():
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
    return embedding
    
