from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

from LLM_utils import LLM as llm
    
from langchain_community.embeddings import SentenceTransformerEmbeddings
    
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="llama3",base_url="http://192.168.0.10:11434")


from langchain.vectorstores import Chroma

from langchain.schema import Document
import pandas as pd

from sentence_transformers import SentenceTransformer

from langchain.embeddings import HuggingFaceEmbeddings



    
class IR():
    def __init__(self , k = 1):
        # self.embeddings = OllamaEmbeddings(
        #  model="llama3",base_url="http://192.168.0.10:11434")
        
        self.embeddings = HuggingFaceEmbeddings(model_name="hiieu/halong_embedding")
        self.k = k
        docs = []
        data = pd.read_excel("question.xlsx").to_numpy()
        for i in range(len(data)):

            doc = Document(data[i][0] + ' ' + data[i][1])
            docs.append(doc)

        # Create vectorstore from documents
        self.vectorstore = Chroma.from_documents(docs, self.embeddings)

        # Retrieve top 2 documents
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
        
    def get(self, text):
        retrive = self.retriever.invoke(text)
        combined_text = ' '.join(doc.page_content for doc in retrive)
        return combined_text
    
    def get_retriever(self):
        return self.retriever

    def get_vectorstore(self):
        return self.vectorstore
