from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings



    
class IR():
    def __init__(self , k = 1):

        
        # self.embeddings = HuggingFaceEmbeddings(model_name="hiieu/halong_embedding")
        self.embeddings = HuggingFaceEmbeddings(model_name="/home/mahdi/loan_recommender_chatbot/model/halong_finetuned_APN_1036_frezzed-V0.9")
        self.k = k
        docs = []
        data = pd.read_excel("MEC-question-V0.2.xlsx").to_numpy()
        for i in range(len(data)):

            doc = Document(data[i][0] + ' ' + data[i][1])
            docs.append(doc)


        self.vectorstore = Chroma.from_documents(docs, self.embeddings)


        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
        
    def get(self, text):
        retrive = self.retriever.invoke(text)
        combined_text = ' '.join(doc.page_content for doc in retrive)
        return combined_text
    
    def get_retriever(self):
        return self.retriever

    def get_vectorstore(self):
        return self.vectorstore
