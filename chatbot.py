
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



from LLM_utils import LLM
from information_retrival import IR



class Chat_Rag():
    def __init__(self):
        self.llm = LLM()
        self.retriever = IR().get_retriever()
        self.chat_history = []
        self.documents_chain = self.stuff_documents_chain()

    def history_aware_retriever(self):



        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )


        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt)
        return aware_retriever
    
    def stuff_documents_chain(self):


        qa_system_prompt = (
        """You are a helpful assistant. Please answer **only based on the provided context**.
        Do **not** generate answers beyond the context.
        Limit your response to **one sentence only**.
        If the context does **not** contain enough information to answer, reply with **"I can't answer."**
        All generated texts must be in Persian.

        context:
        {context}
        """
        )

        # Create a prompt template for answering questions
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)


        rag_chain = create_retrieval_chain(self.history_aware_retriever(), question_answer_chain)
        
        return rag_chain
    
    
    def chat_history_updata(self , query , result):
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(SystemMessage(content=result))

    def chat(self , query):
        result = self.documents_chain.invoke({"input": query, "chat_history": self.chat_history})

        
        self.chat_history_updata(query=query , result=result["answer"])
        return result["answer"]