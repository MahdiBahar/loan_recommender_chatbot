from LLM_utils import LLM
from information_retrival import IR
from langchain.prompts import PromptTemplate

class Chat():
    def __init__(self):
        self.ir = IR()
        self.llm = LLM(temp=0)
        
    def prompt_rag(self , context , query):
        qa_system_prompt = (
        """"You are a helpful assistant. Please answer only based on the Context.  
            Do not generate answers beyond the context. Do not include any information, assumption, or explanation that is not explicitly present in the provided context.  
            Limit your response to a maximum of three sentences.  
            If the context does not contain enough information to answer, reply with 'متأسفم، من قادر به پاسخگویی به این درخواست نیستم. لطفاً توجه داشته باشید که فقط در حوزه وام‌های بانک ملت می‌توانم اطلاعات و راهنمایی ارائه دهم.'  
            If the user's question contains swear words, offensive language, or inappropriate content, reply with 'متأسفم، من قادر به پاسخگویی به این درخواست نیستم. لطفاً توجه داشته باشید که فقط در حوزه وام‌های بانک ملت می‌توانم اطلاعات و راهنمایی ارائه دهم.'
            All generated texts must be in Persian.

            Context:  
            {context}

            User Question:  
            {user_query}

            Answer:"
            """)
        
        prompt = PromptTemplate(template=qa_system_prompt , input_variables=["user_query"  , "context"])
        prompt = prompt.format(context = context , user_query = query)
        return prompt
    
    def QA_with_rag(self , query):
        context = self.ir.get(query)
        prompt = self.prompt_rag(context , query)
        response = self.llm(prompt)
        return response
    

                
