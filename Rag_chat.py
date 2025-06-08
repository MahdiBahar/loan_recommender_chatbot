from LLM_utils import LLM
from information_retrival import IR
from langchain.prompts import PromptTemplate

class Chat():
    def __init__(self):
        self.ir = IR()
        self.llm = LLM(temp=0)
        
    def prompt_rag(self , context , query):
        qa_system_prompt = (
        "شما یک دستیار متخصص در حوزه وام‌های بانک ملت هستید. "
        "فقط بر اساس اطلاعات موجود در «محتوا» پاسخ دهید و از حدس یا افزودن اطلاعات خودداری کنید.\n\n"
        "دستورالعمل‌ها:\n"
        "1. اگر سؤال کاربر فقط شامل عبارت‌های خوش‌آمدگویی یا احوال‌پرسی باشد، با یک جملهٔ خوش‌آمدگویی رسمی پاسخ دهید و هیچ توضیح اضافه دیگری نده؛\n"
        "   مثال: «سلام! چطور می‌توانم در زمینه وام‌های بانک ملت کمکتان کنم؟»\n"
        "2. پاسخ اصلی را در حداکثر سه جمله حفظ کنید.\n"
        "3. اگر محتوا اطلاعات کافی ندارد، فقط این متن را برگردانید و هیچ توضیح اضافه ای ندهید:\n"
        "   «متأسفم، اطلاعات کافی برای پاسخگویی موجود نیست. تنها در حوزه وام‌های بانک ملت راهنمایی می‌کنم.»\n"
        "4. اگر سؤال کاربر حاوی الفاظ رکیک یا محتوای نامناسب باشد، همان متن بازگردانید.\n"
        "5. تمام خروجی‌ها باید به زبان فارسی و با لحنی رسمی و موجز باشند.\n\n"
        "محتوا:\n{context}\n\n"
        "سؤال کاربر:\n{user_query}\n\n"
        # "### Now process this input:\n"
        #     "Input: \"{user_query}\n\n"
        #     "Output:\n{context}\n\n"
        # "پاسخ:"
    )
        
        prompt = PromptTemplate(template=qa_system_prompt , input_variables=["user_query"  , "context"])
        prompt = prompt.format(context = context , user_query = query)
        return prompt
    
    def QA_with_rag(self , query):
        context = self.ir.get(query)
        prompt = self.prompt_rag(context , query)
        response = self.llm(prompt)
        return response
    

                
