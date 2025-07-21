from typing import Literal
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

class RelevanceCheckData(BaseModel):
    is_relevant: Literal[0, 1] = Field(
        description="1 if the answer is relevant to the question, 0 if not."
    )
    confidence: float = Field(
        description="Confidence level of the relevance judgment, between 0 and 1."
    )

relevance_chat_model = init_chat_model(
    temperature=0,
    model="jacob-ebey/phi4-tools:latest",
    model_provider="ollama",
    base_url="http://192.168.0.10:11434"  
).with_structured_output(RelevanceCheckData)



def cheker(query , response):
    
    prompt = f"""
    You are an AI assistant designed to calculate semantic similarity.

    Your task is to evaluate the semantic similarity between the given question and answer.

    Specifically, focus on:
    - The **concept and semantic concept** of the question and answer.
    - The **use of names**: check if key names (e.g., personal names, organization names) mentioned in the question are correctly and meaningfully referenced in the answer.
    
    Example:
    Question: سلام حالت چطوره؟
    Answer: سلام! چطور می‌توانم در زمینه وام‌های بانک ملت کمکتان کنم؟ 
    → is_relevant: 1, because the answer is a natural and semantically appropriate response to the greeting in the question. The concepts of greeting and well-being match.

    Example:
    Question:  حالت چطوره؟
    Answer: سلام! چطور می‌توانم در زمینه وام‌های بانک ملت کمکتان کنم؟ 
    → is_relevant: 1, because the answer is a natural and semantically appropriate response to the greeting in the question. The concepts of greeting and well-being match.


    Example:
    Question: من یک وام احمد میخوام  
    Answer: وام قرض الحسنه یکی از وام های بانک ملت است  
    → is_relevant: 0, because the name "احمد" in the question is not addressed or connected in the answer, and the semantic concepts do not match.
    
    Example:
    Question: چطور می‌تونم کارت ملی جدید بگیرم؟
    Answer: برای دریافت کارت ملی جدید باید به دفاتر پیشخوان دولت مراجعه کنید.
    → is_relevant: 1, because the answer directly addresses the process of getting a new national ID card.
        
        
    Example:
    Question: بهترین کتاب برای یادگیری زبان انگلیسی چیه؟
    Answer: هوا امروز خیلی خوبه و آفتابی شده.
    → is_relevant: 0, because the answer is unrelated to the topic of English learning and does not match semantically.
    
    Example:
    Question: آیا فردا بانک‌ها باز هستند؟
    Answer: فردا به مناسبت تعطیلات رسمی، اکثر ادارات و بانک‌ها تعطیل هستند.
    → is_relevant: 1, because the answer clearly addresses whether banks will be open, which is the intent of the question.
    
    Example:
    Question: وام گلدون قصد دارم بگیرم. چیکار کنم الان
    Answer: برای دریافت وام شایان نیاز به افتتاح حساب سپرده کوتاه مدت شایان یک دارید. همچنین باید معدل حساب خود را در میانگین 1، 2، 3 و 6 ماه بهبود بخشید. انواع عقد مرابحه شامل عقد جعاله و کارت اعتباری است. سقف وام تا ۳۰۰ میلیون تومان است و رتبه‌های اعتباری مجاز برای دریافت وام شامل E، D، C، B و A می‌شود. برای وام‌های تا ۱۰۰ میلیون تومان نیاز به یک ضامن و بالاتر از این مبلغ نیاز به دو ضامن است. مشتریان با رتبه‌های E و D نیز در حدود قوانین جاری می‌توانند از تسهیلات طرح شایان یک بهره‌مند شوند. نرخ سود برای این وام‌ها ۱۴، ۱۸ و ۲۳ درصد است.
    → is_relevant: 0, because the question is about a specific loan product called "وام گلدون", but the answer explains a different loan plan ("طرح شایان"). There is no clear connection between "وام گلدون" and the "شایان" loan, so the semantic intent does not align.
    
    Now evaluate the following:

    Question: {query}

    Answer: {response}
    """

    response = relevance_chat_model.invoke(prompt)
    
    return response.is_relevant , response.confidence
