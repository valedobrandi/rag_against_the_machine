from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .models import MinimalSource, MinimalAnswer
from transformers import BitsAndBytesConfig

class AnswerGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True),
            low_cpu_mem_usage=True
        )

    def generate_answer(self, question: str, retrieved_sources: List[str]) -> str:
        # 1. Create the Context String
        context = "\n---\n".join(retrieved_sources)
        
        # 2. Build the Prompt (The Instructions)
        prompt = f"""
        You are a technical assistant for the vLLM project. 
        Use the following code snippets and documentation to answer the question.
        If the answer is not in the context, say you don't know.
        
        Context:
        {context}
        
        Question: {question}
        Answer:"""

        # 3. Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=80, 
                do_sample=False,
                num_beams=1,
                use_cache=True, 
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_text.split("Answer:")[-1].strip()