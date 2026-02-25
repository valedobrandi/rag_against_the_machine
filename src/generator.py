from typing import List
from .models import MinimalSource, MinimalAnswer
import ollama

class AnswerGenerator:
    def __init__(self, model_name: str = "qwen3:0.6b", max_workers: int = 1):
        self.model_name = model_name
        self.max_workers = max_workers
    
    def generate_answer(self, question: str, retrieved_sources: List[str]) -> str:
        context = "\n---\n".join(retrieved_sources)
        prompt = (
            f"<|im_start|>system\nYou are a technical assistant. Use the provided context to answer the question briefly.<|im_end|>\n"
            f"<|im_start|>user\nContext: {context}\n\nQuestion: {question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            think=False,
            options={
                "num_thread": 8,
                }
            )
        
        answer = response.get('response', "").strip()

        return answer