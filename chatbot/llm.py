import json
import requests
from config import OLLAMA_MODEL
from utils import pretty_print

class LLMClient:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model
        self.base_url = "http://localhost:11434/api/generate"
    
    def generate_response(self, prompt: str) -> str:
        """Send prompt to Ollama and return the response."""
        pretty_print("[OLLAMA]", "Sending prompt to Ollama model...")
        try:
            response = requests.post(
                self.base_url,
                json={"model": self.model, "prompt": prompt},
                stream=True
            )
            
            full_reply = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if 'response' in chunk:
                            full_reply += chunk['response']
                    except json.JSONDecodeError as e:
                        print("JSON decode error:", e)
                        continue
            
            return full_reply
            
        except Exception as e:
            print("Ollama call failed:", e)
            return ""